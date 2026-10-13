"""Independent NumPy model oracle for small diagnostic checkpoints.

Reuses the existing NumPy LFM equations, executing one token at a time with
F32 projections. FP8 rounding uses a numeric E4M3 lookup table rather than any
Tensor codec/kernel helper. This module is a correctness tool, not a runtime.
"""

import numpy as np
from support.numpy_reference import Reference, _HalfWeights
from akas.models.lfm2.weights import ReferenceCheckpoint


def e4m3_values():
    exponent = np.arange(127, dtype="int32") // 8
    fraction = np.arange(127, dtype="int32") % 8
    return np.where(
        exponent == 0, fraction / 512, (1 + fraction / 8) * 2.0 ** (exponent - 7)
    ).astype("float32")


def fp8_round(values):
    source = values.astype("float32")
    scale = np.maximum(
        np.max(np.abs(source), axis=-1, keepdims=True), np.float32(1e-12)
    ) / np.float32(448)
    table = e4m3_values()
    normalized = np.abs(source) / scale
    upper = np.minimum(np.searchsorted(table, normalized), 126)
    lower = np.maximum(upper - 1, 0)
    up_distance, lo_distance = (
        np.abs(table[upper] - normalized),
        np.abs(table[lower] - normalized),
    )
    choose_up = (up_distance < lo_distance) | (
        (up_distance == lo_distance) & ((upper % 2) == 0)
    )
    indices = np.where(choose_up, upper, lower)
    return np.copysign(table[indices], source) * scale


class Cache:
    def __init__(self, shape, dtype):
        self.data = np.zeros(shape, dtype="float32")
        self.dtype = dtype

    def __setitem__(self, index, values):
        self.data[index] = (
            fp8_round(values)
            if self.dtype == "fp8"
            else values.astype(self.dtype).astype("float32")
        )

    def __getitem__(self, index):
        return self.data[index]


class Oracle(Reference):
    def __init__(self, path, *, context=4096, kv_dtype="float16"):
        checkpoint = ReferenceCheckpoint(path, tokenizer=False)
        self.config = checkpoint.config
        self.context, self.kv_dtype = context, kv_dtype
        self.weights = {name: checkpoint.read(name) for name in checkpoint.names}
        self.half_weights = _HalfWeights(self.weights)
        self.reset()

    def reset(self):
        super().reset()
        c = self.config
        self.caches = {
            i: (
                Cache((self.context, c.kv_heads, c.head_dim), self.kv_dtype),
                Cache((self.context, c.kv_heads, c.head_dim), self.kv_dtype),
            )
            for i, kind in enumerate(c.layers)
            if kind == "attention"
        }

    def forward(self, tokens):
        if not len(tokens):
            raise ValueError("oracle requires nonempty token IDs")
        for token in tokens:
            result = self._chunk(np.array([token], dtype="int32"))
        return result


def tensor_core_attention(q, keys, values, start):
    """Independent tiled online softmax with FP16 MMA operands and FP32 sums."""
    q = q.transpose(1, 0, 2).astype("float16").astype("float32")
    keys = keys.astype("float16").astype("float32")
    values = values.astype("float16").astype("float32")
    result = np.zeros_like(q)
    for begin in range(0, q.shape[1], 32):
        queries = q[:, begin : begin + 32]
        maximum = np.full((*queries.shape[:2], 1), -np.inf, dtype="float32")
        normalizer = np.zeros_like(maximum)
        accum = np.zeros_like(queries)
        for tile in range(0, keys.shape[1], 64):
            score = (
                queries @ keys[:, tile : tile + 64].transpose(0, 2, 1)
            ) * np.float32(q.shape[-1] ** -0.5)
            causal = (
                np.arange(tile, min(tile + 64, keys.shape[1]))[None, :]
                <= np.arange(start + begin, start + begin + queries.shape[1])[:, None]
            )
            score = np.where(causal[None], score, -np.inf)
            next_max = np.maximum(maximum, score.max(axis=-1, keepdims=True))
            correction = np.exp(maximum - next_max)
            probability = np.exp(score - next_max)
            normalizer = normalizer * correction + probability.sum(
                axis=-1, keepdims=True
            )
            accum = (
                accum * correction
                + probability.astype("float16").astype("float32")
                @ values[:, tile : tile + 64]
            )
            maximum = next_max
        result[:, begin : begin + queries.shape[1]] = accum / normalizer
    return result


class OptimizedOracle(Oracle):
    """Match declared projection/attention precision without importing kernels."""

    def __init__(
        self,
        path,
        *,
        provider,
        batch,
        chunk,
        minimum_mma_groups=None,
        maximum_gemv_rows=31,
        **options,
    ):
        super().__init__(path, **options)
        self.weights = {
            name: value.astype("float16").astype("float32")
            if value.ndim == 2 and not name.endswith("conv.weight")
            else value
            for name, value in self.weights.items()
        }
        self.half_weights = _HalfWeights(self.weights)
        self.provider, self.batch, self.chunk = provider, batch, chunk
        self.maximum_gemv_rows = maximum_gemv_rows
        self.minimum_mma_groups = (
            (512 if provider == "webgpu" else 32)
            if minimum_mma_groups is None
            else minimum_mma_groups
        )

    def projection_precision(self, rows):
        def half(name):
            columns = self.weights[name].shape[0]
            # Match the declared register-tile policy, independently of kernel helpers.
            m, n = (16, 32) if self.provider == "webgpu" else (32, 64)
            groups = ((rows + m - 1) // m) * ((columns + n - 1) // n)
            underfilled = groups < self.minimum_mma_groups and (
                self.provider == "webgpu" or rows <= self.maximum_gemv_rows
            )
            return rows > 8 and not underfilled

        return half

    def load_state(self, session, slot=0):
        """Check a forward pass from identical inputs and quantized history.

        Separate quantized trajectories can diverge when small F32 differences
        cross rounding thresholds. This imports state, never logits or operator
        results, and decodes E4M3 through an independent numeric table.
        """
        self.reset()
        self.position = int(session.positions[slot])
        for layer, kind in enumerate(self.config.layers):
            prefix = f"blk.{layer}."
            if kind == "conv":
                self.states[layer] = (
                    session.buffers[prefix + "conv_state"].to_numpy()[slot].copy()
                )
                continue
            for cache, name, meta in zip(
                self.caches[layer], ("kc", "vc"), ("ks", "vs")
            ):
                values = session.buffers[prefix + name].to_numpy()[slot]
                scale = session.buffers[prefix + meta].to_numpy()[slot]
                if self.kv_dtype == "fp8":
                    bits = values.view("uint8").reshape(
                        self.config.kv_heads, self.context, self.config.head_dim
                    )
                    values = e4m3_values()[bits & 127] * np.where(
                        bits & 128, -1, 1
                    ).astype("float32")
                values = (
                    values.astype("float32")
                    * np.repeat(scale, 128, axis=-1)[..., : self.config.head_dim]
                )
                cache.data[:] = values.transpose(1, 0, 2)

    def forward(self, tokens, *, prefill=True):
        if not len(tokens):
            raise ValueError("oracle requires nonempty token IDs")
        chunk = self.chunk if prefill else 1
        for begin in range(0, len(tokens), chunk):
            result = self._chunk(
                np.asarray(tokens[begin : begin + chunk], dtype="int32"),
                projection_half=self.projection_precision(self.batch * chunk),
                tail_half=self.projection_precision(self.batch) if chunk > 1 else None,
                attention=tensor_core_attention
                if self.provider == "cuda" and chunk > 1
                else None,
                head_half=self.projection_precision(self.batch),
            )
        return result
