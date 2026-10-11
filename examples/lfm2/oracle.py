"""Independent NumPy model oracle for small diagnostic checkpoints.

Reuses the existing NumPy LFM equations, executing one token at a time with
F32 projections. FP8 rounding uses a numeric E4M3 lookup table rather than any
Tensor codec/kernel helper. This module is a correctness tool, not a runtime.
"""

import numpy as np
from benchmarks.lfm2.numpy_reference import Reference, _HalfWeights
from akas.models.lfm2.weights import ReferenceCheckpoint


def fp8_round(values):
    bits = values.astype("float32").view("uint32")
    bf16 = ((bits + 32767 + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view("float32")
    scale = np.maximum(
        np.max(np.abs(bf16), axis=-1, keepdims=True), np.float32(1e-12)
    ) / np.float32(448)
    exponent = np.arange(127, dtype="int32") // 8
    fraction = np.arange(127, dtype="int32") % 8
    table = np.where(
        exponent == 0, fraction / 512, (1 + fraction / 8) * 2.0 ** (exponent - 7)
    ).astype("float32")
    normalized = np.abs(bf16) / scale
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
    return np.copysign(table[indices], bf16) * scale


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
