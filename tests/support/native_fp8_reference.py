"""Bounded-memory NumPy equations for native LFM2 MoE qualification."""

from collections import OrderedDict
from collections.abc import Mapping

import numpy as np

from akas.checkpoints.safetensors import read_tensor, read_float_tensor
from akas.quantization._shared.e4m3 import decode_storage, encode_storage
from support.numpy_reference import Reference


class Weights(Mapping):
    """Decode individual source matrices on demand, including expert banks."""

    def __init__(self, checkpoint, cache_bytes=256 * 1024**2):
        self.checkpoint, self.limit = checkpoint, cache_bytes
        self.cache, self.bytes = OrderedDict(), 0

    def __iter__(self):
        return iter(self.checkpoint.groups)

    def __len__(self):
        return len(self.checkpoint.groups)

    def __contains__(self, name):
        return name in self.checkpoint.groups

    def __getitem__(self, name):
        sources = self.checkpoint.groups[name]
        if len(sources) == 1:
            return self.read(name, sources[0]).reshape(self.checkpoint.shapes[name])
        owner = self

        class Bank:
            def __getitem__(self, expert):
                return owner.read(name, sources[int(expert)])

        return Bank()

    def read(self, name, source):
        if source in self.cache:
            self.cache.move_to_end(source)
            return self.cache[source]
        checkpoint = self.checkpoint
        if name in checkpoint.scales:
            values = decode_storage(read_tensor(checkpoint.tensors[source])).astype("float64")
            scales = read_float_tensor(checkpoint.tensors[source[:-6] + "weight_scale"])
            values *= np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)[:values.shape[0], :values.shape[1]]
        else:
            values = read_float_tensor(checkpoint.tensors[source])
            if checkpoint.conversions[name] == "float16":
                values = values.astype("float16").astype("float32")
        if values.nbytes <= self.limit:
            while self.bytes + values.nbytes > self.limit:
                _, previous = self.cache.popitem(last=False)
                self.bytes -= previous.nbytes
            self.cache[source] = values
            self.bytes += values.nbytes
        return values


class NativeFP8Reference(Reference):
    def __init__(self, checkpoint, *, context=64, cache_bytes=256 * 1024**2,
                 projection_half=False, contribution_dtype="float32", activation_dtype="float32",
                 residual_dtype="float32", attention_half=False, attention_decode_half=False):
        if contribution_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("unsupported expert contribution storage")
        if activation_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("unsupported expert activation storage")
        if residual_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("unsupported residual storage")
        self.checkpoint, self.config, self.context = checkpoint, checkpoint.config, context
        self.weights = Weights(checkpoint, cache_bytes)
        self.projection_half = projection_half
        self.contribution_dtype = contribution_dtype
        self.activation_dtype = activation_dtype
        self.residual_dtype = residual_dtype
        self.attention_half = attention_half
        self.attention_decode_half = attention_decode_half
        self.reset()

    @staticmethod
    def rounded(values, dtype):
        if dtype == "bfloat16":
            from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

            return decode_bfloat16(encode_bfloat16(values))
        return values.astype(dtype).astype("float32")

    @staticmethod
    def quantize(values):
        shape = values.shape
        blocks = values.astype("float32").reshape(*shape[:-1], shape[-1] // 128, 128)
        scales = np.maximum(np.max(np.abs(blocks), axis=-1, keepdims=True), np.float32(1e-12)) / np.float32(448)
        return (decode_storage(encode_storage(blocks / scales)) * scales).reshape(shape)

    def project(self, name, values, *, single=False):
        if name in self.checkpoint.scales:
            values = self.quantize(values)
        elif (self.projection_half and not single
              and self.checkpoint.conversions[name] == "float16"):
            values = values.astype("float16").astype("float32")
        return (values.astype("float64") @ self.weights[name].astype("float64").T).astype("float32")

    def forward(self, tokens, *, observer=None):
        from support.oracle import tensor_core_attention

        residual = (None if self.residual_dtype == "float32" else
                    lambda x, y: self.rounded(x.astype("float32") + y.astype("float32"), self.residual_dtype))
        return self._chunk(np.asarray(tokens, dtype="int32"), projector=self.project,
                           observer=observer, residual=residual,
                           attention=tensor_core_attention if (self.attention_half and len(tokens) > 1
                               or self.attention_decode_half and len(tokens) == 1) else None)

    def feed_forward(self, layer, values, project):
        if layer < self.config.dense_layers:
            return super().feed_forward(layer, values, project)
        p = f"blk.{layer}."
        scores = 1 / (1 + np.exp(-project(p + "ffn_gate_inp.weight", values)))
        ids = np.argsort(-(scores + self.weights[p + "exp_probs_b.bias"]), axis=-1, kind="stable")[:, :self.config.top_k]
        probability = np.take_along_axis(scores, ids, axis=1)
        if self.config.normalize_routes:
            probability /= probability.sum(axis=-1, keepdims=True) + 1e-6
        probability *= self.config.routing_scale
        source = self.quantize(values).astype("float64")
        contributions = np.empty((*ids.shape, values.shape[1]), dtype="float32")
        for expert in np.unique(ids):
            owners, ranks = np.nonzero(ids == expert)
            selected = source[owners]
            gate, up = ((selected @ self.weights[p + f"ffn_{part}_exps.weight"][expert].T).astype("float32")
                        for part in ("gate", "up"))
            gate, up = (self.rounded(value, self.activation_dtype) for value in (gate, up))
            activated = self.quantize(self.rounded(gate / (1 + np.exp(-gate)) * up,
                                                 self.activation_dtype)).astype("float64")
            product = (
                activated @ self.weights[p + "ffn_down_exps.weight"][expert].T
            ).astype("float32")
            product = self.rounded(product, self.contribution_dtype)
            contributions[owners, ranks] = product
        out = np.zeros_like(values)
        for rank in range(self.config.top_k):
            out += contributions[:, rank] * probability[:, rank, None]
        return out
