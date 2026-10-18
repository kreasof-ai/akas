"""Small packed LFM2 MoE checkpoint and an independent model equation oracle."""

from pathlib import Path

import numpy as np

from akas.checkpoints.gguf import GGUF
from support.diagnostic import fixture as dense_fixture
from support.qwen_gguf import write
from support.numpy_reference import Reference, _HalfWeights


def fixture(path, *, dense_layers=1, context=384, mutate=None):
    path = Path(path)
    base_path = path.with_suffix(".dense.gguf")
    base = GGUF(dense_fixture(base_path, context=context))
    metadata = {
        name.replace("lfm2.", "lfm2moe."): value
        for name, value in base.metadata.items()
    }
    metadata.update(
        {
            "general.architecture": "lfm2moe",
            "lfm2moe.expert_count": 4,
            "lfm2moe.expert_used_count": 2,
            "lfm2moe.expert_feed_forward_length": 256,
            "lfm2moe.leading_dense_block_count": dense_layers,
            "lfm2moe.expert_gating_func": 2,
        }
    )
    tensors = {
        name: (info.shape, info.type, base.packed(name).copy())
        for name, info in base.tensors.items()
    }
    rng = np.random.default_rng(413)
    for layer in range(dense_layers, 2):
        prefix = f"blk.{layer}."
        for part in ("gate", "up", "down"):
            del tensors[prefix + "ffn_" + part + ".weight"]
            shape = (4, 256, 256)
            blocks = rng.integers(0, 256, (np.prod(shape) // 32, 18), dtype="uint8")
            blocks[:, :2] = (
                rng.uniform(0.001, 0.004, (len(blocks), 1))
                .astype("float16")
                .view("uint8")
            )
            tensors[prefix + "ffn_" + part + "_exps.weight"] = (shape, 2, blocks)
        tensors[prefix + "ffn_gate_inp.weight"] = (
            (4, 256),
            0,
            rng.normal(0, 0.1, (4, 256)).astype("float32"),
        )
        tensors[prefix + "exp_probs_b.bias"] = (
            (4,),
            0,
            np.array([0.1, -0.2, 0.05, 0], dtype="float32"),
        )
    if mutate:
        mutate(metadata, tensors)
    write(path, metadata, tensors)
    del base
    base_path.unlink()
    return path


class Oracle(Reference):
    def __init__(self, path, *, provider, context=64):
        from akas.models.lfm2.weights import LFM2MoECheckpoint

        loaded = LFM2MoECheckpoint(path, tokenizer=False)
        self.config = loaded.config
        self.weights = {
            name: loaded.gguf.array(name).copy() for name in loaded.gguf.tensors
        }
        self.half_weights = _HalfWeights(self.weights)
        self.context = context
        self.provider = provider
        self.reset()
        if provider == "webgpu":
            self.caches = {
                i: tuple(np.zeros_like(cache, dtype="float32") for cache in pair)
                for i, pair in self.caches.items()
            }

    def feed_forward(self, layer, x, project):
        if layer < self.config.dense_layers:
            return super().feed_forward(layer, x, project)
        p = f"blk.{layer}."
        logits = project(p + "ffn_gate_inp.weight", x)
        probabilities = 1 / (1 + np.exp(-logits))
        selected = np.argsort(
            -(probabilities + self.weights[p + "exp_probs_b.bias"]),
            axis=-1,
            kind="stable",
        )[:, : self.config.top_k]
        weights = np.take_along_axis(probabilities, selected, axis=1)
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-6
        out = np.zeros_like(x)

        def operand(value):
            return (
                value.astype("float16").astype("float32")
                if self.provider == "cuda"
                else value
            )

        for row, experts in enumerate(selected):
            for rank, expert in enumerate(experts):
                gate, up = (
                    operand(self.weights[p + "ffn_" + name + "_exps.weight"][expert])
                    @ operand(x[row])
                    for name in ("gate", "up")
                )
                activated = gate / (1 + np.exp(-gate)) * up
                value = operand(
                    self.weights[p + "ffn_down_exps.weight"][expert]
                ) @ operand(activated)
                out[row] += value * weights[row, rank]
        return out
