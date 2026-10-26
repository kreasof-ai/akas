"""Validated Spark BF16, channel-scaled FP8, and packed GGUF checkpoints."""

from collections import Counter
from functools import cached_property
from pathlib import Path

from akas.checkpoints.gguf import GGUF
from akas.checkpoints.safetensors import (
    inspect_tensors,
    json_document,
    read_float_tensor,
    read_tensor,
    tensor_index,
)
from akas.models.spark.config import SparkConfig
from akas.text.tokenizer import Tokenizer


class _Tokenizer(Tokenizer):
    """The checkpoint's default single-user, high-thinking conversation."""

    def chat(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        system = (
            "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant."
            "\n`thinking_effort` guides on how much to think in your thinking channel "
            "(not including the response channel), supported values include `low` and `high`."
            "\nNow the system is invoked with `thinking_effort=high`."
            "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>"
        )
        suffix = "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|><think>"
        return (
            self.encode(system, add_bos=False, parse_special=True)
            + self.encode(text, add_bos=False)
            + self.encode(suffix, add_bos=False, parse_special=True)
        )


def tensor_shapes(c, *, output=False):
    shapes = {"token_embd.weight": (c.vocab, c.width), "output_norm.weight": (c.width,)}
    if output:
        shapes["output.weight"] = (c.vocab, c.width)
    for i in range(len(c.layers)):
        p = f"blk.{i}."
        shapes.update(
            {
                p + "attn_norm.weight": (c.width,),
                p + "ffn_norm.weight": (c.width,),
                p + "attn_qkv.weight": (
                    (c.heads + 2 * c.kv_heads) * c.head_dim,
                    c.width,
                ),
                p + "attn_gate.weight": (c.heads, c.width),
                p + "attn_output.weight": (c.width, c.heads * c.head_dim),
                p + "ffn_gate.weight": (c.ff, c.width),
                p + "ffn_up.weight": (c.ff, c.width),
                p + "ffn_down.weight": (c.width, c.ff),
            }
        )
    return shapes


def hf_names(c, *, output=False):
    names = {
        "token_embd.weight": "model.embedding.weight",
        "output_norm.weight": "model.norm.weight",
    }
    if output:
        names["output.weight"] = "lm_head.weight"
    for i in range(len(c.layers)):
        mapped = {
            "attn_norm": "input_layernorm",
            "ffn_norm": "post_attention_layernorm",
            "attn_qkv": "self_attn.q_k_v_proj",
            "attn_gate": "self_attn.g_proj",
            "attn_output": "self_attn.out_proj",
            "ffn_gate": "mlp.gate_proj",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj",
        }
        names.update(
            {
                f"blk.{i}.{k}.weight": f"model.layers.{i}.{v}.weight"
                for k, v in mapped.items()
            }
        )
    return names


class SparkCheckpoint:
    def __init__(self, path, *, tokenizer=True):
        self.path = Path(path)
        self.load_tokenizer = tokenizer
        self.gguf = None
        self.fp8 = False
        if self.path.is_file():
            self.gguf = GGUF(self.path)
            self.config = SparkConfig.from_gguf(self.gguf)
            shapes = tensor_shapes(
                self.config, output="output.weight" in self.gguf.tensors
            )
            if self.gguf.tensors.keys() != shapes.keys():
                raise ValueError(
                    "Spark GGUF tensor inventory differs from architecture"
                )
            for name, shape in shapes.items():
                info = self.gguf.tensors[name]
                if info.shape != shape or (len(shape) == 1 and info.type != 0):
                    raise ValueError(f"invalid Spark tensor: {name}")
            self.names = {n: n for n in shapes}
            return
        data = json_document((self.path / "config.json").read_text())
        self.config = SparkConfig.from_huggingface(data)
        q = data.get("quantization_config")
        self.fp8 = q is not None
        if q is not None:
            groups = q.get("config_groups", {})
            if (
                q.get("quant_method") != "compressed-tensors"
                or q.get("format") != "float-quantized"
                or len(groups) != 1
                or q.get("kv_cache_scheme") is not None
            ):
                raise ValueError("unsupported Spark FP8 quantization")
            group = next(iter(groups.values()))
            w, a = group["weights"], group["input_activations"]
            if (
                w["num_bits"] != 8
                or w["type"] != "float"
                or w["strategy"] != "channel"
                or w["dynamic"]
                or not w["symmetric"]
                or w.get("group_size") is not None
                or a["num_bits"] != 8
                or a["type"] != "float"
                or a["strategy"] != "token"
                or not a["dynamic"]
                or not a["symmetric"]
            ):
                raise ValueError(
                    "Spark FP8 requires static channel weights and dynamic token activations"
                )
        index = tensor_index(self.path)
        self.names = hf_names(self.config, output=not self.config.tied)
        shapes = {
            self.names[n]: s
            for n, s in tensor_shapes(self.config, output=not self.config.tied).items()
        }
        dtypes = {n: "BF16" for n in shapes}
        self.scales = {}
        if self.fp8:
            for n, s in list(shapes.items()):
                if (
                    len(s) == 2
                    and n != "model.embedding.weight"
                    and not n.endswith(("g_proj.weight", "lm_head.weight"))
                ):
                    scale = n[: -len("weight")] + "weight_scale"
                    shapes[scale] = (s[0], 1)
                    dtypes[n] = "F8_E4M3"
                    dtypes[scale] = "BF16"
                    self.scales[n] = scale
        # Some exporters retain the tied head as an explicit duplicate.
        unexpected = set(index) - shapes.keys()
        if unexpected - ({"lm_head.weight"} if self.config.tied else set()):
            raise ValueError(f"unexpected Spark tensors: {sorted(unexpected)[:4]}")
        self.tensors = inspect_tensors(self.path, index, shapes, dtypes)
        for name in self.scales.values():
            import numpy as np

            scale = read_float_tensor(self.tensors[name])
            if not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f"invalid Spark FP8 scale: {name}")

    @cached_property
    def tokenizer(self):
        if not self.load_tokenizer:
            return None
        return (
            _Tokenizer.from_gguf(self.gguf.metadata)
            if self.gguf is not None
            else _Tokenizer.from_pretrained(self.path)
        )

    def read(self, name):
        if self.gguf is not None:
            return self.gguf.array(name)
        name = self.names[name]
        if name not in self.scales:
            return read_float_tensor(self.tensors[name])
        import numpy as np
        from akas.quantization._shared.e4m3 import decode_storage

        bits = read_tensor(self.tensors[name])
        if np.any((bits & 127) == 127):
            raise ValueError(f"nonfinite Spark FP8 weight: {name}")
        return decode_storage(bits) * read_float_tensor(self.tensors[self.scales[name]])

    def graph_weights(self, graph, *, provider, prefill_cache=False):
        if provider != "cuda":
            raise ValueError("Spark currently requires CUDA")
        if self.gguf is not None:
            return PackedWeights(graph, self.gguf, prefill_cache=prefill_cache)
        from akas.checkpoints.partition import Weights

        return Weights(
            graph,
            tensor_shapes(self.config, output=not self.config.tied),
            max_bytes=2**31,
            matrix_dtype="float16",
        )

    def program_upload_options(self, weights, *, model_options):
        contexts = {
            spec.shape[0]
            for name, spec in weights.items()
            if name.startswith("rope.") and name.endswith(".table")
        }
        if len(contexts) > 1:
            raise ValueError("rotary tables disagree on context capacity")
        options = {}
        if contexts:
            options["rope_context"] = contexts.pop()
        if any(name.endswith(".prefill") for name in weights):
            options["prefill_cache"] = True
        return options

    def upload(self, device, *, prefill_cache=False, rope_context=0):
        if device.info["provider"] != "cuda":
            raise ValueError("Spark currently requires CUDA")
        if self.gguf is not None:
            from akas.checkpoints.upload import upload_packed

            result = upload_packed(device, self.gguf)
        else:
            result = {}
        try:
            if self.gguf is None:
                for name in self.names:
                    values = self.read(name)
                    values = values.astype("float16" if values.ndim == 2 else "float32")
                    result[name] = device.from_numpy(values.reshape(-1))
            elif prefill_cache:
                for name, info in self.gguf.tensors.items():
                    if cacheable(name, info):
                        values = self.gguf.array(name).astype("float16")
                        result[name + ".prefill"] = device.from_numpy(
                            values.reshape(-1)
                        )
                        self.gguf.discard(name)
            if rope_context:
                if (
                    type(rope_context) is not int
                    or not 1 <= rope_context <= self.config.max_context
                ):
                    raise ValueError("invalid rotary table context")
                import numpy as np

                for i, (theta, rotary) in enumerate(rotary_parameters(self.config)):
                    angle = np.arange(rope_context, dtype="float64")[
                        :, None
                    ] * theta ** (
                        -2 * np.arange(rotary // 2, dtype="float64")[None, :] / rotary
                    )
                    table = np.concatenate(
                        (np.cos(angle), np.sin(angle)), axis=1
                    ).astype("float32")
                    result[f"rope.{i}.table"] = device.from_numpy(table)
            return result
        except BaseException:
            for value in result.values():
                value.release()
            raise

    def inventory(self):
        if self.gguf is not None:
            return dict(
                format="gguf",
                tensors=len(self.gguf.tensors),
                encodings=dict(Counter(t.encoding for t in self.gguf.tensors.values())),
            )
        return dict(
            format="fp8" if self.fp8 else "bf16",
            tensors=len(self.tensors),
            encodings=dict(Counter(t.dtype for t in self.tensors.values())),
            execution_weights="float16",
            fp8_scales="per-output-channel" if self.fp8 else None,
        )


def rotary_parameters(config):
    """Stable shared tables for the architecture's distinct rotary subspaces."""
    return tuple(dict.fromkeys(zip(config.theta, config.rotary)))


def cacheable(name, info):
    return len(info.shape) == 2 and name not in ("token_embd.weight", "output.weight")


class PackedWeights:
    """Preserve each matrix's GGML encoding and share embedding/output storage."""

    def __init__(self, graph, gguf, *, prefill_cache=False):
        from math import prod
        from akas.quantization.weights.ggml.definition import weight_storage

        self.gguf = gguf
        self.buffers = {}
        self.cached = {}
        for name, info in gguf.tensors.items():
            count, dtype = weight_storage(info.type, prod(info.shape))
            self.buffers[name] = graph.buffer(name, (count,), dtype, role="weight")
            if prefill_cache and cacheable(name, info):
                self.cached[name] = graph.buffer(
                    name + ".prefill", (prod(info.shape),), "float16", role="weight"
                )

    def __getitem__(self, name):
        return self.buffers[name]

    def matrix(self, name):
        return [
            (
                self.cached.get(name, self.buffers[name]),
                0,
                self.gguf.tensors[name].shape[0],
            )
        ]

    def parameters(self, name):
        return {
            "ggml": True,
            "type": 1 if name in self.cached else self.gguf.tensors[name].type,
        }
