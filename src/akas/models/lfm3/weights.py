"""Validated BF16, block-scaled FP8 and packed GGUF LFM3 checkpoints."""

from collections import Counter
from functools import cached_property
from math import prod
from pathlib import Path

from akas.checkpoints.gguf import GGUF
from akas.checkpoints.safetensors import (
    inspect_tensors,
    json_document,
    tensor_index,
    read_float_tensor,
    read_tensor,
)
from akas.models.lfm3.config import LFM3Config
from akas.text.tokenizer import Tokenizer


class _Tokenizer(Tokenizer):
    """Official single-user conversation with the template's default thinking mode."""

    def __init__(self, rules):
        super().__init__(rules)
        self.generation_prefix = "<|im_start|>assistant\n<think>"


def tensor_shapes(c):
    shapes = {"token_embd.weight": (c.vocab, c.width), "output_norm.weight": (c.width,)}
    if not c.tied:
        shapes["output.weight"] = (c.vocab, c.width)
    for i, kind in enumerate(c.layers):
        p = f"blk.{i}."
        shapes.update(
            {
                p + n + ".weight": s
                for n, s in {
                    "attn_norm": (c.width,),
                    "ffn_norm": (c.width,),
                    "ffn_gate": (c.ff, c.width),
                    "ffn_up": (c.ff, c.width),
                    "ffn_down": (c.width, c.ff),
                }.items()
            }
        )
        if c.sandwich:
            shapes[p + "post_attention_norm.weight"] = (c.width,)
            shapes[p + "post_ffw_norm.weight"] = (c.width,)
        if kind == "conv":
            shapes.update(
                {
                    p + "shortconv." + n + ".weight": s
                    for n, s in {
                        "in_proj": (3 * c.width, c.width),
                        "out_proj": (c.width, c.width),
                        "conv": (c.width, c.conv),
                    }.items()
                }
            )
        else:
            shapes.update(
                {
                    p + n + ".weight": s
                    for n, s in {
                        "attn_q": (c.heads * c.head_dim, c.width),
                        "attn_k": (c.kv_heads * c.head_dim, c.width),
                        "attn_v": (c.kv_heads * c.head_dim, c.width),
                        "attn_output": (c.width, c.heads * c.head_dim),
                        "attn_gate": (c.heads, c.width),
                        "attn_q_norm": (c.head_dim,),
                        "attn_k_norm": (c.head_dim,),
                    }.items()
                }
            )
    return shapes


def hf_names(c):
    names = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.embedding_norm.weight",
    }
    if not c.tied:
        names["output.weight"] = "lm_head.weight"
    for i, kind in enumerate(c.layers):
        mapped = {
            "attn_norm": "operator_norm",
            "ffn_norm": "ffn_norm",
            "ffn_gate": "feed_forward.w1",
            "ffn_up": "feed_forward.w3",
            "ffn_down": "feed_forward.w2",
        }
        if c.sandwich:
            mapped.update(
                post_attention_norm="post_operator_norm", post_ffw_norm="post_ffn_norm"
            )
        if kind == "conv":
            mapped.update(
                {"shortconv." + n: "conv." + n for n in ("conv", "in_proj", "out_proj")}
            )
        else:
            mapped.update(
                {
                    "attn_" + n: "self_attn." + hf
                    for n, hf in {
                        "q": "q_proj",
                        "k": "k_proj",
                        "v": "v_proj",
                        "output": "o_proj",
                        "gate": "gate_proj",
                        "q_norm": "q_norm",
                        "k_norm": "k_norm",
                    }.items()
                }
            )
        names.update(
            {
                f"blk.{i}.{n}.weight": f"model.layers.{i}.{hf}.weight"
                for n, hf in mapped.items()
            }
        )
    return names


class LFM3Checkpoint:
    def __init__(self, path, *, tokenizer=True):
        self.path, self.load_tokenizer = Path(path), tokenizer
        self.gguf, self.fp8, self.scales = None, False, {}
        if self.path.is_file():
            self.gguf = GGUF(self.path)
            self.config = LFM3Config.from_gguf(self.gguf)
            shapes = {
                **tensor_shapes(self.config),
                "rope_freqs.weight": (self.config.head_dim // 2,),
            }
            if shapes.keys() != self.gguf.tensors.keys():
                raise ValueError("LFM3 GGUF tensor inventory differs from architecture")
            for name, shape in shapes.items():
                info = self.gguf.tensors[name]
                if info.shape != shape or (
                    (len(shape) == 1 or name.endswith("conv.weight")) and info.type != 0
                ):
                    raise ValueError(f"invalid LFM3 tensor: {name}")
            self.names = {n: n for n in tensor_shapes(self.config)}
            return
        data = json_document((self.path / "config.json").read_text())
        self.config = LFM3Config.from_huggingface(data)
        q = data.get("quantization_config")
        self.fp8 = q is not None
        if self.fp8:
            groups = q.get("config_groups", {})
            if (
                q.get("quant_method") != "compressed-tensors"
                or q.get("format") != "float-quantized"
                or len(groups) != 1
                or q.get("kv_cache_scheme") is not None
            ):
                raise ValueError("unsupported LFM3 FP8 quantization")
            group = next(iter(groups.values()))
            w, a = group.get("weights", {}), group.get("input_activations", {})
            if (
                w.get("num_bits") != 8
                or w.get("type") != "float"
                or w.get("strategy") != "block"
                or w.get("block_structure") != [128, 128]
                or w.get("dynamic") is not False
                or w.get("symmetric") is not True
                or a.get("num_bits") != 8
                or a.get("type") != "float"
                or a.get("strategy") != "group"
                or a.get("group_size") != 128
                or a.get("dynamic") is not True
                or a.get("symmetric") is not True
            ):
                raise ValueError(
                    "LFM3 FP8 requires static 128x128 weights and dynamic group activations"
                )
        self.names = hf_names(self.config)
        shapes = {self.names[n]: s for n, s in tensor_shapes(self.config).items()}
        for name in list(shapes):
            if name.endswith("conv.conv.weight"):
                shapes[name] = (self.config.width, 1, self.config.conv)
        dtypes = {n: "BF16" for n in shapes}
        if self.fp8:
            for name, shape in list(shapes.items()):
                if (
                    len(shape) == 2
                    and name != "model.embed_tokens.weight"
                    and name != "lm_head.weight"
                    and ".conv." not in name
                    and not name.endswith("self_attn.gate_proj.weight")
                ):
                    scale = name[:-6] + "weight_scale"
                    shapes[scale] = ((shape[0] + 127) // 128, (shape[1] + 127) // 128)
                    dtypes[name], dtypes[scale] = "F8_E4M3", "BF16"
                    self.scales[name] = scale
        index = tensor_index(self.path)
        if index.keys() != shapes.keys():
            raise ValueError(
                f"LFM3 safetensors inventory differs: {sorted(index.keys() ^ shapes.keys())[:6]}"
            )
        self.tensors = inspect_tensors(self.path, index, shapes, dtypes)
        import numpy as np

        for name in self.scales.values():
            scale = read_float_tensor(self.tensors[name])
            if not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f"invalid LFM3 FP8 scale: {name}")

    @cached_property
    def tokenizer(self):
        if not self.load_tokenizer:
            return None
        return (
            _Tokenizer.from_gguf(self.gguf.metadata)
            if self.gguf
            else _Tokenizer.from_pretrained(self.path)
        )

    def read(self, name):
        if self.gguf is not None:
            return self.gguf.array(name)
        hf = self.names[name]
        if hf not in self.scales:
            return read_float_tensor(self.tensors[hf]).reshape(
                tensor_shapes(self.config)[name]
            )
        from akas.quantization._shared.e4m3 import decode_storage

        bits = self._fp8_bits(hf)
        scale = read_float_tensor(self.tensors[self.scales[hf]])
        return (
            decode_storage(bits)
            * scale.repeat(128, 0).repeat(128, 1)[: bits.shape[0], : bits.shape[1]]
        )

    def _fp8_bits(self, name):
        import numpy as np

        bits = read_tensor(self.tensors[name])
        if np.any((bits & 127) == 127):
            raise ValueError(f"nonfinite LFM3 FP8 weight: {name}")
        return bits

    def packed_layouts(self, layout):
        if layout not in ("raw", "planar"):
            raise ValueError("LFM3 Q4 layout must be raw or planar")
        return (
            {
                n: layout
                for n, t in self.gguf.tensors.items()
                if layout != "raw"
                and t.type == 2
                and len(t.shape) == 2
                and t.shape[-1] % 256 == 0
                and n not in ("token_embd.weight", "output.weight")
            }
            if self.gguf
            else {}
        )

    def graph_weights(self, graph, *, provider, q4_layout="raw"):
        if provider != "cuda":
            raise ValueError("LFM3 currently requires CUDA")
        if self.gguf is not None:
            from akas.checkpoints.partition import PackedWeights

            return PackedWeights(
                graph, self.gguf, layouts=self.packed_layouts(q4_layout)
            )
        return FloatingWeights(graph, self)

    def program_upload_options(self, weights, *, model_options):
        fine = {
            b.shape[0]
            for n, b in weights.items()
            if n.startswith("rope.") and n.endswith(".fine")
        }
        coarse = {
            b.shape[0]
            for n, b in weights.items()
            if n.startswith("rope.") and n.endswith(".coarse")
        }
        if len(fine) != 1 or len(coarse) != 1:
            raise ValueError("LFM3 compact rotary tables disagree on geometry")
        return dict(
            rope_layout=dict(stride=fine.pop(), blocks=coarse.pop()),
            weight_arena=model_options.get("weight_arena", False),
            q4_layout=model_options.get("q4_layout", "raw"),
        )

    def upload(self, device, *, rope_layout=None, weight_arena=False, q4_layout="raw"):
        import numpy as np

        if device.info["provider"] != "cuda":
            raise ValueError("LFM3 currently requires CUDA")
        from akas.checkpoints.upload import upload_packed

        result = (
            upload_packed(
                device,
                self.gguf,
                arena=weight_arena,
                layouts=self.packed_layouts(q4_layout),
            )
            if self.gguf
            else {}
        )
        try:
            if self.gguf is None:
                for name, shape in tensor_shapes(self.config).items():
                    hf = self.names[name]
                    if hf in self.scales:
                        result[name] = device.from_numpy(self._fp8_bits(hf))
                        scale = read_float_tensor(self.tensors[self.scales[hf]])
                        result[name + ".scale"] = device.from_numpy(
                            scale, dtype="bfloat16"
                        )
                    else:
                        dtype = (
                            "float16"
                            if len(shape) == 2 and not name.endswith("conv.weight")
                            else "float32"
                        )
                        result[name] = device.from_numpy(
                            self.read(name).astype(dtype).reshape(-1)
                        )
            if rope_layout:
                stride, blocks = rope_layout["stride"], rope_layout["blocks"]
                if (
                    type(stride) is not int
                    or type(blocks) is not int
                    or stride < 1
                    or blocks < 1
                    or stride > self.config.max_context
                    or (blocks - 1) * stride >= self.config.max_context
                ):
                    raise ValueError("invalid LFM3 rotary geometry")
                for i, (theta, rotary) in enumerate(rotary_parameters(self.config)):
                    freq = theta ** (
                        -2
                        * np.arange(rotary // 2, dtype="float64")
                        / self.config.head_dim
                    )
                    for label, pos in (
                        ("fine", np.arange(stride, dtype="float64")),
                        ("coarse", np.arange(blocks, dtype="float64") * stride),
                    ):
                        angle = pos[:, None] * freq[None, :]
                        table = np.concatenate(
                            (np.cos(angle), np.sin(angle)), axis=1
                        ).astype("float32")
                        result[f"rope.{i}.{label}"] = device.from_numpy(table)
            return result
        except BaseException:
            for b in result.values():
                b.release()
            raise

    def inventory(self):
        return dict(
            format="gguf" if self.gguf else "fp8" if self.fp8 else "bf16",
            tensors=len(self.gguf.tensors) if self.gguf else len(self.tensors),
            encodings=dict(Counter(t.type for t in self.gguf.tensors.values()))
            if self.gguf
            else dict(Counter(t.dtype for t in self.tensors.values())),
            execution_weights="packed"
            if self.gguf
            else "fp8-block"
            if self.fp8
            else "float16",
        )


def rotary_parameters(c):
    return tuple(
        dict.fromkeys(
            (t, r) for t, r, k in zip(c.theta, c.rotary, c.layers) if k != "conv"
        )
    )


class FloatingWeights:
    def __init__(self, graph, checkpoint):
        self.buffers, self.scales = {}, {}
        self.shapes = tensor_shapes(checkpoint.config)
        for name, shape in self.shapes.items():
            if checkpoint.names[name] in checkpoint.scales:
                self.buffers[name] = graph.buffer(name, shape, "uint8", role="weight")
                self.scales[name] = graph.buffer(
                    name + ".scale",
                    ((shape[0] + 127) // 128, shape[1] // 128),
                    "bfloat16",
                    role="weight",
                )
            else:
                dtype = (
                    "float16"
                    if len(shape) == 2 and not name.endswith("conv.weight")
                    else "float32"
                )
                self.buffers[name] = graph.buffer(
                    name, (prod(shape),), dtype, role="weight"
                )

    def __getitem__(self, name):
        return self.buffers[name]

    def matrix(self, name):
        return [(self.buffers[name], 0, self.shapes[name][0])]

    def parameters(self, name):
        return {"weight_dtype": self.buffers[name].dtype}
