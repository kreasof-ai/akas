"""Validate an LFM2 GGUF checkpoint and upload its packed weights."""

from akas.checkpoints.gguf import GGUF, GGUFError
from akas.checkpoints.upload import upload_packed
from akas.models.lfm2.config import Config, LFM2MoEConfig
from akas.text.tokenizer import Tokenizer


class LFM2Checkpoint:
    config_type = Config

    def __new__(cls, path, *, tokenizer=True):
        from pathlib import Path

        if Path(path).is_dir():
            return ReferenceCheckpoint(path, tokenizer=tokenizer)
        return super().__new__(cls)

    def __init__(self, path, *, tokenizer=True):
        self.gguf = GGUF(path)
        self.config = self.config_type.from_gguf(self.gguf)
        validate_tensors(self.config, self.gguf)
        self.tokenizer = Tokenizer.from_gguf(self.gguf.metadata) if tokenizer else None

    @property
    def definition_options(self):
        return {"output": "output.weight" in self.gguf.tensors}

    def graph_weights(
        self, graph, *, provider, expert_weight_layout="raw", dense_weight_layout="raw"
    ):
        return PackedWeights(
            graph,
            self,
            provider=provider,
            expert_weight_layout=expert_weight_layout,
            dense_weight_layout=dense_weight_layout,
        )

    def inventory(self):
        from collections import Counter

        return dict(
            tensors=len(self.gguf.tensors),
            encodings=dict(Counter(t.type for t in self.gguf.tensors.values())),
        )

    def program_upload_options(self, weights, *, model_options):
        layout = model_options.get("expert_weight_layout", "raw")
        if layout not in ("raw", "planar"):
            raise ValueError("expert weight layout must be raw or planar")
        dense = model_options.get("dense_weight_layout", "raw")
        if dense not in ("raw", "planar"):
            raise ValueError("dense weight layout must be raw or planar")
        return {
            **(dict(expert_weight_layout=layout) if layout != "raw" else {}),
            **(dict(dense_weight_layout=dense) if dense != "raw" else {}),
        }

    def upload(self, device, *, expert_weight_layout="raw", dense_weight_layout="raw"):
        if expert_weight_layout not in ("raw", "planar"):
            raise ValueError("expert weight layout must be raw or planar")
        if dense_weight_layout not in ("raw", "planar"):
            raise ValueError("dense weight layout must be raw or planar")
        if dense_weight_layout != "raw" and device.info["provider"] != "webgpu":
            raise ValueError("dense planar weights currently require WebGPU")
        layouts = {
            name: "planar"
            for name, info in self.gguf.tensors.items()
            if (
                expert_weight_layout == "planar"
                and len(info.shape) == 3
                and info.type == 2
            )
            or dense_planar(name, info, dense_weight_layout)
        }
        return upload_packed(device, self.gguf, layouts=layouts)


class LFM2MoECheckpoint(LFM2Checkpoint):
    """Original packed LFM2 MoE tensors, including sigmoid selection biases."""

    config_type = LFM2MoEConfig

    def __new__(cls, path, *, tokenizer=True):
        from pathlib import Path

        if Path(path).is_dir():
            return NativeMoECheckpoint(path, tokenizer=tokenizer)
        if not Path(path).is_file():
            raise ValueError("LFM2 MoE requires a checkpoint file or directory")
        return object.__new__(cls)


class NativeMoECheckpoint:
    """Map original safetensors into shared encoded and floating operations.

    Expert tensors are grouped using the common shard uploader. FP8 payloads
    and their original BF16 scales stay resident without a decoded weight copy.
    """

    def __init__(self, path, *, tokenizer=True):
        from pathlib import Path
        from akas.checkpoints.safetensors import (
            json_document,
            block_fp8_config,
            tensor_index,
            inspect_tensors,
            read_float_tensor,
        )
        import numpy as np

        self.path, self.gguf = Path(path), None
        data = json_document((self.path / "config.json").read_text())
        self.config = LFM2MoEConfig.from_huggingface(data)
        self.fp8 = block_fp8_config(data)
        index = tensor_index(self.path)
        self.output = "lm_head.weight" in index
        self.shapes = tensor_shapes(self.config, output=self.output)
        mapping = huggingface_names(self.config, output=self.output)
        self.groups = {
            name: (mapping[name],) for name in self.shapes if name in mapping
        }
        for layer in range(self.config.dense_layers, len(self.config.layers)):
            logical, source = f"blk.{layer}.", f"model.layers.{layer}.feed_forward."
            self.groups[logical + "ffn_gate_inp.weight"] = (source + "gate.weight",)
            self.groups[logical + "exp_probs_b.bias"] = (source + "expert_bias",)
            for part, original in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
                self.groups[logical + f"ffn_{part}_exps.weight"] = tuple(
                    source + f"experts.{expert}.{original}.weight"
                    for expert in range(self.config.experts)
                )
        shapes, dtypes, self.scales, self.layouts, self.conversions = {}, {}, {}, {}, {}
        from math import prod

        for name, shape in self.shapes.items():
            sources = self.groups[name]
            quantized = self.fp8 and (
                name.endswith("_exps.weight")
                or (
                    len(shape) == 2
                    and name not in ("token_embd.weight", "output.weight")
                    and "shortconv." not in name
                    and "ffn_gate_inp" not in name
                )
            )
            local = shape[1:] if len(sources) > 1 else shape
            source_shape = (
                (self.config.width, 1, self.config.conv)
                if name.endswith("conv.weight")
                else local
            )
            source_dtype = (
                "F8_E4M3" if quantized else "F32" if name.endswith(".bias") else "BF16"
            )
            for expert, source in enumerate(sources):
                shapes[source], dtypes[source] = source_shape, source_dtype
                width = 1 if quantized else 4 if source_dtype == "F32" else 2
                target_shape = shape if quantized else (prod(shape),)
                self.layouts[source] = (
                    name,
                    expert * prod(local) * width,
                    target_shape,
                )
            if quantized:
                scales = tuple(source[:-6] + "weight_scale" for source in sources)
                scale_shape = ((local[-2] + 127) // 128, local[-1] // 128)
                grouped_scale = (
                    (len(sources), *scale_shape) if len(sources) > 1 else scale_shape
                )
                self.scales[name] = name + ".scale"
                for expert, source in enumerate(scales):
                    shapes[source], dtypes[source] = scale_shape, "BF16"
                    self.layouts[source] = (
                        name + ".scale",
                        expert * prod(scale_shape) * 2,
                        grouped_scale,
                    )
            else:
                self.conversions[name] = (
                    "float16"
                    if len(shape) >= 2
                    and not name.endswith(("conv.weight", "ffn_gate_inp.weight"))
                    else "float32"
                )
        if index.keys() != shapes.keys():
            raise ValueError(
                f"LFM2 MoE tensor inventory differs: {sorted(index.keys() ^ shapes.keys())[:8]}"
            )
        self.tensors = inspect_tensors(self.path, index, shapes, dtypes)
        for name in shapes:
            if name.endswith("weight_scale"):
                values = read_float_tensor(self.tensors[name])
                if not np.isfinite(values).all() or (values <= 0).any():
                    raise ValueError(f"invalid FP8 weight scale: {name}")
        self.tokenizer = Tokenizer.from_pretrained(self.path) if tokenizer else None

    @property
    def definition_options(self):
        return {"output": self.output}

    def program_upload_options(self, weights, *, model_options):
        return {}

    def graph_weights(
        self, graph, *, provider, expert_weight_layout="raw", dense_weight_layout="raw"
    ):
        if (
            provider != "cuda"
            or expert_weight_layout != "raw"
            or dense_weight_layout != "raw"
        ):
            raise ValueError(
                "native LFM2 MoE requires CUDA and original weight layouts"
            )
        return NativeWeights(graph, self)

    def upload(self, device):
        from akas.checkpoints.upload import upload_shards

        if device.info["provider"] != "cuda":
            raise ValueError("native LFM2 MoE requires CUDA")
        return upload_shards(
            device,
            self.tensors,
            tuple(self.layouts),
            self.layouts,
            conversions=self.conversions,
        )

    def inventory(self):
        return {"tensors": len(self.tensors), "format": "fp8" if self.fp8 else "bf16"}


class NativeWeights:
    """The family's logical names bound to declared native physical storage."""

    def __init__(self, graph, checkpoint):
        self.checkpoint, self.buffers, self.scales = checkpoint, {}, {}
        for name, _, shape in checkpoint.layouts.values():
            if name not in self.buffers:
                dtype = "bfloat16" if name.endswith(".scale") else "uint8"
                self.buffers[name] = graph.buffer(
                    name, shape, checkpoint.conversions.get(name, dtype), role="weight"
                )
        self.scales = {
            name: self.buffers[scale] for name, scale in checkpoint.scales.items()
        }

    def __getitem__(self, name):
        return self.buffers[name]

    def matrix(self, name):
        return [(self.buffers[name], 0, self.checkpoint.shapes[name][0])]

    def parameters(self, name):
        return {"weight_dtype": self.buffers[name].dtype}

    def weight(self, name):
        from math import prod
        from akas.quantization.weights import Weight

        shape = self.checkpoint.shapes[name]
        scaled = name in self.scales
        return Weight(
            shape,
            (self.buffers[name], self.scales[name])
            if scaled
            else (self.buffers[name],),
            (
                {
                    "format": "fp8" if scaled else "ggml",
                    "rows": prod(shape[:-1]),
                    "options": {} if scaled else {"encoding": 1},
                },
            ),
        )


def tensor_shapes(config, *, output=False):
    expected = {
        "token_embd.weight": (config.vocab, config.width),
        "token_embd_norm.weight": (config.width,),
    }
    for layer, kind in enumerate(config.layers):
        p = f"blk.{layer}."
        c = config.width
        f = config.ff
        d = config.head_dim
        kh = config.kv_heads
        expected.update(
            {
                p + "attn_norm.weight": (c,),
                p + "ffn_norm.weight": (c,),
            }
        )
        if isinstance(config, LFM2MoEConfig) and layer >= config.dense_layers:
            e, f = config.experts, config.expert_width
            expected.update(
                {
                    p + "ffn_gate_inp.weight": (e, c),
                    p + "exp_probs_b.bias": (e,),
                    p + "ffn_gate_exps.weight": (e, f, c),
                    p + "ffn_up_exps.weight": (e, f, c),
                    p + "ffn_down_exps.weight": (e, c, f),
                }
            )
        else:
            expected.update(
                {
                    p + "ffn_gate.weight": (f, c),
                    p + "ffn_up.weight": (f, c),
                    p + "ffn_down.weight": (c, f),
                }
            )
        if kind == "conv":
            expected.update(
                {
                    p + "shortconv.in_proj.weight": (3 * c, c),
                    p + "shortconv.out_proj.weight": (c, c),
                    p + "shortconv.conv.weight": (c, 3),
                }
            )
        else:
            expected.update(
                {
                    p + "attn_q.weight": (c, c),
                    p + "attn_k.weight": (kh * d, c),
                    p + "attn_v.weight": (kh * d, c),
                    p + "attn_output.weight": (c, c),
                    p + "attn_q_norm.weight": (d,),
                    p + "attn_k_norm.weight": (d,),
                }
            )
    if output:
        expected["output.weight"] = (config.vocab, config.width)
    return expected


def validate_tensors(config, gguf):
    expected = tensor_shapes(config, output="output.weight" in gguf.tensors)
    if set(expected) != set(gguf.tensors):
        raise GGUFError("unsupported/missing LFM2 tensor inventory")
    for name, shape in expected.items():
        if gguf.tensors[name].shape != shape:
            raise GGUFError(f"{name}: incompatible LFM2 tensor shape")
        if len(shape) == 1 or name.endswith(("conv.weight", "ffn_gate_inp.weight")):
            if gguf.tensors[name].type != 0:
                raise GGUFError(f"{name}: requires float32 auxiliary weights")


class ReferenceCheckpoint:
    """Canonical floating weights for the readable batched reference path.

    GGUF quantized matrices are dequantized once at load time. The existing
    packed path continues to use their original compressed storage.
    """

    @property
    def definition_options(self):
        return {"output": "output.weight" in self.names}

    def program_upload_options(self, weights, *, model_options):
        matrices = tensor_shapes(self.config, output="output.weight" in self.names)
        dtypes = {
            spec.dtype
            for name, spec in weights.items()
            if len(matrices[name.split(".part")[0]]) == 2
            and (not name.endswith("conv.weight"))
        }
        if len(dtypes) != 1:
            raise ValueError(
                "floating checkpoint requires one matrix storage precision"
            )
        return dict(
            matrix_dtype=dtypes.pop(),
            max_weight_bytes=model_options.get("max_weight_bytes", 64 * 1024 * 1024),
        )

    def __init__(self, path, *, tokenizer=True):
        import json
        from pathlib import Path

        from akas.checkpoints.safetensors import inspect_float_tensors, tensor_index

        self.path = Path(path)
        self.gguf = None
        if self.path.is_file():
            self.gguf = GGUF(self.path)
            self.config = Config.from_gguf(self.gguf)
            self.names = {name: name for name in self.gguf.tensors}
            self.tokenizer = (
                Tokenizer.from_gguf(self.gguf.metadata) if tokenizer else None
            )
        else:
            self.config = Config.from_huggingface(
                json.loads((self.path / "config.json").read_text())
            )
            index = tensor_index(self.path)
            self.names = huggingface_names(
                self.config, output="lm_head.weight" in index
            )
            shapes = tensor_shapes(self.config, output="output.weight" in self.names)
            hf_shapes = {self.names[name]: shape for name, shape in shapes.items()}
            for name in list(hf_shapes):
                if name.endswith("conv.weight"):
                    hf_shapes[name] = (self.config.width, 1, self.config.conv)
            self.tensors = inspect_float_tensors(self.path, index, hf_shapes)
            self.tokenizer = Tokenizer.from_pretrained(self.path) if tokenizer else None

    def read(self, name):
        from akas.checkpoints.safetensors import read_float_tensor

        if self.gguf is not None:
            return self.gguf.array(name).astype("float32")
        return read_float_tensor(self.tensors[self.names[name]]).reshape(
            tensor_shapes(self.config, output="output.weight" in self.names)[name]
        )

    def upload(
        self, device, *, max_weight_bytes=64 * 1024 * 1024, matrix_dtype="float32"
    ):
        from akas.checkpoints.partition import partitions

        result = {}
        try:
            for name in self.names:
                values = self.read(name)
                if values.ndim == 2 and (not name.endswith("conv.weight")):
                    values = values.astype(matrix_dtype)
                for part in partitions(name, values.shape, max_bytes=max_weight_bytes):
                    selected = (
                        values
                        if part.name == name
                        else values[part.start : part.start + part.rows]
                    )
                    result[part.name] = device.from_numpy(selected.reshape(-1))
            return result
        except BaseException:
            for buffer in result.values():
                buffer.release()
            raise


def huggingface_names(config, *, output=False):
    names = {
        "token_embd.weight": "model.embed_tokens.weight",
        "token_embd_norm.weight": "model.embedding_norm.weight",
    }
    if output:
        names["output.weight"] = "lm_head.weight"
    for i, kind in enumerate(config.layers):
        prefix, hf = (f"blk.{i}.", f"model.layers.{i}.")
        mapped = {
            "attn_norm": "operator_norm",
            "ffn_norm": "ffn_norm",
            "ffn_gate": "feed_forward.w1",
            "ffn_up": "feed_forward.w3",
            "ffn_down": "feed_forward.w2",
        }
        if kind == "conv":
            mapped.update(
                {
                    f"shortconv.{part}": f"conv.{part}"
                    for part in ("in_proj", "out_proj", "conv")
                }
            )
        else:
            mapped.update(
                {
                    "attn_q": "self_attn.q_proj",
                    "attn_k": "self_attn.k_proj",
                    "attn_v": "self_attn.v_proj",
                    "attn_output": "self_attn.out_proj",
                    "attn_q_norm": "self_attn.q_layernorm",
                    "attn_k_norm": "self_attn.k_layernorm",
                }
            )
        names.update(
            {
                prefix + key + ".weight": hf + value + ".weight"
                for key, value in mapped.items()
            }
        )
    return names


def dense_planar(name, info, layout):
    """Embedding gathers keep their existing layout; only dense Q4 dots move."""
    return (
        layout == "planar"
        and len(info.shape) == 2
        and info.type == 2
        and name != "token_embd.weight"
    )


class PackedWeights:
    """Declare original GGML bytes; mathematical shapes stay with the checkpoint."""

    def __init__(
        self,
        graph,
        checkpoint,
        *,
        provider,
        expert_weight_layout="raw",
        dense_weight_layout="raw",
    ):
        from math import prod

        from akas.quantization.weights.ggml.definition import weight_storage

        if expert_weight_layout not in ("raw", "planar"):
            raise ValueError("expert weight layout must be raw or planar")
        if dense_weight_layout not in ("raw", "planar"):
            raise ValueError("dense weight layout must be raw or planar")
        if dense_weight_layout != "raw" and provider != "webgpu":
            raise ValueError("dense planar weights currently require WebGPU")
        self.checkpoint = checkpoint
        self.expert_weight_layout = expert_weight_layout
        self.dense_weight_layout = dense_weight_layout
        self.buffers = {}
        for name, info in checkpoint.gguf.tensors.items():
            if dense_planar(name, info, dense_weight_layout):
                from akas.quantization.weights.ggml.definition import validate_layout

                validate_layout(info.shape, encoding=2, layout="planar")
            count, dtype = weight_storage(
                info.type, prod(info.shape), packed_words=provider == "webgpu"
            )
            self.buffers[name] = graph.buffer(name, (count,), dtype, role="weight")

    def __getitem__(self, name):
        return self.buffers[name]

    def matrix(self, name):
        return [(self.buffers[name], 0, self.checkpoint.gguf.tensors[name].shape[0])]

    def parameters(self, name):
        info = self.checkpoint.gguf.tensors[name]
        return {
            "ggml": True,
            "type": info.type,
            **(
                dict(weight_layout="planar")
                if dense_planar(name, info, self.dense_weight_layout)
                else {}
            ),
        }

    def parts(self, name):
        from math import prod

        info = self.checkpoint.gguf.tensors[name]
        options = {"encoding": info.type}
        if (
            self.expert_weight_layout == "planar"
            and len(info.shape) == 3
            and info.type == 2
        ) or dense_planar(name, info, self.dense_weight_layout):
            from akas.quantization.weights.ggml.definition import validate_layout

            validate_layout(info.shape, encoding=2, layout="planar")
            options["layout"] = "planar"
        return (dict(rows=prod(info.shape[:-1]), format="ggml", options=options),)
