"""Qwen checkpoint inventory, raw storage constraints and projection layouts."""

from __future__ import annotations
from functools import cached_property
from akas.checkpoints.upload import upload_shards
from akas.checkpoints.upload import upload_concatenated
from akas.models.qwen35.config import Qwen35Config
from collections import Counter
from pathlib import Path
from akas.checkpoints.safetensors import inspect_tensors
from akas.checkpoints.safetensors import read_tensor
from akas.checkpoints.safetensors import json_document as _json


class Qwen35Checkpoint:
    """Validate the complete text checkpoint before allocating GPU weights."""

    storage = {"F8_E4M3": ("uint8", 1), "BF16": ("uint16", 2), "F32": ("float32", 4)}

    @cached_property
    def tokenizer(self):
        from akas.text.tokenizer import Tokenizer

        return Tokenizer.from_pretrained(self.directory)

    def __init__(self, directory, *, branch="text"):
        self.directory = Path(directory).resolve()
        value = _json((self.directory / "config.json").read_bytes())
        self.config = Qwen35Config.from_json(value)
        if branch not in ("text", "mtp"):
            raise ValueError("checkpoint branch must be text or mtp")
        self.branch = branch
        if branch == "mtp" and (
            value["text_config"].get("mtp_num_hidden_layers") != 1
            or value["text_config"].get("mtp_use_dedicated_embeddings") is not False
        ):
            raise ValueError("MTP requires one decoder layer and shared embeddings")
        index = _json((self.directory / "model.safetensors.index.json").read_bytes())[
            "weight_map"
        ]
        selected = (
            self.config.expected_tensors()
            if branch == "text"
            else self.config.expected_mtp_tensors()
        )
        actual = (
            {
                n
                for n in index
                if n.startswith("model.language_model.") or n == "lm_head.weight"
            }
            if branch == "text"
            else {n for n in index if n.startswith("mtp.")}
        )
        if actual != selected.keys():
            raise ValueError(
                f"Qwen35 {branch} tensor set mismatch: missing={len(selected.keys() - actual)}, unexpected={len(actual - selected.keys())}"
            )
        dtypes = {}
        for name in selected:
            dtype = (
                "F8_E4M3"
                if name.endswith(".weight") and name + "_scale_inv" in selected
                else "BF16"
            )
            if name.endswith((".linear_attn.A_log", ".linear_attn.norm.weight")):
                dtype = "F32"
            dtypes[name] = dtype
        self.tensors = inspect_tensors(self.directory, index, selected, dtypes)

    @property
    def weight_bytes(self):
        return sum((info.nbytes for info in self.tensors.values()))

    def inventory(self):
        counts, sizes = (Counter(), Counter())
        for info in self.tensors.values():
            counts[info.dtype] += 1
            sizes[info.dtype] += info.nbytes
        return dict(
            weight_bytes=self.weight_bytes,
            tensors=dict(counts),
            bytes_by_dtype=dict(sizes),
        )

    def read(self, name):
        """Copy a single raw tensor for producer/reference use, preserving bits."""
        return read_tensor(self.tensors[name])

    def upload(self, device, *, names=None, pack_experts=False):
        """Upload native bytes; close all allocations if any transfer fails.

        FP8 is stored as uint8 with an explicit E4M3FN kernel decoder; BF16
        scales remain BF16. A synchronous default-stream barrier completes each
        transfer before unmapping its host shard or launching on another stream.
        The returned buffers are owned by the caller.
        """
        wanted = list(self.tensors) if names is None else list(names)
        layouts = self.storage_layouts(names=wanted, pack_experts=pack_experts)
        return upload_shards(device, self.tensors, wanted, layouts)

    def storage_layouts(self, *, names=None, pack_experts=False):
        """Use the same grouped storage declaration for graphs and transfers."""
        wanted = list(self.tensors) if names is None else list(names)
        if len(set(wanted)) != len(wanted) or any(
            (n not in self.tensors for n in wanted)
        ):
            raise ValueError("unknown or duplicate requested checkpoint tensor")
        layouts = {}
        for name in wanted:
            info = self.tensors[name]
            if pack_experts and ".mlp.experts." in name:
                prefix, suffix = name.split(".mlp.experts.", 1)
                expert, tail = suffix.split(".", 1)
                group = prefix + ".mlp.experts." + tail
                layouts[name] = (
                    group,
                    int(expert) * info.nbytes,
                    (self.config.experts, *info.shape),
                )
            else:
                layouts[name] = (name, 0, info.shape)
        if pack_experts:
            counts = Counter((v[0] for v in layouts.values()))
            for group, count in counts.items():
                if ".mlp.experts." in group and count != self.config.experts:
                    raise ValueError(
                        "packed loading requires every expert in each selected projection"
                    )
        return layouts

    def storage_specs(self, *, names=None, pack_experts=False):
        from akas.checkpoints.upload import DTYPES

        return {group: (tuple(shape), DTYPES[self.tensors[name].dtype])
                for name, (group, _, shape) in
                self.storage_layouts(names=names, pack_experts=pack_experts).items()}


from akas.models.qwen35.config import DenseConfig


class DenseCheckpoint:
    def program_upload_options(self, weights, *, model_options):
        """Select the checkpoint branch from the program's declared weights."""
        return {"branch": "all" if any(name.startswith("mtp.") for name in weights) else "text"}

    @cached_property
    def tokenizer(self):
        from akas.text.tokenizer import Tokenizer

        return Tokenizer.from_pretrained(self.directory)

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.config = DenseConfig.from_json(
            _json((self.directory / "config.json").read_bytes())
        )
        index = _json((self.directory / "model.safetensors.index.json").read_bytes())[
            "weight_map"
        ]
        self.tensors = {}
        selected = {
            name for name in index if name.startswith(("model.language_model.", "mtp."))
        }
        expected = self.config.expected_tensors()
        if selected != expected.keys():
            raise ValueError("dense checkpoint tensor set mismatch")
        dtypes = {
            name: (
                "F32"
                if name.endswith(("linear_attn.A_log", "linear_attn.norm.weight"))
                else "BF16"
            )
            for name in expected
        }
        self.tensors = inspect_tensors(self.directory, index, expected, dtypes)

    def read(self, name):
        return read_tensor(self.tensors[name])

    def upload(self, device, *, branch="text"):
        if branch not in ("text", "mtp", "all"):
            raise ValueError("invalid checkpoint branch")
        selected = {
            name: info
            for name, info in self.tensors.items()
            if branch == "all" or name.startswith("mtp.") == (branch == "mtp")
        }
        groups = dense_groups(self.config, branch)
        return upload_concatenated(device, selected, self.read, groups)


    def storage_specs(self, *, branch="text"):
        if branch not in ("text", "mtp", "all"):
            raise ValueError("invalid checkpoint branch")
        from akas.checkpoints.upload import DTYPES
        selected = {name:info for name,info in self.tensors.items()
                    if branch == "all" or name.startswith("mtp.") == (branch == "mtp")}
        specs = {name:(tuple(info.shape),DTYPES[info.dtype]) for name,info in selected.items()}
        for destination,names,dtype in dense_groups(self.config,branch):
            shapes = [specs[name][0] for name in names]
            specs[destination] = ((sum(shape[0] for shape in shapes),*shapes[0][1:]),dtype)
            for name in names:
                del specs[name]
        return specs


def dense_groups(config, branch="text"):
    if branch == "all":
        return [*dense_groups(config, "text"), *dense_groups(config, "mtp")]
    groups = []
    layers = config.layers if branch == "text" else ("full_attention",)
    for layer, kind in enumerate(layers):
        root = (
            f"model.language_model.layers.{layer}."
            if branch == "text"
            else f"mtp.layers.{layer}."
        )
        if kind == "linear_attention":
            names = tuple(
                root + "linear_attn.in_proj_" + name + ".weight"
                for name in ("qkv", "z", "a", "b")
            )
            destination = root + "linear_attn.input.weight"
        else:
            names = tuple(
                root + "self_attn." + name + "_proj.weight"
                for name in ("q", "k", "v")
            )
            destination = root + "self_attn.input.weight"
        groups.append((destination, names, "bfloat16"))
        names = tuple(
            root + "mlp." + name + "_proj.weight" for name in ("gate", "up")
        )
        groups.append((root + "mlp.input.weight", names, "bfloat16"))
    return groups

def expected_moe_tensors(config):
    shapes = {
        "model.language_model.embed_tokens.weight": (config.vocab, config.width),
        "model.language_model.norm.weight": (config.width,),
        "lm_head.weight": (config.vocab, config.width),
    }

    def vector(name, shape):
        shapes[name] = shape

    def fp8(name, rows, columns):
        shapes[name + ".weight"] = (rows, columns)
        shapes[name + ".weight_scale_inv"] = (
            (rows + 127) // 128,
            (columns + 127) // 128,
        )

    for layer, kind in enumerate(config.layers):
        root = f"model.language_model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            vector(root + name + ".weight", (config.width,))
        if kind == "linear_attention":
            prefix = root + "linear_attn."
            vector(prefix + "A_log", (config.value_heads,))
            vector(prefix + "dt_bias", (config.value_heads,))
            vector(prefix + "conv1d.weight", (8192, 1, config.conv_width))
            vector(prefix + "norm.weight", (config.value_dim,))
            for name in ("a", "b"):
                vector(
                    prefix + "in_proj_" + name + ".weight",
                    (config.value_heads, config.width),
                )
            for name, rows, columns in (
                ("in_proj_qkv", 8192, config.width),
                ("in_proj_z", 4096, config.width),
                ("out_proj", config.width, 4096),
            ):
                fp8(prefix + name, rows, columns)
        else:
            prefix = root + "self_attn."
            for name, rows, columns in (
                ("q_proj", 8192, config.width),
                ("k_proj", 512, config.width),
                ("v_proj", 512, config.width),
                ("o_proj", config.width, 4096),
            ):
                fp8(prefix + name, rows, columns)
            for name in ("q_norm", "k_norm"):
                vector(prefix + name + ".weight", (config.head_dim,))
        prefix = root + "mlp."
        vector(prefix + "gate.weight", (config.experts, config.width))
        vector(prefix + "shared_expert_gate.weight", (1, config.width))
        for expert in (
            "shared_expert",
            *(f"experts.{i}" for i in range(config.experts)),
        ):
            for name, rows, columns in (
                ("gate_proj", config.expert_width, config.width),
                ("up_proj", config.expert_width, config.width),
                ("down_proj", config.width, config.expert_width),
            ):
                fp8(prefix + expert + "." + name, rows, columns)
    return shapes


def expected_dense_tensors(config):
    shapes = {
        "model.language_model.embed_tokens.weight": (config.vocab, config.width),
        "model.language_model.norm.weight": (config.width,),
    }
    for layer, kind in enumerate(config.layers):
        root = f"model.language_model.layers.{layer}."
        shapes.update(
            {
                root + n + ".weight": (config.width,)
                for n in ("input_layernorm", "post_attention_layernorm")
            }
        )
        for name, shape in (
            ("gate", (config.intermediate, config.width)),
            ("up", (config.intermediate, config.width)),
            ("down", (config.width, config.intermediate)),
        ):
            shapes[root + "mlp." + name + "_proj.weight"] = shape
        if kind == "linear_attention":
            prefix = root + "linear_attn."
            shapes.update(
                {
                    prefix + "A_log": (config.value_heads,),
                    prefix + "dt_bias": (config.value_heads,),
                    prefix + "conv1d.weight": (config.qkv_width, 1, 4),
                    prefix + "norm.weight": (config.value_dim,),
                }
            )
            for name, shape in (
                ("qkv", (config.qkv_width, config.width)),
                ("z", (config.value_width, config.width)),
                ("a", (config.value_heads, config.width)),
                ("b", (config.value_heads, config.width)),
            ):
                shapes[prefix + "in_proj_" + name + ".weight"] = shape
            shapes[prefix + "out_proj.weight"] = (config.width, config.value_width)
        elif kind == "full_attention":
            prefix = root + "self_attn."
            for name, shape in (
                ("q", (2 * config.heads * config.head_dim, config.width)),
                ("k", (config.kv_heads * config.head_dim, config.width)),
                ("v", (config.kv_heads * config.head_dim, config.width)),
                ("o", (config.width, config.heads * config.head_dim)),
            ):
                shapes[prefix + name + "_proj.weight"] = shape
            shapes.update(
                {prefix + n + "_norm.weight": (config.head_dim,) for n in ("q", "k")}
            )
        else:
            raise ValueError("unsupported dense layer type")
    full = next((i for i, kind in enumerate(config.layers) if kind == "full_attention"))
    prefix = f"model.language_model.layers.{full}."
    shapes.update(
        {
            "mtp.layers.0." + name[len(prefix) :]: shape
            for name, shape in tuple(shapes.items())
            if name.startswith(prefix)
        }
    )
    shapes.update(
        {
            "mtp." + n + ".weight": (config.width,)
            for n in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden")
        }
    )
    shapes["mtp.fc.weight"] = (config.width, 2 * config.width)
    return shapes


def expected_mtp_tensors(config):
    """The official one-layer draft branch; embedding/head are shared."""
    prefix = "model.language_model.layers.3."
    shapes = {
        "mtp.layers.0." + name[len(prefix) :]: shape
        for name, shape in expected_tensors(config).items()
        if name.startswith(prefix)
    }
    shapes.update(
        {
            "mtp." + name + ".weight": (config.width,)
            for name in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden")
        }
    )
    shapes["mtp.fc.weight"] = (config.width, 2 * config.width)
    return shapes


def expected_tensors(config):
    return (
        expected_moe_tensors(config)
        if isinstance(config, Qwen35Config)
        else expected_dense_tensors(config)
    )
