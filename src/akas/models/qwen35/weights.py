"""Qwen checkpoint inventory, raw storage constraints and projection layouts."""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from pathlib import Path

from akas.checkpoints.safetensors import inspect_tensors, read_tensor
from akas.checkpoints.metadata import json_document as _json
from akas.checkpoints.upload import upload_concatenated, upload_shards
from akas.models.qwen35.config import (
    Qwen35DenseConfig,
    Qwen35MoEConfig,
)


class Qwen35MoECheckpoint:
    """Validate the MoE text checkpoint before allocating GPU weights."""

    storage = {"F8_E4M3": ("uint8", 1), "BF16": ("uint16", 2), "F32": ("float32", 4)}

    def __new__(cls, directory, *, branch="text"):
        if Path(directory).is_file() or Path(directory).suffix.lower() == ".gguf":
            from ._gguf import Checkpoint

            return Checkpoint(directory, moe=True, branch=branch)
        return super().__new__(cls)

    @cached_property
    def tokenizer(self):
        from akas.text.tokenizer import Tokenizer

        return Tokenizer.from_pretrained(self.directory)

    def __init__(self, directory, *, branch="text"):
        self.directory = Path(directory).resolve()
        value = _json((self.directory / "config.json").read_bytes())
        self.config = Qwen35MoEConfig.from_json(value)
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

    @cached_property
    def mtp_checkpoint(self):
        return type(self)(self.directory, branch="mtp")

    def program_upload_options(self, weights, *, model_options):
        branch = "all" if any(name.startswith("mtp.") for name in weights) else "text"
        stored = self.storage_specs(pack_experts=True, branch=branch)
        conversions = {}
        for name, spec in weights.items():
            shape, dtype = stored[name]
            if tuple(shape) != spec.shape:
                raise ValueError("program weight shape differs from checkpoint storage")
            if dtype != spec.dtype:
                if dtype != "uint8" or spec.dtype not in ("float16", "float32"):
                    raise ValueError("unsupported program weight conversion")
                conversions[name] = spec.dtype
        return {
            "pack_experts": True,
            "branch": branch,
            **({"conversions": conversions} if conversions else {}),
        }

    @property
    def weight_bytes(self):
        return sum(info.nbytes for info in self.tensors.values())

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

    def upload(
        self, device, *, names=None, pack_experts=False, branch=None, conversions=None
    ):
        """Upload native bytes; close all allocations if any transfer fails.

        FP8 is stored as uint8 with an explicit E4M3FN kernel decoder; BF16
        scales remain BF16. A synchronous default-stream barrier completes each
        transfer before unmapping its host shard or launching on another stream.
        The returned buffers are owned by the caller.
        """
        if branch not in (None, "text", "mtp", "all"):
            raise ValueError("invalid checkpoint branch")
        if branch == "mtp" and self.branch != "mtp":
            return self.mtp_checkpoint.upload(
                device, names=names, pack_experts=pack_experts, conversions=conversions
            )
        if branch == "all":
            if names is not None or self.branch != "text":
                raise ValueError(
                    "combined upload requires the complete text checkpoint"
                )
            conversions = dict(conversions or {})
            text = self.storage_specs(pack_experts=pack_experts)
            mtp = self.mtp_checkpoint.storage_specs(pack_experts=pack_experts)
            if conversions.keys() - (text.keys() | mtp.keys()):
                raise ValueError("weight conversion names an undeclared storage group")
            result = self.upload(
                device,
                pack_experts=pack_experts,
                conversions={
                    name: dtype for name, dtype in conversions.items() if name in text
                },
            )
            try:
                result.update(
                    self.mtp_checkpoint.upload(
                        device,
                        pack_experts=pack_experts,
                        conversions={
                            name: dtype
                            for name, dtype in conversions.items()
                            if name in mtp
                        },
                    )
                )
                return result
            except BaseException:
                for value in result.values():
                    value.release()
                raise
        wanted = list(self.tensors) if names is None else list(names)
        layouts = self.storage_layouts(names=wanted, pack_experts=pack_experts)
        return upload_shards(
            device, self.tensors, wanted, layouts, conversions=conversions
        )

    def storage_layouts(self, *, names=None, pack_experts=False):
        """Use the same grouped storage declaration for graphs and transfers."""
        wanted = list(self.tensors) if names is None else list(names)
        if len(set(wanted)) != len(wanted) or any(
            n not in self.tensors for n in wanted
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
            counts = Counter(v[0] for v in layouts.values())
            for group, count in counts.items():
                if ".mlp.experts." in group and count != self.config.experts:
                    raise ValueError(
                        "packed loading requires every expert in each selected projection"
                    )
        return layouts

    def storage_specs(self, *, names=None, pack_experts=False, branch=None):
        from akas.checkpoints.upload import DTYPES

        if branch not in (None, "text", "mtp", "all"):
            raise ValueError("invalid checkpoint branch")
        if branch == "mtp" and self.branch != "mtp":
            return self.mtp_checkpoint.storage_specs(
                names=names, pack_experts=pack_experts
            )
        if branch == "all":
            if names is not None or self.branch != "text":
                raise ValueError(
                    "combined declaration requires the complete text checkpoint"
                )
            return {
                **self.storage_specs(pack_experts=pack_experts),
                **self.mtp_checkpoint.storage_specs(pack_experts=pack_experts),
            }
        return {
            group: (tuple(shape), DTYPES[self.tensors[name].dtype])
            for name, (group, _, shape) in self.storage_layouts(
                names=names, pack_experts=pack_experts
            ).items()
        }


class Qwen35DenseCheckpoint:
    """Read the dense text checkpoint and its existing MTP weights."""

    def __new__(cls, directory):
        if Path(directory).is_file() or Path(directory).suffix.lower() == ".gguf":
            from ._gguf import Checkpoint

            return Checkpoint(directory, moe=False)
        return super().__new__(cls)

    def program_upload_options(self, weights, *, model_options):
        """Select the checkpoint branch from the program's declared weights."""
        return {
            "branch": "all"
            if any(name.startswith("mtp.") for name in weights)
            else "text"
        }

    @cached_property
    def tokenizer(self):
        from akas.text.tokenizer import Tokenizer

        return Tokenizer.from_pretrained(self.directory)

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.config = Qwen35DenseConfig.from_json(
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
        if getattr(device, "info", {}).get("provider") == "webgpu":
            import numpy as np
            from math import prod
            from tensor.runtime.dtypes import decode_bfloat16

            if branch == "mtp":
                emb = "model.language_model.embed_tokens.weight"
                selected[emb] = self.tensors[emb]
            result = {}
            try:
                for name, info in selected.items():
                    raw = self.read(name)
                    if info.dtype == "BF16":
                        values = decode_bfloat16(raw).astype("float16")
                    else:
                        values = raw.astype("float32", copy=False)
                    if len(info.shape) == 2:
                        values = values.reshape((prod(info.shape),))
                    result[name] = device.from_numpy(np.ascontiguousarray(values))
                return result
            except BaseException:
                for buffer in result.values():
                    buffer.release()
                raise
        groups = dense_groups(self.config, branch)
        return upload_concatenated(device, selected, self.read, groups)

    def graph_weights(self, graph, *, provider, branch="text"):
        if branch not in ("text", "mtp", "all"):
            raise ValueError("invalid checkpoint branch")
        from math import prod
        from akas.quantization.weights import Weight

        selected = {
            name: info
            for name, info in self.tensors.items()
            if branch == "all" or name.startswith("mtp.") == (branch == "mtp")
        }
        if branch == "mtp":
            emb = "model.language_model.embed_tokens.weight"
            selected[emb] = self.tensors[emb]
        buffers = {}
        result = {}
        for name, info in selected.items():
            shape = tuple(info.shape)
            if len(shape) == 2:
                buf = graph.buffer(name, (prod(shape),), "float16", role="weight")
                buffers[name] = buf
                result[name] = Weight(
                    shape,
                    (buf,),
                    (
                        dict(
                            rows=shape[0],
                            format="ggml",
                            options={"encoding": 1},
                            row_order=None,
                            column_order=None,
                        ),
                    ),
                )
            else:
                dtype = "float32" if info.dtype == "F32" else "float16"
                buf = graph.buffer(name, shape, dtype, role="weight")
                buffers[name] = buf
                result[name] = buf
        for destination, names, _ in dense_groups(self.config, branch):
            shapes = [tuple(selected[name].shape) for name in names]
            result[destination] = Weight(
                (sum(s[0] for s in shapes), *shapes[0][1:]),
                tuple(buffers[name] for name in names),
                tuple(
                    dict(
                        rows=s[0],
                        format="ggml",
                        options={"encoding": 1},
                        row_order=None,
                        column_order=None,
                    )
                    for s in shapes
                ),
            )
        return result

    def storage_specs(self, *, branch="text"):
        if branch not in ("text", "mtp", "all"):
            raise ValueError("invalid checkpoint branch")
        from akas.checkpoints.upload import DTYPES

        selected = {
            name: info
            for name, info in self.tensors.items()
            if branch == "all" or name.startswith("mtp.") == (branch == "mtp")
        }
        specs = {
            name: (tuple(info.shape), DTYPES[info.dtype])
            for name, info in selected.items()
        }
        for destination, names, dtype in dense_groups(self.config, branch):
            shapes = [specs[name][0] for name in names]
            specs[destination] = (
                (sum(shape[0] for shape in shapes), *shapes[0][1:]),
                dtype,
            )
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
                root + "self_attn." + name + "_proj.weight" for name in ("q", "k", "v")
            )
            destination = root + "self_attn.input.weight"
        groups.append((destination, names, "bfloat16"))
        names = tuple(root + "mlp." + name + "_proj.weight" for name in ("gate", "up"))
        groups.append((root + "mlp.input.weight", names, "bfloat16"))
    return groups


# Compatibility for existing imports and stored definition paths.
Qwen35Checkpoint = Qwen35MoECheckpoint
DenseCheckpoint = Qwen35DenseCheckpoint
# These config names were also importable from this module.
Qwen35Config = Qwen35MoEConfig
DenseConfig = Qwen35DenseConfig
