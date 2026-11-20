"""Read, validate and upload the released offline safetensors checkpoint."""

from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from math import prod
import json
from pathlib import Path
import struct

import numpy as np

from akas.checkpoints.safetensors import (
    inspect_float_tensors,
    read_float_tensor,
    read_tensor,
)
from akas.models.voice_isolator.config import VoiceIsolatorConfig


_FP32_PROJECTION_GROUPS = {
    "mamba_coefficients": lambda name: (
        name.startswith("blocks.")
        and name.endswith((".x_proj.weight", ".dt_proj.weight"))
    ),
    "time_qkv": lambda name: (
        name.startswith("blocks.")
        and name.endswith(".time_transformer.attn.qkv.weight")
    ),
    "decoder_output": lambda name: (
        name.startswith("denoise_decoder.heads.") and name.endswith(".3.weight")
    ),
}


@dataclass
class _Store:
    path: Path

    def __post_init__(self):
        with self.path.open("rb") as stream:
            if len(raw := stream.read(8)) != 8:
                raise ValueError("truncated Voice Isolator checkpoint")
            length = struct.unpack("<Q", raw)[0]
            if length > self.path.stat().st_size - 8:
                raise ValueError("truncated Voice Isolator header")
            header = json.loads(stream.read(length))
        if "config" not in header.get("__metadata__", {}):
            raise ValueError("Voice Isolator checkpoint has no embedded architecture")
        self.metadata = header["__metadata__"]


class VoiceIsolatorCheckpoint:
    def __init__(self, path, *, weight_dtype="float32", fp32_projections=()):
        if weight_dtype not in ("float16", "float32"):
            raise ValueError(
                "Voice Isolator weights require float16 or float32 storage"
            )
        if not isinstance(fp32_projections, (list, tuple)) or any(
            not isinstance(group, str) or group not in _FP32_PROJECTION_GROUPS
            for group in fp32_projections
        ):
            raise ValueError("unknown Voice Isolator FP32 projection group")
        if weight_dtype == "float32" and fp32_projections:
            raise ValueError("FP32 projection exceptions require a float16 checkpoint")
        self.fp32_projections = tuple(sorted(set(fp32_projections)))
        self.path = Path(path).resolve()
        if (
            sha256(self.path.read_bytes()).hexdigest()
            != "647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53"
        ):
            raise ValueError(
                "Voice Isolator checkpoint must match the released offline SHA-256"
            )
        self.store = _Store(self.path)
        self.config = VoiceIsolatorConfig.from_metadata(
            json.loads(self.store.metadata["config"])
        )
        self.weight_dtype = weight_dtype
        c = self.config
        shapes = {
            "band_embedding": (len(c.edges), c.width),
            "denoise_decoder.condition.weight": (2 * c.width, 2),
            "denoise_decoder.condition.bias": (2 * c.width,),
        }
        for index, (start, end) in enumerate(c.edges):
            bins = end - start
            for suffix, shape in (
                ("0.weight", (2 * bins,)),
                ("0.bias", (2 * bins,)),
                ("1.weight", (c.width, 2 * bins)),
                ("1.bias", (c.width,)),
            ):
                shapes[f"encoders.{index}.{suffix}"] = shape
            for suffix, shape in (
                ("0.weight", (c.width,)),
                ("0.bias", (c.width,)),
                ("1.weight", (c.width, c.width)),
                ("1.bias", (c.width,)),
                ("3.weight", (4 * bins, c.width)),
                ("3.bias", (4 * bins,)),
            ):
                shapes[f"denoise_decoder.heads.{index}.{suffix}"] = shape
        for index, kind in enumerate(c.block_types):
            prefix = f"blocks.{index}"
            if kind == "transformer":
                for axis, attention in (("time", "attn"), ("band", "self_attn")):
                    for suffix, shape in (
                        ("norm1.weight", (c.width,)),
                        ("norm1.bias", (c.width,)),
                        ("norm2.weight", (c.width,)),
                        ("norm2.bias", (c.width,)),
                    ):
                        shapes[f"{prefix}.{axis}_transformer.{suffix}"] = shape
                    if axis == "time":
                        for suffix in ("qkv", "proj"):
                            shapes[
                                f"{prefix}.time_transformer.attn.{suffix}.weight"
                            ] = (
                                (3 * c.width, c.width)
                                if suffix == "qkv"
                                else (c.width, c.width)
                            )
                            shapes[f"{prefix}.time_transformer.attn.{suffix}.bias"] = (
                                (3 * c.width,) if suffix == "qkv" else (c.width,)
                            )
                        shapes[f"{prefix}.time_transformer.ffn.0.weight"] = (
                            4 * c.width,
                            c.width,
                        )
                        shapes[f"{prefix}.time_transformer.ffn.0.bias"] = (4 * c.width,)
                        shapes[f"{prefix}.time_transformer.ffn.3.weight"] = (
                            c.width,
                            4 * c.width,
                        )
                        shapes[f"{prefix}.time_transformer.ffn.3.bias"] = (c.width,)
                    else:
                        for suffix, shape in (
                            ("in_proj_weight", (3 * c.width, c.width)),
                            ("in_proj_bias", (3 * c.width,)),
                            ("out_proj.weight", (c.width, c.width)),
                            ("out_proj.bias", (c.width,)),
                            ("linear1.weight", (4 * c.width, c.width)),
                            ("linear1.bias", (4 * c.width,)),
                            ("linear2.weight", (c.width, 4 * c.width)),
                            ("linear2.bias", (c.width,)),
                        ):
                            shapes[
                                f"{prefix}.band_transformer.self_attn.{suffix}"
                                if suffix.startswith(("in_", "out_"))
                                else f"{prefix}.band_transformer.{suffix}"
                            ] = shape
            else:
                for axis in ("time_mamba", "freq_mamba"):
                    shapes[f"{prefix}.{axis}.output_proj.weight"] = (
                        c.width,
                        2 * c.width,
                    )
                    shapes[f"{prefix}.{axis}.output_proj.bias"] = (c.width,)
                    shapes[f"{prefix}.{axis}.norm.weight"] = (c.width,)
                    shapes[f"{prefix}.{axis}.norm.bias"] = (c.width,)
                    for direction in ("forward_blocks", "backward_blocks"):
                        branch = f"{prefix}.{axis}.{direction}"
                        for suffix, shape in (
                            ("in_proj.weight", (2 * c.inner, c.width)),
                            ("conv1d.weight", (c.inner, 1, c.mamba_conv)),
                            ("conv1d.bias", (c.inner,)),
                            ("x_proj.weight", (c.dt_rank + 2 * c.mamba_state, c.inner)),
                            ("dt_proj.weight", (c.inner, c.dt_rank)),
                            ("dt_proj.bias", (c.inner,)),
                            ("A_log", (c.inner, c.mamba_state)),
                            ("D", (c.inner,)),
                            ("out_proj.weight", (c.width, c.inner)),
                        ):
                            shapes[f"{branch}.{suffix}"] = shape
        self.shapes = shapes
        index = {name: self.path.name for name in shapes}
        self.tensors = inspect_float_tensors(self.path.parent, index, shapes)
        if set(self.store.metadata) - {
            "config",
            "format",
            "num_params",
            "voice_policy",
        }:
            raise ValueError("unsupported Voice Isolator checkpoint metadata")
        if self.store.metadata.get("format") != "inworld-voice-isolation":
            raise ValueError(
                "Voice Isolator checkpoint format is not inworld-voice-isolation"
            )
        self._cache = {}

    def array(self, name):
        if name not in self._cache:
            self._cache[name] = read_float_tensor(self.tensors[name])
        return self._cache[name]

    def storage_dtype(self, name):
        info = self.tensors[name]
        if len(info.shape) == 2 and name.endswith((".weight", ".in_proj_weight")):
            if any(
                _FP32_PROJECTION_GROUPS[group](name) for group in self.fp32_projections
            ):
                return "float32"
            return self.weight_dtype
        return "float32"

    def packed(self, name):
        dtype = self.storage_dtype(name)
        if dtype == "float32":
            return read_tensor(self.tensors[name]).view("uint8")
        return np.ascontiguousarray(self.array(name).astype(dtype)).view("uint8")

    def discard(self, name):
        self._cache.pop(name, None)

    def buffer(self, graph, name, *, provider):
        info = self.tensors[name]
        dtype = self.storage_dtype(name)
        if name not in graph.buffers:
            graph.buffer(name, (prod(info.shape),), dtype, role="weight")
        return graph.buffers[name]

    def derived(self, name):
        c = self.config
        if name.startswith("inverse_basis@"):
            frequency = np.arange(c.bins, dtype="float64")
            tap = np.arange(c.n_fft, dtype="float64")[:, None]
            angle = 2 * np.pi * frequency[None, :] * tap / c.n_fft
            weight = np.full(c.bins, 2.0)
            weight[[0, -1]] = 1.0
            return (
                np.stack((np.cos(angle), -np.sin(angle)), axis=-1).reshape(
                    c.n_fft, 2 * c.bins
                )
                * weight.repeat(2)[None, :]
                / c.n_fft
            ).astype("float32")
        if name == "window":
            return (
                0.5 - 0.5 * np.cos(2 * np.pi * np.arange(c.n_fft) / c.n_fft)
            ).astype("float32")
        if name in ("rope_fine", "rope_coarse"):
            stride = 512
            positions = np.arange(stride if name == "rope_fine" else 2) * (
                1 if name == "rope_fine" else stride
            )
            angles = positions[:, None] * 10000.0 ** (
                -np.arange(c.head_dim // 2) / (c.head_dim // 2)
            )
            return np.concatenate((np.cos(angles), np.sin(angles)), axis=1).astype(
                "float32"
            )
        if name.startswith("decay."):
            source = name.removeprefix("decay.")
            return -np.exp(self.array(source))
        if name.startswith("position."):
            raw = name.removeprefix("position.").split("@", 1)
            axis, samples = raw[0], int(raw[1])
            if ".time_transformer." in axis or axis.endswith("time_transformer"):
                return np.tile(
                    np.arange(self.config.frames(samples), dtype="int32"),
                    len(self.config.edges),
                )
            return np.tile(
                np.arange(len(self.config.edges), dtype="int32"),
                self.config.frames(samples),
            )
        if name.startswith("unit."):
            return np.ones(int(name.removeprefix("unit.")), dtype="float32")
        if name.startswith("decoder_condition."):
            part = name.rsplit(".", 1)[-1]
            values = self.array("denoise_decoder.condition.weight") @ np.array(
                [1.0, 0.0], dtype="float32"
            )
            values += self.array("denoise_decoder.condition.bias")
            return values[: c.width] if part == "scale" else values[c.width :]
        if name.startswith("band_embedding."):
            return self.array("band_embedding")[int(name.rsplit(".", 1)[-1])]
        raise KeyError(name)

    def upload(self, device, names):
        from akas.checkpoints.upload import UploadBatch

        weights = {}
        with ExitStack() as cleanup, UploadBatch(device) as transfers:
            for name in sorted(names):
                if name in self.tensors:
                    dtype = self.storage_dtype(name)
                    host = self.packed(name).view(dtype).reshape(-1)
                    self.discard(name)
                else:
                    host = self.derived(name)
                buffer = device.from_numpy(np.ascontiguousarray(host))
                weights[name] = buffer
                cleanup.callback(buffer.release)
                transfers.add(host.nbytes)
            cleanup.pop_all()
        return weights

    def inventory(self):
        return {
            "format": "safetensors",
            "source_dtype": "float32",
            "weight_dtype": self.weight_dtype,
            "tensors": len(self.tensors),
        }
