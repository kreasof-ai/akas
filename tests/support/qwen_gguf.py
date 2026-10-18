"""Small dense/MoE GGUFs with the published Qwen35 layout and mixed encodings."""

import json
from math import prod
from pathlib import Path
import struct

import numpy as np


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def field(value):
    if isinstance(value, str):
        return 8, string(value)
    if type(value) is bool:
        return 7, struct.pack("<?", value)
    if type(value) is int:
        return 4, struct.pack("<I", value)
    if type(value) is float:
        return 6, struct.pack("<f", value)
    if isinstance(value, (tuple, list)):
        if not value:
            return 9, struct.pack("<IQ", 8, 0)
        rows = [field(item) for item in value]
        assert len({kind for kind, _ in rows}) == 1
        return 9, struct.pack("<IQ", rows[0][0], len(rows)) + b"".join(
            data for _, data in rows
        )
    raise ValueError(value)


def write(path, metadata, tensors):
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        kind, data = field(value)
        header += string(key) + struct.pack("<I", kind) + data
    payload = bytearray()
    for name, (shape, kind, raw) in tensors.items():
        payload.extend(b"\0" * (-len(payload) % 32))
        header += (
            string(name)
            + struct.pack("<I", len(shape))
            + struct.pack("<" + "Q" * len(shape), *shape[::-1])
            + struct.pack("<IQ", kind, len(payload))
        )
        payload.extend(np.ascontiguousarray(raw).view("uint8").tobytes())
    header += b"\0" * (-len(header) % 32)
    Path(path).write_bytes(header + payload)
    return Path(path)


def fixture(path, *, moe=False, output=False, mutate=None):
    """Two decoder layers; GDN has two V heads per K head to test permutations."""
    from akas.quantization.weights.ggml.definition import ENCODINGS

    arch = "qwen35moe" if moe else "qwen35"
    settings = {
        "block_count": 2,
        "context_length": 128,
        "embedding_length": 256,
        "feed_forward_length": 512,
        "attention.head_count": 4,
        "attention.head_count_kv": 1,
        "attention.key_length": 256,
        "attention.value_length": 256,
        "attention.layer_norm_rms_epsilon": 1e-6,
        "rope.freq_base": 10000000.0,
        "rope.dimension_sections": [11, 11, 10, 0],
        "rope.dimension_count": 64,
        "ssm.conv_kernel": 4,
        "ssm.state_size": 128,
        "ssm.group_count": 2,
        "ssm.time_step_rank": 4,
        "ssm.inner_size": 512,
        "full_attention_interval": 2,
    }
    if moe:
        settings.update(
            expert_count=4,
            expert_used_count=2,
            expert_feed_forward_length=256,
            expert_shared_feed_forward_length=256,
        )
    metadata = {
        "general.architecture": arch,
        **{arch + "." + key: value for key, value in settings.items()},
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "gpt2",
        "tokenizer.ggml.tokens": [f"x{i}" for i in range(64)],
    }
    tensors = {}
    rng = np.random.default_rng(709)

    def matrix(name, shape, kind=12):
        _, block, size = ENCODINGS[kind]
        if kind == 0:
            raw = rng.normal(0, 0.02, shape).astype("float32")
        else:
            assert shape[-1] % block == 0
            raw = rng.integers(0, 256, (prod(shape) // block, size), dtype="uint8")
            scale_offset = 208 if kind == 14 else 0
            raw[:, scale_offset : scale_offset + 2] = np.full(
                (len(raw), 1), 0.0001, dtype="float16"
            ).view("uint8")
            if kind in (12, 13):
                raw[:, 2:4] = np.full((len(raw), 1), 0.0001, dtype="float16").view(
                    "uint8"
                )
        tensors[name] = (shape, kind, raw)

    def vector(name, shape, value=1.0):
        tensors[name] = (shape, 0, np.full(shape, value, dtype="float32"))

    matrix("token_embd.weight", (64, 256), 14)
    vector("output_norm.weight", (256,))
    if output:
        matrix("output.weight", (64, 256), 8)
    for layer in (0, 1):
        root = f"blk.{layer}."
        vector(root + "attn_norm.weight", (256,))
        vector(root + "post_attention_norm.weight", (256,))
        if layer == 0:
            for name, shape, kind in (
                ("attn_qkv.weight", (1024, 256), 13),
                ("attn_gate.weight", (512, 256), 12),
                ("ssm_alpha.weight", (4, 256), 8),
                ("ssm_beta.weight", (4, 256), 8),
                ("ssm_out.weight", (256, 512), 13),
            ):
                matrix(root + name, shape, kind)
            vector(root + "ssm_a", (4,), -0.5)
            vector(root + "ssm_dt.bias", (4,), 0)
            vector(root + "ssm_norm.weight", (128,))
            vector(root + "ssm_conv1d.weight", (1024, 4), 0.125)
        else:
            for name, shape, kind in (
                ("attn_q.weight", (2048, 256), 12),
                ("attn_k.weight", (256, 256), 13),
                ("attn_v.weight", (256, 256), 14),
                ("attn_output.weight", (256, 1024), 12),
            ):
                matrix(root + name, shape, kind)
            vector(root + "attn_q_norm.weight", (256,))
            vector(root + "attn_k_norm.weight", (256,))
        if moe:
            matrix(root + "ffn_gate_inp.weight", (4, 256), 0)
            matrix(root + "ffn_gate_inp_shexp.weight", (256,), 0)
            for name, shape, kind in (
                ("ffn_gate_exps.weight", (4, 256, 256), 12),
                ("ffn_up_exps.weight", (4, 256, 256), 13),
                ("ffn_down_exps.weight", (4, 256, 256), 14),
                ("ffn_gate_shexp.weight", (256, 256), 12),
                ("ffn_up_shexp.weight", (256, 256), 13),
                ("ffn_down_shexp.weight", (256, 256), 14),
            ):
                matrix(root + name, shape, kind)
        else:
            for name, shape, kind in (
                ("ffn_gate.weight", (512, 256), 12),
                ("ffn_up.weight", (512, 256), 13),
                ("ffn_down.weight", (256, 512), 14),
            ):
                matrix(root + name, shape, kind)
    if mutate:
        mutate(metadata, tensors)
    return write(path, metadata, tensors)
