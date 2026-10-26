"""Small deterministic Spark checkpoints and an independent Torch equation oracle."""

import json
import struct
from pathlib import Path
import numpy as np


def fixture(directory):
    from akas.models.spark.config import SparkConfig
    from akas.models.spark.weights import tensor_shapes, hf_names

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = dict(
        model_type="spark2_5",
        architectures=["Spark2_5ForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64,
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        vocab_size=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        sliding_window=16,
        hidden_act="gelu",
        headwise_attn_output_gate=True,
        gate_attn_act_mode="sigmoid",
        tie_word_embeddings=True,
        rope_parameters={
            "sliding_attention": dict(partial_rotary_factor=1.0, rope_theta=10000),
            "full_attention": dict(partial_rotary_factor=0.25, rope_theta=5000000),
        },
    )
    (directory / "config.json").write_text(json.dumps(data))
    c = SparkConfig.from_huggingface(data)
    rng = np.random.default_rng(73)
    payload, header = bytearray(), {}
    for name, shape in tensor_shapes(c).items():
        x = (np.ones(shape) if len(shape) == 1 else rng.normal(0, 0.035, shape)).astype(
            "float32"
        )
        bits = x.view("uint32")
        raw = ((bits + 32767 + ((bits >> 16) & 1)) >> 16).astype("uint16").tobytes()
        begin = len(payload)
        payload.extend(raw)
        header[hf_names(c)[name]] = dict(
            dtype="BF16", shape=list(shape), data_offsets=[begin, len(payload)]
        )
    encoded = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )
    return directory


def oracle(checkpoint, tokens):
    import torch
    import torch.nn.functional as F

    c = checkpoint.config
    weights = {n: torch.from_numpy(checkpoint.read(n).copy()) for n in checkpoint.names}

    def linear(x, name):
        # CUDA projection operands are F16, accumulated into F32.
        return x.half().float() @ weights[name].half().float().T

    def norm(x, name):
        return (
            x * torch.rsqrt((x * x).mean(-1, keepdim=True) + c.epsilon) * weights[name]
        )

    hidden = weights["token_embd.weight"][tokens].half().float()
    seq = len(tokens)

    def rotary(x, i):
        rot = c.rotary[i]
        x = x.clone()
        angle = torch.arange(seq, dtype=torch.float64)[:, None, None] * c.theta[i] ** (
            -2 * torch.arange(rot // 2, dtype=torch.float64)[None, None, :] / rot
        )
        co, si = angle.cos().float(), angle.sin().float()
        left, right = x[..., : rot // 2].clone(), x[..., rot // 2 : rot].clone()
        x[..., : rot // 2] = left * co - right * si
        x[..., rot // 2 : rot] = left * si + right * co
        return x

    for i, kind in enumerate(c.layers):
        p = f"blk.{i}."
        x = norm(hidden, p + "attn_norm.weight")
        qkv = linear(x, p + "attn_qkv.weight")
        q, k, v = torch.split(
            qkv,
            [c.heads * c.head_dim, c.kv_heads * c.head_dim, c.kv_heads * c.head_dim],
            dim=-1,
        )
        q = rotary(q.reshape(seq, c.heads, c.head_dim), i).half().float()
        k = rotary(k.reshape(seq, c.kv_heads, c.head_dim), i).half().float()
        v = v.reshape(seq, c.kv_heads, c.head_dim).half().float()
        k, v = [a.repeat_interleave(c.heads // c.kv_heads, dim=1) for a in (k, v)]
        scores = torch.einsum("thd,shd->hts", q, k) * c.head_dim**-0.5
        positions = torch.arange(seq)
        mask = positions[None, :] <= positions[:, None]
        if kind == "sliding_attention":
            mask &= positions[None, :] > positions[:, None] - c.window
        scores.masked_fill_(~mask[None], float("-inf"))
        probs = scores.softmax(-1).half().float()
        attended = torch.einsum("hts,shd->thd", probs, v)
        attended *= linear(x, p + "attn_gate.weight").sigmoid()[:, :, None]
        hidden = hidden + linear(attended.reshape(seq, -1), p + "attn_output.weight")
        x = norm(hidden, p + "ffn_norm.weight")
        hidden = hidden + linear(
            F.gelu(linear(x, p + "ffn_gate.weight")) * linear(x, p + "ffn_up.weight"),
            p + "ffn_down.weight",
        )
    return linear(norm(hidden[-1:], "output_norm.weight"), "token_embd.weight")[
        0
    ].numpy()


def fp8_fixture(directory):
    from akas.quantization._shared.e4m3 import encode_storage
    from akas.models.spark import SparkCheckpoint

    path = fixture(directory)
    source = SparkCheckpoint(path, tokenizer=False)
    data = json.loads((path / "config.json").read_text())
    data["quantization_config"] = dict(
        quant_method="compressed-tensors",
        format="float-quantized",
        kv_cache_scheme=None,
        config_groups={
            "group_0": dict(
                weights=dict(
                    num_bits=8,
                    type="float",
                    strategy="channel",
                    dynamic=False,
                    symmetric=True,
                    group_size=None,
                ),
                input_activations=dict(
                    num_bits=8,
                    type="float",
                    strategy="token",
                    dynamic=True,
                    symmetric=True,
                ),
            )
        },
    )
    header, payload = {}, bytearray()
    for name, hf in source.names.items():
        x = source.read(name)
        arrays = []
        if (
            x.ndim == 2
            and name not in ("token_embd.weight",)
            and not name.endswith("attn_gate.weight")
        ):
            scale = np.full((x.shape[0], 1), 0.001, dtype="float32")
            bits = (scale.view("uint32") >> 16).astype("uint16")
            scale = (bits.astype("uint32") << 16).view("float32")
            arrays = [
                (hf, encode_storage(x / scale), "F8_E4M3"),
                (hf[:-6] + "weight_scale", bits, "BF16"),
            ]
        else:
            arrays = [(hf, (x.view("uint32") >> 16).astype("uint16"), "BF16")]
        for key, raw, dtype in arrays:
            begin = len(payload)
            payload.extend(raw.tobytes())
            header[key] = dict(
                dtype=dtype, shape=list(raw.shape), data_offsets=[begin, len(payload)]
            )
    encoded = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )
    (path / "config.json").write_text(json.dumps(data))
    return path
