"""Small LFM3 checkpoints and independent NumPy full-sequence equations."""

import json
import struct
from pathlib import Path
import numpy as np


def fixture(directory, *, fp8=False):
    from akas.models.lfm3.config import LFM3Config
    from akas.models.lfm3.weights import tensor_shapes, hf_names
    from akas.quantization._shared.e4m3 import encode_storage

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = dict(
        model_type="lfm3",
        hidden_size=128,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_hidden_layers=4,
        layer_types=["conv", "full_attention", "sliding_attention", "conv"],
        vocab_size=64,
        norm_eps=1e-5,
        max_position_embeddings=128,
        sliding_window=16,
        conv_L_cache=7,
        conv_bias=False,
        tie_word_embeddings=True,
        use_sandwich_norm=True,
        rope_parameters={
            "full_attention": dict(partial_rotary_factor=0.5, rope_theta=1000000),
            "sliding_attention": dict(rope_theta=10000),
        },
    )
    if fp8:
        data["quantization_config"] = dict(
            quant_method="compressed-tensors",
            format="float-quantized",
            kv_cache_scheme=None,
            config_groups={
                "FP8_BLOCK": dict(
                    weights=dict(
                        num_bits=8,
                        type="float",
                        strategy="block",
                        block_structure=[128, 128],
                        dynamic=False,
                        symmetric=True,
                    ),
                    input_activations=dict(
                        num_bits=8,
                        type="float",
                        strategy="group",
                        group_size=128,
                        dynamic=True,
                        symmetric=True,
                    ),
                )
            },
        )
    c = LFM3Config.from_huggingface(data)
    rng = np.random.default_rng(73)
    payload, header = bytearray(), {}

    def append(name, x, dtype):
        begin = len(payload)
        if dtype == "BF16":
            bits = x.astype("float32").view("uint32")
            raw = ((bits + 32767 + ((bits >> 16) & 1)) >> 16).astype("uint16").tobytes()
        else:
            raw = x.tobytes()
        payload.extend(raw)
        header[name] = dict(
            dtype=dtype, shape=list(x.shape), data_offsets=[begin, len(payload)]
        )

    for name, shape in tensor_shapes(c).items():
        hf = hf_names(c)[name]
        x = (
            rng.uniform(0.8, 1.2, shape)
            if len(shape) == 1
            else rng.normal(0, 0.035, shape)
        ).astype("float32")
        if name.endswith("conv.weight"):
            x = x.reshape(c.width, 1, c.conv)
        quant = (
            fp8
            and len(shape) == 2
            and name != "token_embd.weight"
            and "shortconv." not in name
            and not name.endswith("attn_gate.weight")
        )
        if quant:
            # Fixed exactly BF16 scale makes the fixture independent of quantizer tuning.
            scale = np.full(
                ((shape[0] + 127) // 128, shape[1] // 128), 1 / 256, dtype="float32"
            )
            append(hf, encode_storage(x * 256), "F8_E4M3")
            append(hf[:-6] + "weight_scale", scale, "BF16")
        else:
            append(hf, x, "BF16")
    (directory / "config.json").write_text(json.dumps(data))
    encoded = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )
    return directory


def oracle(checkpoint, tokens):
    c = checkpoint.config
    weights = {n: checkpoint.read(n).copy() for n in checkpoint.names}

    def linear(x, name):
        return (
            x.astype("float16").astype("float32")
            @ weights[name].astype("float16").astype("float32").T
        )

    def norm(x, name):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + c.epsilon) * weights[name]

    x = weights["token_embd.weight"][tokens].astype("float16").astype("float32")
    seq = len(tokens)
    for i, kind in enumerate(c.layers):
        p = f"blk.{i}."
        n = norm(x, p + "attn_norm.weight")
        if kind == "conv":
            B, C, z = np.split(linear(n, p + "shortconv.in_proj.weight"), 3, axis=-1)
            values = B * z
            mixed = np.zeros_like(values)
            taps = weights[p + "shortconv.conv.weight"]
            for t in range(seq):
                for lag in range(min(t + 1, c.conv)):
                    mixed[t] += values[t - lag] * taps[:, c.conv - 1 - lag]
            mixed = linear(C * mixed, p + "shortconv.out_proj.weight")
        else:
            d, h, kh = c.head_dim, c.heads, c.kv_heads
            q = norm(
                linear(n, p + "attn_q.weight").reshape(seq, h, d),
                p + "attn_q_norm.weight",
            )
            k = norm(
                linear(n, p + "attn_k.weight").reshape(seq, kh, d),
                p + "attn_k_norm.weight",
            )
            v = (
                linear(n, p + "attn_v.weight")
                .reshape(seq, kh, d)
                .astype("float16")
                .astype("float32")
            )
            # LFM3 rotates pairs across full head halves, and uses the full head
            # dimension in the frequency exponent even for partially rotated heads.
            half = c.rotary[i] // 2
            angle = np.arange(seq, dtype="float64")[:, None, None] * c.theta[i] ** (
                -2 * np.arange(half)[None, None, :] / d
            )
            co, si = np.cos(angle), np.sin(angle)
            for a in (q, k):
                l, r = a[..., :half].copy(), a[..., d // 2 : d // 2 + half].copy()
                a[..., :half] = l * co - r * si
                a[..., d // 2 : d // 2 + half] = l * si + r * co
            k = k.astype("float16").astype("float32")
            k, v = [np.repeat(a, h // kh, axis=1) for a in (k, v)]
            scores = np.einsum("thd,shd->hts", q, k) * d**-0.5
            positions = np.arange(seq)
            mask = positions[None, :] <= positions[:, None]
            if kind == "sliding_attention":
                mask &= positions[None, :] > positions[:, None] - c.window
            scores = np.where(mask[None], scores, -np.inf)
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            probs = probs.astype("float16").astype("float32")
            att = np.einsum("hts,shd->thd", probs, v)
            gate = 1 / (1 + np.exp(-linear(n, p + "attn_gate.weight")))
            mixed = linear(
                (att * gate[..., None]).reshape(seq, -1), p + "attn_output.weight"
            )
        if c.sandwich:
            mixed = norm(mixed, p + "post_attention_norm.weight")
        x = x + mixed
        n = norm(x, p + "ffn_norm.weight")
        gate = linear(n, p + "ffn_gate.weight")
        mixed = linear(
            gate / (1 + np.exp(-gate)) * linear(n, p + "ffn_up.weight"),
            p + "ffn_down.weight",
        )
        if c.sandwich:
            mixed = norm(mixed, p + "post_ffw_norm.weight")
        x = x + mixed
    return linear(norm(x[-1:], "output_norm.weight"), "token_embd.weight")[0]


def gguf_fixture(path):
    from support.qwen_gguf import write
    from akas.models.lfm3 import LFM3Checkpoint
    from akas.models.lfm3.weights import tensor_shapes

    source = LFM3Checkpoint(fixture(Path(path).parent / "source"), tokenizer=False)
    c = source.config
    kv = [0 if k == "conv" else c.kv_heads for k in c.layers]
    settings = {
        "block_count": len(c.layers),
        "embedding_length": c.width,
        "feed_forward_length": c.ff,
        "context_length": c.max_context,
        "attention.head_count": c.heads,
        "attention.head_count_kv": kv,
        "attention.key_length": c.head_dim,
        "attention.value_length": c.head_dim,
        "attention.sliding_window_pattern": [
            k == "sliding_attention" for k in c.layers
        ],
        "attention.sliding_window": c.window,
        "attention.layer_norm_rms_epsilon": c.epsilon,
        "rope.freq_base": 1000000.0,
        "rope.freq_base_swa": 10000.0,
        "rope.dimension_count": c.head_dim,
        "shortconv.l_cache": c.conv,
    }
    tensors = {}
    for name, shape in tensor_shapes(c).items():
        x = source.read(name)
        if len(shape) == 1 or name.endswith("conv.weight"):
            tensors[name] = (shape, 0, x)
        else:
            blocks = x.reshape(-1, 32)
            scale = (np.max(np.abs(blocks), axis=1) / 7).astype("float16")
            q = np.clip(np.rint(blocks / scale[:, None]) + 8, 0, 15).astype("uint8")
            raw = np.concatenate(
                (scale[:, None].view("uint8"), q[:, :16] | (q[:, 16:] << 4)), axis=1
            )
            tensors[name] = (shape, 2, raw)
    factors = np.r_[np.ones(c.head_dim // 4), np.full(c.head_dim // 4, 1e30)].astype(
        "float32"
    )
    tensors["rope_freqs.weight"] = ((c.head_dim // 2,), 0, factors)
    write(
        path,
        {
            "general.architecture": "lfm3",
            **{"lfm3." + k: v for k, v in settings.items()},
        },
        tensors,
    )
    return path
