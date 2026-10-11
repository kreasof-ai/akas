"""LFM2 shape, precision and provider selection rules."""

from __future__ import annotations
from akas.profiles.independent.parameters import DECODE_GEMV
from akas.profiles.independent.parameters import DECODE_GEMV_COMMON
from akas.profiles.independent.parameters import HALF_PREFILL
from akas.profiles.independent.parameters import PREFILL_OUTER
from akas.profiles.independent.parameters import Q16_PREFILL
from akas.profiles.independent.parameters import QUANT_ATTENTION
from akas.profiles.independent.parameters import QUANT_DECODE
from akas.profiles.independent.parameters import QUANT_PREFILL
from akas.profiles.independent.parameters import WIDE_TILE
from akas.profiles.independent.parameters import WIDE_TILE_COLUMNS


def prefill_tail_rows(cfg, gguf, profile):
    """Bounded suffix liveness: two width-three convs need five input rows.

    Crop queries after the final attention has stored every K/V. Eight rows
    cover its pointwise FFN and both conv histories; their discarded early
    outputs may differ, but the final output and persistent histories agree.
    """
    if (
        profile not in ("prefill_mixed", "cuda_optimized")
        or (cfg.width, cfg.ff) != (2048, 10752)
        or tuple(cfg.layers[-3:]) != ("attention", "conv", "conv")
    ):
        return 0
    for i in range(len(cfg.layers) - 3, len(cfg.layers)):
        names = ["ffn_gate", "ffn_up", "ffn_down"]
        if cfg.layers[i] == "conv":
            names += ["shortconv.in_proj", "shortconv.out_proj"]
        else:
            names += ["attn_q", "attn_output"]
        allowed = (0, 1, 2, 12, 14) if profile == "cuda_optimized" else (2,)
        if any(
            gguf.tensors[f"blk.{i}.{name}.weight"].type not in allowed for name in names
        ):
            return 0
        if (
            gguf.tensors[f"blk.{i}.ffn_gate.weight"].type
            != gguf.tensors[f"blk.{i}.ffn_up.weight"].type
        ):
            return 0
    return 8


def valid_rows(provider, rows, profile):
    if type(rows) is not tuple or not rows or any(type(r) is not int for r in rows):
        return False
    if provider == "cuda":
        return rows == (1, 128) or (
            profile == "optimized"
            and rows[0] == 1
            and len(rows) > 1
            and rows[1:] == tuple(sorted(set(rows[1:])))
            and all(r in (32, 64, 128) for r in rows[1:])
        )
    if provider != "webgpu":
        return False
    return rows == (1, 32) or (
        profile
        in (
            "prefill_unrolled",
            "prefill_outer",
            "prefill_chunked",
            "quant_searched",
            "prefill_q16",
            "prefill_mixed",
        )
        and rows[0] == 1
        and len(rows) > 1
        and rows[1:] == tuple(sorted(set(rows[1:])))
        and all(r in (32, 64, 128) for r in rows[1:])
    )


def paired_ffn(rows, gate, up, profile=None):
    # Quantized decode benefits from shared unpack/input work; the measured
    # F16 decode pair loses occupancy. Prefill benefits from register reuse.
    searched = (
        profile
        in (
            "decode_searched",
            "decode_fused",
            "prefill_unrolled",
            "prefill_outer",
            "prefill_chunked",
            "quant_searched",
            "prefill_q16",
            "prefill_mixed",
        )
        and rows == 1
        and gate.type == 1
        and DECODE_GEMV.get(tuple(reversed(gate.shape)), {}).get("family") == "paired"
    )
    cuda_pair = (
        profile == "cuda_optimized"
        and rows == 1
        and gate.type == 1
        and gate.shape == (10752, 2048)
    )
    return gate.type == up.type and (
        rows > 1 or gate.type not in (0, 1) or searched or cuda_pair
    )


def projection_tile(rows, columns):
    return {"tile": WIDE_TILE} if rows > 1 and columns >= WIDE_TILE_COLUMNS else {}


def webgpu_parameters(kind, p, profile):
    """Apply opt-in schedules measured on RX 6700 XT."""
    if profile == "prefill_mixed":
        selected = HALF_PREFILL.get(
            (kind, p.get("r"), p.get("k"), p.get("o"), p.get("type"))
        )
        if selected is not None:
            return dict(p, sg=True, schedule="outer", outer=dict(selected))
        return webgpu_parameters(kind, p, "prefill_q16")
    if profile == "prefill_q16":
        result = webgpu_parameters(kind, p, "quant_searched")
        selected = Q16_PREFILL.get(
            (kind, p.get("r"), p.get("k"), p.get("o"), p.get("type"))
        )
        if selected is not None:
            return dict(p, q16=True, integer=dict(selected))
        if kind == "rms" and p.get("r", 1) > 1:
            result["parallel_rows"] = True
        return result
    if profile == "quant_searched":
        result = webgpu_parameters(kind, p, "prefill_chunked")
        if p.get("type") in (2, 14) and kind in ("linear", "linear_add", "ffn"):
            selected = (
                QUANT_DECODE.get((p["k"], p["o"], p["type"]))
                if p.get("r") == 1 and kind != "ffn"
                else QUANT_PREFILL.get((kind, p.get("r"), p["k"], p["o"], p["type"]))
            )
            if selected is not None:
                result.update(selected)
                if "outer" in selected:
                    result["outer"] = dict(selected["outer"])
        if (
            kind == "attention"
            and p.get("r") == 1
            and (p["h"], p["kh"], p["d"]) == (32, 8, 64)
        ):
            result.update(QUANT_ATTENTION)
        return result
    if profile in ("prefill_unrolled", "prefill_outer", "prefill_chunked"):
        rows = p.get("r", 1)
        # Carry the current F16 FFN schedules over to larger prefill chunks.
        seed = {**p, "r": 32} if rows > 1 and kind in ("linear", "ffn") else p
        result = webgpu_parameters(kind, seed, "decode_searched")
        if "r" in p:
            result["r"] = rows
        if rows > 1 and kind in ("linear", "ffn") and p.get("type") == 1:
            selected = (
                PREFILL_OUTER.get((kind, rows, p["k"], p["o"]))
                if profile == "prefill_outer"
                else None
            )
            if selected is not None:
                result.update(selected)
                if "outer" in selected:
                    result["outer"] = dict(selected["outer"])
            elif profile == "prefill_unrolled":
                result["explicit_unroll"] = True
                result["unroll"] = (
                    16 if result.get("schedule") == "partitioned" else True
                )
        return result
    if profile == "decode_fused":
        result = webgpu_parameters(kind, p, "decode_searched")
        if (
            kind == "attention"
            and p.get("r") == 1
            and (p["h"], p["kh"], p["d"]) == (16, 8, 64)
        ):
            result.update(fused_scores=True, channels=32, value_parts=16)
        return result
    p = dict(p)
    if profile in ("subgroup", "searched", "decode_searched") and kind in (
        "linear",
        "linear_add",
        "ffn",
        "attention",
        "attention_scores",
        "rms",
        "add_rms",
    ):
        p["sg"] = True
    if (
        profile in ("searched", "decode_searched")
        and kind in ("linear", "ffn")
        and p.get("r") == 32
        and p.get("type") == 1
    ):
        shape = (p["k"], p["o"])
        if shape in ((1024, 2560), (2560, 1024)):
            gate = shape == (1024, 2560)
            p.update(
                schedule="partitioned",
                tile=(8 if kind == "ffn" else 16, 8, 64) if gate else (4, 8, 64),
                threads=128 if gate else 64,
                partitions=16,
                unroll=4,
                dot_width=4,
                owner_axis="row",
                k_layout="blocked" if gate else "striped",
            )
    if profile == "decode_searched" and p.get("r") == 1:
        if kind in ("linear", "linear_add", "ffn") and p.get("type") == 1:
            config = DECODE_GEMV.get((p["k"], p["o"]))
            if config is not None:
                p.update(DECODE_GEMV_COMMON)
                p.update({k: v for k, v in config.items() if k != "family"})
                p["decode_schedule"] = "streamed"
        if kind == "attention" and (p["h"], p["kh"], p["d"]) == (16, 8, 64):
            p.update(
                attention_schedule="partitioned_values", channels=64, value_parts=16
            )
    return p
