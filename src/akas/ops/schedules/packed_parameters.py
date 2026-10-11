"""Measured LFM2 kernel profile parameters."""

from __future__ import annotations


DECODE_GEMV_COMMON = dict(
    lanes=32,
    micro_rows=1,
    dot_width=4,
    unroll=8,
    accumulators=4,
    k_layout="striped",
    shared_input=False,
)


WEBGPU_PROFILES = (
    "portable",
    "subgroup",
    "searched",
    "decode_searched",
    "decode_fused",
    "prefill_unrolled",
    "prefill_outer",
    "prefill_chunked",
    "quant_searched",
    "prefill_q16",
    "prefill_mixed",
)


SUBGROUP_PROFILES = WEBGPU_PROFILES[1:]


QUANT_DECODE = {
    (2048, 128000, 14): dict(
        gemv_q6_dot=True, gemv_threads=256, gemv_unroll=4, gemv_chains=1
    ),
    (2048, 6144, 2): dict(gemv_unroll=8, gemv_chains=8),
    (10752, 2048, 2): dict(
        gemv_lanes=16, gemv_threads=64, gemv_dot=True, gemv_unroll=4
    ),
    (2048, 2048, 2): dict(gemv_unroll=8, gemv_chains=8),
    (2048, 512, 2): dict(gemv_lanes=32, gemv_threads=64, gemv_dot=True, gemv_unroll=4),
}


QUANT_ATTENTION = dict(
    attention_schedule="partitioned_values",
    channels=64,
    value_parts=8,
    fused_scores=True,
)


Q16_SMALL = dict(
    tile_m=16,
    tile_n=32,
    micro_m=2,
    micro_n=2,
    threads=128,
    owner_axis="column",
    group_order="row",
)


Q16_MEDIUM = dict(
    tile_m=32,
    tile_n=64,
    micro_m=2,
    micro_n=4,
    threads=256,
    owner_axis="column",
    group_order="row",
)


Q16_PREFILL = {
    ("ffn", 128, 2048, 10752, 2): Q16_SMALL,
    ("linear", 128, 2048, 6144, 2): Q16_MEDIUM,
    ("linear", 128, 10752, 2048, 2): dict(
        tile_m=64,
        tile_n=128,
        micro_m=4,
        micro_n=8,
        threads=256,
        owner_axis="column",
        group_order="row",
    ),
    ("linear", 128, 2048, 2048, 2): Q16_MEDIUM,
    ("linear", 128, 2048, 512, 2): Q16_SMALL,
}


HALF_PREFILL = {
    ("ffn", 128, 2048, 10752, 2): dict(
        tile_m=64,
        tile_n=64,
        tile_k=16,
        micro_m=4,
        micro_n=4,
        threads=256,
        lhs_layout="km",
        unroll=8,
        explicit_unroll=True,
        dot_width=2,
        packed_pairs=True,
        half_accum=True,
        lhs_pad=1,
        rhs_pad=1,
        group_order="row",
    ),
    ("linear", 128, 2048, 6144, 2): dict(
        tile_m=64,
        tile_n=128,
        tile_k=16,
        micro_m=4,
        micro_n=8,
        threads=256,
        lhs_layout="km",
        unroll=8,
        explicit_unroll=True,
        dot_width=2,
        packed_pairs=True,
        half_accum=True,
    ),
    ("ffn", 8, 2048, 10752, 2): dict(
        tile_m=8,
        tile_n=64,
        tile_k=16,
        micro_m=1,
        micro_n=4,
        threads=128,
        lhs_layout="km",
        unroll=8,
        explicit_unroll=True,
        dot_width=2,
        packed_pairs=True,
        half_accum=True,
    ),
    **{
        ("linear", 8, k, o, 2): dict(
            tile_m=8,
            tile_n=32,
            tile_k=16,
            micro_m=1,
            micro_n=2,
            threads=128,
            lhs_layout="km",
            unroll=8,
            explicit_unroll=True,
            dot_width=2,
            packed_pairs=True,
        )
        for k, o in ((2048, 6144), (2048, 2048), (10752, 2048))
    },
}


PREFILL_OUTER_MK32 = dict(
    tile_m=32,
    tile_n=32,
    tile_k=32,
    micro_m=2,
    micro_n=2,
    threads=256,
    lhs_layout="mk",
    lhs_pad=0,
    rhs_pad=0,
    owner_axis="column",
    unroll=8,
    explicit_unroll=True,
    fma=False,
)


PREFILL_OUTER_KM16 = {
    **PREFILL_OUTER_MK32,
    "tile_n": 64,
    "tile_k": 16,
    "lhs_layout": "km",
    "fma": True,
}


PREFILL_OUTER = {
    ("linear", 32, 1024, 3072): dict(explicit_unroll=True, unroll=True),
    **{
        ("linear", r, 1024, o): dict(schedule="outer", outer=PREFILL_OUTER_MK32)
        for r, o in (
            (32, 1024),
            (32, 512),
            (64, 3072),
            (64, 1024),
            (64, 512),
            (128, 512),
        )
    },
    ("linear", 128, 1024, 3072): dict(
        schedule="outer",
        outer={**PREFILL_OUTER_KM16, "micro_m": 4, "micro_n": 4, "threads": 128},
    ),
    ("linear", 128, 1024, 1024): dict(
        schedule="outer", outer={**PREFILL_OUTER_KM16, "micro_n": 4}
    ),
}


QUANT_KM16 = {**PREFILL_OUTER_MK32, "tile_k": 16, "lhs_layout": "km", "fma": True}


QUANT_LARGE = {**QUANT_KM16, "tile_m": 64, "tile_n": 64, "micro_m": 4, "micro_n": 4}


QUANT_PREFILL = {
    ("ffn", 32, 2048, 10752, 2): dict(
        schedule="outer",
        outer={**QUANT_KM16, "tile_n": 64, "micro_m": 4, "micro_n": 4, "threads": 128},
    ),
    ("linear", 32, 10752, 2048, 2): dict(schedule="outer", outer=QUANT_KM16),
    **{
        ("linear", r, 2048, o, 2): dict(schedule="outer", outer=PREFILL_OUTER_MK32)
        for r, o in ((32, 6144), (32, 2048), (32, 512), (128, 512))
    },
    ("ffn", 128, 2048, 10752, 2): dict(schedule="outer", outer=QUANT_LARGE),
    ("linear", 128, 2048, 6144, 2): dict(
        schedule="outer", outer={**QUANT_LARGE, "tile_n": 128, "micro_n": 8}
    ),
    **{
        ("linear", 128, k, 2048, 2): dict(schedule="outer", outer=QUANT_LARGE)
        for k in (2048, 10752)
    },
}


DECODE_GEMV = {
    (1024, 2560): dict(family="paired", threads=256),
    (1024, 65536): dict(family="separate", threads=256),
    (2560, 1024): dict(family="separate", threads=64, micro_rows=2, unroll=4),
    (1024, 3072): dict(family="separate", threads=256),
    (1024, 1024): dict(family="separate", threads=64),
    (1024, 512): dict(family="separate", threads=64),
}


WIDE_TILE_COLUMNS = 5120


WIDE_TILE = (32, 32, 64)




CUDA_PROFILES = ("default", "optimized")


CUDA_GROUPED_THRESHOLD = 4096
