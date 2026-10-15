"""LFM2 schedule fixtures used by compiler regression tests."""




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

"""LFM2 shape, precision and provider selection rules."""


def prefill_tail_rows(cfg, gguf, profile):
    """Bounded suffix liveness: two width-three convs need five input rows.

    Crop queries after the final attention has stored every K/V. Eight rows
    cover its pointwise FFN and both conv histories; their discarded early
    outputs may differ, but the final output and persistent histories agree.
    """
    if profile not in ('prefill_mixed', 'cuda_optimized') or (cfg.width, cfg.ff) != (2048, 10752) or tuple(cfg.layers[-3:]) != ('attention', 'conv', 'conv'):
        return 0
    for i in range(len(cfg.layers) - 3, len(cfg.layers)):
        names = ['ffn_gate', 'ffn_up', 'ffn_down']
        if cfg.layers[i] == 'conv':
            names += ['shortconv.in_proj', 'shortconv.out_proj']
        else:
            names += ['attn_q', 'attn_output']
        allowed = (0, 1, 2, 12, 14) if profile == 'cuda_optimized' else (2,)
        if any((gguf.tensors[f'blk.{i}.{name}.weight'].type not in allowed for name in names)):
            return 0
        if gguf.tensors[f'blk.{i}.ffn_gate.weight'].type != gguf.tensors[f'blk.{i}.ffn_up.weight'].type:
            return 0
    return 8

def valid_rows(provider, rows, profile):
    if type(rows) is not tuple or not rows or any((type(r) is not int for r in rows)):
        return False
    if provider == 'cuda':
        return rows == (1, 128) or (profile == 'optimized' and rows[0] == 1 and (len(rows) > 1) and (rows[1:] == tuple(sorted(set(rows[1:])))) and all((r in (32, 64, 128) for r in rows[1:])))
    if provider != 'webgpu':
        return False
    return rows == (1, 32) or (profile in ('prefill_unrolled', 'prefill_outer', 'prefill_chunked', 'quant_searched', 'prefill_q16', 'prefill_mixed') and rows[0] == 1 and (len(rows) > 1) and (rows[1:] == tuple(sorted(set(rows[1:])))) and all((r in (32, 64, 128) for r in rows[1:])))

def paired_ffn(rows, gate, up, profile=None):
    searched = profile in ('decode_searched', 'decode_fused', 'prefill_unrolled', 'prefill_outer', 'prefill_chunked', 'quant_searched', 'prefill_q16', 'prefill_mixed') and rows == 1 and (gate.type == 1) and (DECODE_GEMV.get(tuple(reversed(gate.shape)), {}).get('family') == 'paired')
    cuda_pair = profile == 'cuda_optimized' and rows == 1 and (gate.type == 1) and (gate.shape == (10752, 2048))
    return gate.type == up.type and (rows > 1 or gate.type not in (0, 1) or searched or cuda_pair)

def projection_tile(rows, columns):
    return {'tile': WIDE_TILE} if rows > 1 and columns >= WIDE_TILE_COLUMNS else {}

def webgpu_parameters(kind, p, profile):
    """Apply opt-in schedules measured on RX 6700 XT."""
    if profile == 'prefill_mixed':
        selected = HALF_PREFILL.get((kind, p.get('r'), p.get('k'), p.get('o'), p.get('type')))
        if selected is not None:
            return dict(p, sg=True, schedule='outer', outer=dict(selected))
        return webgpu_parameters(kind, p, 'prefill_q16')
    if profile == 'prefill_q16':
        result = webgpu_parameters(kind, p, 'quant_searched')
        selected = Q16_PREFILL.get((kind, p.get('r'), p.get('k'), p.get('o'), p.get('type')))
        if selected is not None:
            return dict(p, q16=True, integer=dict(selected))
        if kind == 'rms' and p.get('r', 1) > 1:
            result['parallel_rows'] = True
        return result
    if profile == 'quant_searched':
        result = webgpu_parameters(kind, p, 'prefill_chunked')
        if p.get('type') in (2, 14) and kind in ('linear', 'linear_add', 'ffn'):
            selected = QUANT_DECODE.get((p['k'], p['o'], p['type'])) if p.get('r') == 1 and kind != 'ffn' else QUANT_PREFILL.get((kind, p.get('r'), p['k'], p['o'], p['type']))
            if selected is not None:
                result.update(selected)
                if 'outer' in selected:
                    result['outer'] = dict(selected['outer'])
        if kind == 'attention' and p.get('r') == 1 and ((p['h'], p['kh'], p['d']) == (32, 8, 64)):
            result.update(QUANT_ATTENTION)
        return result
    if profile in ('prefill_unrolled', 'prefill_outer', 'prefill_chunked'):
        rows = p.get('r', 1)
        seed = {**p, 'r': 32} if rows > 1 and kind in ('linear', 'ffn') else p
        result = webgpu_parameters(kind, seed, 'decode_searched')
        if 'r' in p:
            result['r'] = rows
        if rows > 1 and kind in ('linear', 'ffn') and (p.get('type') == 1):
            selected = PREFILL_OUTER.get((kind, rows, p['k'], p['o'])) if profile == 'prefill_outer' else None
            if selected is not None:
                result.update(selected)
                if 'outer' in selected:
                    result['outer'] = dict(selected['outer'])
            elif profile == 'prefill_unrolled':
                result['explicit_unroll'] = True
                result['unroll'] = 16 if result.get('schedule') == 'partitioned' else True
        return result
    if profile == 'decode_fused':
        result = webgpu_parameters(kind, p, 'decode_searched')
        if kind == 'attention' and p.get('r') == 1 and ((p['h'], p['kh'], p['d']) == (16, 8, 64)):
            result.update(fused_scores=True, channels=32, value_parts=16)
        return result
    p = dict(p)
    if profile in ('subgroup', 'searched', 'decode_searched') and kind in ('linear', 'linear_add', 'ffn', 'attention', 'attention_scores', 'rms', 'add_rms'):
        p['sg'] = True
    if profile in ('searched', 'decode_searched') and kind in ('linear', 'ffn') and (p.get('r') == 32) and (p.get('type') == 1):
        shape = (p['k'], p['o'])
        if shape in ((1024, 2560), (2560, 1024)):
            gate = shape == (1024, 2560)
            p.update(schedule='partitioned', tile=(8 if kind == 'ffn' else 16, 8, 64) if gate else (4, 8, 64), threads=128 if gate else 64, partitions=16, unroll=4, dot_width=4, owner_axis='row', k_layout='blocked' if gate else 'striped')
    if profile == 'decode_searched' and p.get('r') == 1:
        if kind in ('linear', 'linear_add', 'ffn') and p.get('type') == 1:
            config = DECODE_GEMV.get((p['k'], p['o']))
            if config is not None:
                p.update(DECODE_GEMV_COMMON)
                p.update({k: v for k, v in config.items() if k != 'family'})
                p['decode_schedule'] = 'streamed'
        if kind == 'attention' and (p['h'], p['kh'], p['d']) == (16, 8, 64):
            p.update(attention_schedule='partitioned_values', channels=64, value_parts=16)
    return p
