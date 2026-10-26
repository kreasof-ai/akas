"""CUDA specializations and shared bindings for attention."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def mma_partial(spec: KernelSpec, schedule: Mapping[str, object]):
    """Stage native or encoded KV once per GQA tile for tensor-core QK/PV."""
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, h, kh, cap = (p[n] for n in ("r", "h", "kh", "capacity"))
    splits = p.get("splits", 16)
    bk = p.get("key_rows", 64)
    stages = p.get("stages", 2)
    group = h // kh
    qm = 16
    d = 64
    from akas.ops.attention.kernel import cache_reader
    from akas.quantization.kv.codec import get
    from akas.quantization._shared.access import packed_arguments

    native_half = p.get("kv_dtype") == "float16" and p.get("unit_scales", False)
    storage, read = cache_reader({**p, "provider": "cuda"})
    codec = (
        None
        if native_half
        else get(p["kv_dtype"]).bind((r, kh, cap, d), provider="cuda")
    )
    decode = (
        packed_arguments(codec.require("read_packed")) if codec is not None else None
    )
    if (
        h % kh
        or group not in (2, 4, 8, 16)
        or p.get("d") != 64
        or bk not in (64, 128)
        or stages not in (0, 1, 2, 3)
    ):
        raise ValueError("unsupported staged-half GQA decode schedule")

    @T.macro
    def consume(
        start,
        length,
        query,
        key,
        value,
        prob,
        scores,
        result,
        maximum,
        previous,
        factor,
        normalizer,
        total,
    ):
        T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
        T.copy(maximum, previous)
        for i, j in T.Parallel(qm, bk):
            scores[i, j] = T.if_then_else(
                start + j < length,
                scores[i, j] * 0.125,
                -T.infinity("float32"),
            )
        T.reduce_max(scores, maximum, dim=1, clear=False)
        for i in T.Parallel(qm):
            factor[i] = T.if_then_else(
                T.isfinite(previous[i]), T.exp(previous[i] - maximum[i]), 0
            )
        for i, j in T.Parallel(qm, bk):
            scores[i, j] = T.if_then_else(
                start + j < length,
                T.exp(scores[i, j] - maximum[i]),
                0,
            )
        T.reduce_sum(scores, total, dim=1)
        for i in T.Parallel(qm):
            normalizer[i] = normalizer[i] * factor[i] + total[i]
        for i, j in T.Parallel(qm, d):
            result[i, j] *= factor[i]
        T.copy(scores, prob)
        T.gemm(prob, value, result)

    @kernel
    def compute(
        q: Tensor((r, h * d), "float32"),
        kc: Tensor(storage.values.shape, storage.values.dtype),
        vc: Tensor(storage.values.shape, storage.values.dtype),
        ks: Tensor((r, kh, cap, 1), "float32"),
        vs: Tensor((r, kh, cap, 1), "float32"),
        positions: Tensor(r, "int32"),
        enabled: Tensor(r, "int32"),
    ) -> Tensor(r * h * splits * 66, "float32"):
        out = output(r * h * splits * 66, "float32", position=5)
        with T.Kernel(r * kh, splits, threads=128) as (index, split):
            slot = index // kh
            head = index % kh
            length = T.if_then_else(
                enabled[slot] != 0, T.min(positions[slot] + 1, cap), 0
            )
            span = T.ceildiv(T.ceildiv(length, bk), splits)
            query = T.alloc_shared((qm, d), "float16")
            key = T.alloc_shared((bk, d), "float16")
            value = T.alloc_shared((bk, d), "float16")
            if not native_half:
                encoded_k = T.alloc_shared((bk, d), codec.packed_dtype)
                encoded_v = T.alloc_shared((bk, d), codec.packed_dtype)
                scale_k = T.alloc_shared((bk, 1), "float32")
                scale_v = T.alloc_shared((bk, 1), "float32")
                keys = T.view(kc, dtype=codec.packed_dtype)
                values = T.view(vc, dtype=codec.packed_dtype)
            prob = T.alloc_shared((qm, bk), "float16")
            scores = T.alloc_fragment((qm, bk), "float32")
            result = T.alloc_fragment((qm, d), "float32")
            maximum = T.alloc_fragment((qm,), "float32")
            previous = T.alloc_fragment((qm,), "float32")
            factor = T.alloc_fragment((qm,), "float32")
            normalizer = T.alloc_fragment((qm,), "float32")
            total = T.alloc_fragment((qm,), "float32")
            for i, j in T.Parallel(qm, d):
                query[i, j] = T.if_then_else(
                    i < group,
                    q[slot, (head * group + T.min(i, group - 1)) * d + j]
                    * (ks[slot, head, 0, 0] if native_half else 1),
                    0,
                )
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            tiles = T.min(span, T.max(T.ceildiv(length, bk) - split * span, 0))
            # Pipeline complete groups of four. A final group has at most four
            # tiles and uses guarded loads, so stale/NaN padding never reaches PV.
            full_groups = T.min(span, T.max(length // bk - split * span, 0)) // 4
            for block in T.serial(full_groups):
                for local in T.Pipelined(4, num_stages=stages):
                    start = (split * span + block * 4 + local) * bk
                    if native_half:
                        T.copy(kc[slot, head, start, 0], key)
                        T.copy(vc[slot, head, start, 0], value)
                    else:
                        T.copy(keys[slot, head, start, 0], encoded_k)
                        T.copy(values[slot, head, start, 0], encoded_v)
                        T.copy(ks[slot, head, start, 0], scale_k)
                        T.copy(vs[slot, head, start, 0], scale_v)
                        for i, j in T.Parallel(bk, d):
                            key[i, j] = decode(encoded_k, scale_k, i, j)
                            value[i, j] = decode(encoded_v, scale_v, i, j)
                    consume(
                        start,
                        length,
                        query,
                        key,
                        value,
                        prob,
                        scores,
                        result,
                        maximum,
                        previous,
                        factor,
                        normalizer,
                        total,
                    )
            key_tail = T.alloc_shared((bk, d), "float16")
            value_tail = T.alloc_shared((bk, d), "float16")
            prob_tail = T.alloc_shared((qm, bk), "float16")
            scores_tail = T.alloc_fragment((qm, bk), "float32")
            previous_tail = T.alloc_fragment((qm,), "float32")
            factor_tail = T.alloc_fragment((qm,), "float32")
            total_tail = T.alloc_fragment((qm,), "float32")
            for tile in T.serial(full_groups * 4, tiles):
                start = (split * span + tile) * bk
                for i, j in T.Parallel(bk, d):
                    key_tail[i, j] = T.if_then_else(
                        start + i < length,
                        read(kc, ks, slot, head, T.min(start + i, cap - 1), j),
                        0,
                    )
                    value_tail[i, j] = T.if_then_else(
                        start + i < length,
                        read(vc, vs, slot, head, T.min(start + i, cap - 1), j),
                        0,
                    )
                consume(
                    start,
                    length,
                    query,
                    key_tail,
                    value_tail,
                    prob_tail,
                    scores_tail,
                    result,
                    maximum,
                    previous_tail,
                    factor_tail,
                    normalizer,
                    total_tail,
                )
            for i, j in T.Parallel(qm, d):
                if i < group:
                    out[((slot * h + head * group + i) * splits + split) * 66 + j] = (
                        result[i, j] * (vs[slot, head, 0, 0] if native_half else 1)
                    )
            for i in T.Parallel(qm):
                if i < group:
                    out[((slot * h + head * group + i) * splits + split) * 66 + 64] = (
                        maximum[i]
                    )
                    out[((slot * h + head * group + i) * splits + split) * 66 + 65] = (
                        normalizer[i]
                    )
        return out

    return compute


def packed_prefill(spec: KernelSpec, schedule: Mapping[str, object]):
    """Read each encoded KV tile once into half-precision GQA staging."""
    p = {**spec.parameters, **schedule}
    import tilelang.language as T
    from akas.quantization.kv.codec import get
    from akas.quantization._shared.access import cache_arguments, packed_arguments

    r, h, kh, d, cap, chunk = (p[n] for n in ("r", "h", "kh", "d", "capacity", "chunk"))
    slots = r // chunk
    actual_group = h // kh
    group_queries = p.get("group_queries", True)
    group = actual_group if group_queries else 1
    work_heads = kh if group_queries else h
    threads = p.get("threads", 128)
    qm = p.get("query_rows", 64)
    qt = qm // group
    bk = p.get("key_rows", 64)
    stages = p.get("stages", 1)
    pg = p.get("pipeline_group", 8)
    from akas.ops.attention.kernel import cache_reader

    storage, read = cache_reader({**p, "provider": "cuda"})
    native = p["kv_dtype"] == "float16" and p.get("unit_scales")
    if (
        r % chunk
        or chunk < 2
        or d != 64
        or h % kh
        or group not in (1, 2, 4, 8)
        or qm not in (64, 128)
        or bk not in (64, 128)
        or stages not in (1, 2)
        or threads not in (128, 256)
        or pg != 8
    ):
        raise ValueError("unsupported packed GQA prefill schedule")
    codec = (
        get(p["kv_dtype"]).bind((slots, kh, cap, d), provider="cuda")
        if not native
        else None
    )
    decode = packed_arguments(codec.require("read_packed")) if not native else None

    @T.macro
    def consume(
        start,
        positions,
        slot,
        qblock,
        query,
        key,
        value,
        prob,
        scores,
        result,
        maximum,
        previous,
        factor,
        normalizer,
        total,
    ):
        T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
        T.copy(maximum, previous)
        for i, j in T.Parallel(qm, bk):
            scores[i, j] = T.if_then_else(
                qblock * qt + i // group < chunk
                and start + j
                <= positions[slot * chunk + T.min(qblock * qt + i // group, chunk - 1)],
                scores[i, j] * 0.125,
                -T.infinity("float32"),
            )
        T.reduce_max(scores, maximum, dim=1, clear=False)
        for i in T.Parallel(qm):
            factor[i] = T.if_then_else(
                T.isfinite(previous[i]), T.exp(previous[i] - maximum[i]), 0
            )
        for i, j in T.Parallel(qm, bk):
            scores[i, j] = T.if_then_else(
                qblock * qt + i // group < chunk
                and start + j
                <= positions[slot * chunk + T.min(qblock * qt + i // group, chunk - 1)],
                T.exp(scores[i, j] - maximum[i]),
                0,
            )
        T.reduce_sum(scores, total, dim=1)
        for i in T.Parallel(qm):
            normalizer[i] = normalizer[i] * factor[i] + total[i]
        for i, j in T.Parallel(qm, d):
            result[i, j] *= factor[i]
        T.copy(scores, prob)
        T.gemm(prob, value, result)

    @kernel
    def compute(
        q: Tensor((r, h * d), "float32"),
        kc: Tensor(storage.values.shape, storage.values.dtype),
        vc: Tensor(storage.values.shape, storage.values.dtype),
        ks: Tensor(storage.metadata[0].shape, "float32"),
        vs: Tensor(storage.metadata[0].shape, "float32"),
        positions: Tensor(r, "int32"),
        enabled: Tensor(r, "int32"),
    ) -> Tensor(r * h * d, "float32"):
        out = output(r * h * d, "float32", position=5)
        with T.Kernel(T.ceildiv(chunk, qt), work_heads, slots, threads=threads) as (
            qblock,
            head,
            slot,
        ):
            cache_head = head if group_queries else head // actual_group
            limits = T.alloc_fragment((qm,), "int32")
            last = T.alloc_fragment((1,), "int32")
            for i in T.Parallel(qm):
                index = T.min(qblock * qt + i // group, chunk - 1)
                limits[i] = T.if_then_else(
                    qblock * qt + i // group < chunk
                    and enabled[slot * chunk + index] != 0,
                    positions[slot * chunk + index],
                    -1,
                )
            T.reduce_max(limits, last, dim=0)
            length = T.min(last[0] + 1, cap)
            query = T.alloc_shared((qm, d), "float16")
            key = T.alloc_shared((bk, d), "float16")
            value = T.alloc_shared((bk, d), "float16")
            prob = T.alloc_shared((qm, bk), "float16")
            if not native:
                encoded_k = T.alloc_shared((bk, d), codec.packed_dtype)
                encoded_v = T.alloc_shared((bk, d), codec.packed_dtype)
                scale_k = T.alloc_shared((bk, 1), "float32")
                scale_v = T.alloc_shared((bk, 1), "float32")
                keys = T.view(kc, dtype=codec.packed_dtype)
                values = T.view(vc, dtype=codec.packed_dtype)
            scores = T.alloc_fragment((qm, bk), "float32")
            result = T.alloc_fragment((qm, d), "float32")
            maximum = T.alloc_fragment((qm,), "float32")
            previous = T.alloc_fragment((qm,), "float32")
            factor = T.alloc_fragment((qm,), "float32")
            normalizer = T.alloc_fragment((qm,), "float32")
            total = T.alloc_fragment((qm,), "float32")
            for i, j in T.Parallel(qm, d):
                query[i, j] = T.if_then_else(
                    qblock * qt + i // group < chunk,
                    q[
                        slot * chunk + T.min(qblock * qt + i // group, chunk - 1),
                        (head * group + i % group) * d + j,
                    ]
                    * (ks[slot, cache_head, 0, 0] if native else 1),
                    0,
                )
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            full_groups = length // bk // pg
            for block in T.serial(full_groups):
                for local in T.Pipelined(pg, num_stages=stages):
                    start = (block * pg + local) * bk
                    if native:
                        T.copy(kc[slot, cache_head, start, 0], key)
                        T.copy(vc[slot, cache_head, start, 0], value)
                    else:
                        T.copy(keys[slot, cache_head, start, 0], encoded_k)
                        T.copy(values[slot, cache_head, start, 0], encoded_v)
                        T.copy(ks[slot, cache_head, start, 0], scale_k)
                        T.copy(vs[slot, cache_head, start, 0], scale_v)
                        for i, j in T.Parallel(bk, d):
                            key[i, j] = decode(encoded_k, scale_k, i, j)
                            value[i, j] = decode(encoded_v, scale_v, i, j)
                    consume(
                        start,
                        positions,
                        slot,
                        qblock,
                        query,
                        key,
                        value,
                        prob,
                        scores,
                        result,
                        maximum,
                        previous,
                        factor,
                        normalizer,
                        total,
                    )
            key_tail = T.alloc_shared((bk, d), "float16")
            value_tail = T.alloc_shared((bk, d), "float16")
            prob_tail = T.alloc_shared((qm, bk), "float16")
            scores_tail = T.alloc_fragment((qm, bk), "float32")
            previous_tail = T.alloc_fragment((qm,), "float32")
            factor_tail = T.alloc_fragment((qm,), "float32")
            total_tail = T.alloc_fragment((qm,), "float32")
            for tile in T.serial(full_groups * pg, T.ceildiv(length, bk)):
                start = tile * bk
                for i, j in T.Parallel(bk, d):
                    key_tail[i, j] = T.if_then_else(
                        start + i < length,
                        read(kc, ks, slot, cache_head, T.min(start + i, cap - 1), j),
                        0,
                    )
                    value_tail[i, j] = T.if_then_else(
                        start + i < length,
                        read(vc, vs, slot, cache_head, T.min(start + i, cap - 1), j),
                        0,
                    )
                consume(
                    start,
                    positions,
                    slot,
                    qblock,
                    query,
                    key_tail,
                    value_tail,
                    prob_tail,
                    scores_tail,
                    result,
                    maximum,
                    previous_tail,
                    factor_tail,
                    normalizer,
                    total_tail,
                )
            for i, j in T.Parallel(qm, d):
                if qblock * qt + i // group < chunk:
                    row = slot * chunk + qblock * qt + i // group
                    out[(row * h + head * group + i % group) * d + j] = T.if_then_else(
                        enabled[row] != 0,
                        result[i, j]
                        / normalizer[i]
                        * (vs[slot, cache_head, 0, 0] if native else 1),
                        0,
                    )
        return out

    return compute


def partial_flat(p, splits):
    import tilelang.language as T

    h, kh, d, cap = (p[n] for n in ("h", "kh", "d", "cap"))
    if d != 64 or type(splits) is not int or splits < 1:
        raise ValueError("unsupported warp attention schedule")

    @kernel
    def compute(
        q: Tensor(h * d, "float32"),
        kc: Tensor(cap * kh * d, "float16"),
        vc: Tensor(cap * kh * d, "float16"),
        pos: Tensor(2, "int32"),
    ) -> Tensor(h * splits * 66, "float32"):
        parts = output(h * splits * 66, "float32", position=3)
        with T.Kernel(h, splits, threads=128) as (head, split):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            chunk = T.ceildiv(pos[0] + 1, splits)
            begin = split * chunk
            end = T.min(begin + chunk, pos[0] + 1)
            kvhead = head // (h // kh)
            q0 = q[head * 64 + lane]
            q1 = q[head * 64 + lane + 32]
            maxima = T.alloc_shared((4,), "float32")
            sums = T.alloc_shared((4,), "float32")
            values = T.alloc_shared((4, 64), "float32")
            maximum = T.alloc_var("float32")
            normalizer = T.alloc_var("float32")
            a0 = T.alloc_var("float32")
            a1 = T.alloc_var("float32")
            maximum = -T.infinity("float32")
            normalizer = 0
            a0 = 0
            a1 = 0
            for offset in T.serial(T.ceildiv(T.max(end - begin - warp, 0), 4)):
                token = begin + warp + offset * 4
                base = (token * kh + kvhead) * 64
                score = q0 * T.cast(kc[base + lane], "float32") + q1 * T.cast(
                    kc[base + lane + 32], "float32"
                )
                score = score + T.shfl_down(score, 16)
                score = score + T.shfl_down(score, 8)
                score = score + T.shfl_down(score, 4)
                score = score + T.shfl_down(score, 2)
                score = score + T.shfl_down(score, 1)
                score = T.shfl_sync(score, 0) * 0.125
                next_max = T.max(maximum, score)
                correction = T.exp(maximum - next_max)
                probability = T.exp(score - next_max)
                a0 = a0 * correction + probability * T.cast(vc[base + lane], "float32")
                a1 = a1 * correction + probability * T.cast(
                    vc[base + lane + 32], "float32"
                )
                normalizer = normalizer * correction + probability
                maximum = next_max
            values[warp, lane] = a0
            values[warp, lane + 32] = a1
            if lane == 0:
                maxima[warp] = maximum
                sums[warp] = normalizer
            T.sync_threads()
            if warp == 0:
                merged_max = T.max(
                    T.max(maxima[0], maxima[1]), T.max(maxima[2], maxima[3])
                )
                total = T.alloc_var("float32")
                result0 = T.alloc_var("float32")
                result1 = T.alloc_var("float32")
                total = 0
                result0 = 0
                result1 = 0
                for other in T.unroll(4):
                    factor = T.if_then_else(
                        T.isfinite(merged_max), T.exp(maxima[other] - merged_max), 0
                    )
                    total = total + sums[other] * factor
                    result0 = result0 + values[other, lane] * factor
                    result1 = result1 + values[other, lane + 32] * factor
                base = (head * splits + split) * 66
                parts[base + lane] = result0
                parts[base + lane + 32] = result1
                if lane == 0:
                    parts[base + 64] = merged_max
                    parts[base + 65] = total
        return parts

    return compute


def grouped_flat(p, stage=32, warps=2):
    import tilelang.language as T

    h, kh, d, cap, splits = (p[n] for n in ("h", "kh", "d", "cap", "splits"))
    group = h // kh
    threads = group * warps * 32
    if (
        d != 64
        or h % kh
        or group not in (2, 4)
        or (stage not in (16, 32, 64))
        or (warps not in (2, 4))
    ):
        raise ValueError("unsupported grouped attention schedule")

    @kernel
    def compute(
        q: Tensor(h * d, "float32"),
        kc: Tensor(cap * kh * d, "float16"),
        vc: Tensor(cap * kh * d, "float16"),
        pos: Tensor(2, "int32"),
    ) -> Tensor(h * splits * 66, "float32"):
        parts = output(h * splits * 66, "float32", position=3)
        with T.Kernel(kh, splits, threads=threads) as (kvhead, split):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            local = warp // warps
            worker = warp % warps
            head = kvhead * group + local
            chunk = T.ceildiv(pos[0] + 1, splits)
            begin = split * chunk
            end = T.min(begin + chunk, pos[0] + 1)
            keys = T.alloc_shared((stage, 64), "float16")
            vals = T.alloc_shared((stage, 64), "float16")
            maxima = T.alloc_shared((group * warps,), "float32")
            sums = T.alloc_shared((group * warps,), "float32")
            values = T.alloc_shared((group * warps, 64), "float32")
            q0 = q[head * 64 + lane]
            q1 = q[head * 64 + lane + 32]
            maximum = T.alloc_var("float32")
            normalizer = T.alloc_var("float32")
            a0 = T.alloc_var("float32")
            a1 = T.alloc_var("float32")
            maximum = -T.infinity("float32")
            normalizer = 0
            a0 = 0
            a1 = 0
            for block in T.serial(T.ceildiv(T.max(end - begin, 0), stage)):
                base = begin + block * stage
                for token, col in T.Parallel(stage, 64):
                    keys[token, col] = T.if_then_else(
                        base + token < end,
                        kc[((base + token) * kh + kvhead) * 64 + col],
                        0,
                    )
                    vals[token, col] = T.if_then_else(
                        base + token < end,
                        vc[((base + token) * kh + kvhead) * 64 + col],
                        0,
                    )
                T.sync_threads()
                for offset in T.serial(
                    T.ceildiv(T.max(T.min(stage, end - base) - worker, 0), warps)
                ):
                    token = worker + offset * warps
                    score = q0 * T.cast(keys[token, lane], "float32") + q1 * T.cast(
                        keys[token, lane + 32], "float32"
                    )
                    score = score + T.shfl_down(score, 16)
                    score = score + T.shfl_down(score, 8)
                    score = score + T.shfl_down(score, 4)
                    score = score + T.shfl_down(score, 2)
                    score = score + T.shfl_down(score, 1)
                    score = T.shfl_sync(score, 0) * 0.125
                    next_max = T.max(maximum, score)
                    correction = T.exp(maximum - next_max)
                    probability = T.exp(score - next_max)
                    a0 = a0 * correction + probability * T.cast(
                        vals[token, lane], "float32"
                    )
                    a1 = a1 * correction + probability * T.cast(
                        vals[token, lane + 32], "float32"
                    )
                    normalizer = normalizer * correction + probability
                    maximum = next_max
                T.sync_threads()
            values[warp, lane] = a0
            values[warp, lane + 32] = a1
            if lane == 0:
                maxima[warp] = maximum
                sums[warp] = normalizer
            T.sync_threads()
            if worker == 0:
                merged_max = T.alloc_var("float32")
                total = T.alloc_var("float32")
                result0 = T.alloc_var("float32")
                result1 = T.alloc_var("float32")
                merged_max = -T.infinity("float32")
                total = 0
                result0 = 0
                result1 = 0
                for other in T.unroll(warps):
                    merged_max = T.max(merged_max, maxima[local * warps + other])
                for other in T.unroll(warps):
                    index = local * warps + other
                    factor = T.if_then_else(
                        T.isfinite(merged_max), T.exp(maxima[index] - merged_max), 0
                    )
                    total = total + sums[index] * factor
                    result0 = result0 + values[index, lane] * factor
                    result1 = result1 + values[index, lane + 32] * factor
                base = (head * splits + split) * 66
                parts[base + lane] = result0
                parts[base + lane + 32] = result1
                if lane == 0:
                    parts[base + 64] = merged_max
                    parts[base + 65] = total
        return parts

    return compute


def partitioned_flat(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.attention.cuda import partial_flat as warp_partial_kernel

    p = {**spec.parameters, **schedule}
    return warp_partial_kernel(p, p["splits"])


def grouped(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.attention.cuda import grouped_flat as grouped_kernel

    p = {**spec.parameters, **schedule}
    return grouped_kernel(p, p.get("stage", 32), p.get("warps", 2))


def partial_bf16(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, cap, splits, d = (p["r"], p["capacity"], p.get("splits", 16), 256)

    @kernel
    def compute(
        q: Tensor((r, 16, d), "bfloat16"),
        kc: Tensor((r, 2, cap, d), "bfloat16"),
        vc: Tensor((r, 2, cap, d), "bfloat16"),
        positions: Tensor((r,), "int32"),
        active: Tensor((r,), "int32"),
    ) -> tuple[
        Tensor((r, 2, splits, 8, d), "float32"), Tensor((r, 2, splits, 8, 2), "float32")
    ]:
        out = output((r, 2, splits, 8, d), "float32")
        stats = output((r, 2, splits, 8, 2), "float32")
        with T.Kernel(r, 2, splits, threads=128) as (slot, head, part):
            query = T.alloc_shared((16, d), "bfloat16")
            key = T.alloc_shared((64, d), "bfloat16")
            value = T.alloc_shared((64, d), "bfloat16")
            prob = T.alloc_shared((16, 64), "bfloat16")
            scores = T.alloc_fragment((16, 64), "float32")
            result = T.alloc_fragment((16, d), "float32")
            maximum = T.alloc_fragment((16,), "float32")
            previous = T.alloc_fragment((16,), "float32")
            factor = T.alloc_fragment((16,), "float32")
            normalizer = T.alloc_fragment((16,), "float32")
            total = T.alloc_fragment((16,), "float32")
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            for i, j in T.Parallel(16, d):
                query[i, j] = T.if_then_else(i < 8, q[slot, head * 8 + i % 8, j], 0)
            count = positions[slot] + 1
            tiles = T.ceildiv(T.ceildiv(count, 64), splits)
            if (active[slot] != 0) & (part * tiles * 64 < count):
                for tile in T.serial(tiles):
                    for i, j in T.Parallel(64, d):
                        pos = (part * tiles + tile) * 64 + i
                        key[i, j] = T.if_then_else(
                            pos < count, kc[slot, head, T.min(pos, cap - 1), j], 0
                        )
                        value[i, j] = T.if_then_else(
                            pos < count, vc[slot, head, T.min(pos, cap - 1), j], 0
                        )
                    T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(16, 64):
                        scores[i, j] = T.if_then_else(
                            (part * tiles + tile) * 64 + j < count,
                            scores[i, j] * d ** (-0.5),
                            -T.infinity("float32"),
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(16):
                        factor[i] = T.exp(previous[i] - maximum[i])
                    for i, j in T.Parallel(16, 64):
                        scores[i, j] = T.exp(scores[i, j] - maximum[i])
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(16):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(16, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result)
            for i, j in T.Parallel(8, d):
                out[slot, head, part, i, j] = result[i, j]
            for i in T.Parallel(8):
                stats[slot, head, part, i, 0] = maximum[i]
                stats[slot, head, part, i, 1] = normalizer[i]
        return (out, stats)

    return compute


def partial_quantized(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.kv import get

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    format = get(p.get("kv_dtype", "fp8"))
    codec = format.bind(
        (
            p.get("pool", p.get("slots", p.get("r", 1))),
            p.get("kv_heads", 2),
            p["capacity"],
            p.get("d", 256),
        ),
        provider="cuda",
    )
    from akas.quantization._shared.access import cache_arguments, packed_arguments

    decode = cache_arguments(codec.read)
    packed_dtype = codec.packed_dtype
    decode_packed = (
        packed_arguments(codec.require("read_packed"))
        if p.get("packed_loads", False) and packed_dtype
        else None
    )
    r, cap, splits, d = (p["r"], p["capacity"], p.get("splits", 16), 256)
    packed = p.get("packed_loads", False)
    tile_rows = p.get("key_rows", 64)
    query_rows = 32 if tile_rows == 16 else 16
    if packed and cap % tile_rows:
        raise ValueError("packed KV requires capacity divisible by the key tile")
    storage = codec.storage
    if len(storage.metadata) != 1:
        raise ValueError("this attention ABI requires one metadata buffer per cache")
    cache_shape, cache_dtype = (storage.values.shape, storage.values.dtype)
    metadata_shape, metadata_dtype = (
        storage.metadata[0].shape,
        storage.metadata[0].dtype,
    )
    metadata_columns = metadata_shape[-1]
    if packed and (
        not packed_dtype or cache_shape[-1] != d or len(metadata_shape) != 4
    ):
        raise ValueError("KV codec does not support this packed attention load")

    @kernel
    def compute(
        q: Tensor((r, 16, d), "bfloat16"),
        kc: Tensor(cache_shape, cache_dtype),
        vc: Tensor(cache_shape, cache_dtype),
        ks: Tensor(metadata_shape, metadata_dtype),
        vs: Tensor(metadata_shape, metadata_dtype),
        positions: Tensor((r,), "int32"),
        active: Tensor((r,), "int32"),
    ) -> tuple[
        Tensor((r, 2, splits, 8, d), "float32"), Tensor((r, 2, splits, 8, 2), "float32")
    ]:
        out = output((r, 2, splits, 8, d), "float32")
        stats = output((r, 2, splits, 8, 2), "float32")
        with T.Kernel(r, 2, splits, threads=128) as (slot, head, part):
            query = T.alloc_shared((query_rows, d), "bfloat16")
            key = T.alloc_shared((tile_rows, d), "bfloat16")
            value = T.alloc_shared((tile_rows, d), "bfloat16")
            prob = T.alloc_shared((query_rows, tile_rows), "bfloat16")
            if packed:
                encoded = T.alloc_shared((tile_rows, d), packed_dtype)
                scales = T.alloc_shared((tile_rows, metadata_columns), metadata_dtype)
            scores = T.alloc_fragment((query_rows, tile_rows), "float32")
            result = T.alloc_fragment((query_rows, d), "float32")
            maximum = T.alloc_fragment((query_rows,), "float32")
            previous = T.alloc_fragment((query_rows,), "float32")
            factor = T.alloc_fragment((query_rows,), "float32")
            normalizer = T.alloc_fragment((query_rows,), "float32")
            total = T.alloc_fragment((query_rows,), "float32")
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            for i, j in T.Parallel(query_rows, d):
                query[i, j] = T.if_then_else(i < 8, q[slot, head * 8 + i % 8, j], 0)
            count = positions[slot] + 1
            tiles = T.ceildiv(T.ceildiv(count, tile_rows), splits)
            if (active[slot] != 0) & (part * tiles * tile_rows < count):
                for tile in T.serial(tiles):
                    if packed:
                        start = (part * tiles + tile) * tile_rows
                        if start < cap:
                            keys = T.view(kc, dtype=packed_dtype)
                            values = T.view(vc, dtype=packed_dtype)
                            T.copy(
                                keys[slot, head, start : start + tile_rows, 0:d],
                                encoded,
                            )
                            T.copy(
                                ks[
                                    slot,
                                    head,
                                    start : start + tile_rows,
                                    0:metadata_columns,
                                ],
                                scales,
                            )
                            for i, j in T.Parallel(tile_rows, d):
                                key[i, j] = T.if_then_else(
                                    start + i < count,
                                    decode_packed(encoded, scales, i, j),
                                    0,
                                )
                            T.copy(
                                values[slot, head, start : start + tile_rows, 0:d],
                                encoded,
                            )
                            T.copy(
                                vs[
                                    slot,
                                    head,
                                    start : start + tile_rows,
                                    0:metadata_columns,
                                ],
                                scales,
                            )
                            for i, j in T.Parallel(tile_rows, d):
                                value[i, j] = T.if_then_else(
                                    start + i < count,
                                    decode_packed(encoded, scales, i, j),
                                    0,
                                )
                        else:
                            T.clear(key)
                            T.clear(value)
                    else:
                        for i, j in T.Parallel(tile_rows, d):
                            pos = (part * tiles + tile) * tile_rows + i
                            key[i, j] = T.if_then_else(
                                pos < count,
                                decode(kc, ks, slot, head, T.min(pos, cap - 1), j),
                                0,
                            )
                            value[i, j] = T.if_then_else(
                                pos < count,
                                decode(vc, vs, slot, head, T.min(pos, cap - 1), j),
                                0,
                            )
                    T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(query_rows, tile_rows):
                        scores[i, j] = T.if_then_else(
                            (part * tiles + tile) * tile_rows + j < count,
                            scores[i, j] * d ** (-0.5),
                            -T.infinity("float32"),
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(query_rows):
                        factor[i] = T.exp(previous[i] - maximum[i])
                    for i, j in T.Parallel(query_rows, tile_rows):
                        scores[i, j] = T.exp(scores[i, j] - maximum[i])
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(query_rows):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(query_rows, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result)
            for i, j in T.Parallel(8, d):
                out[slot, head, part, i, j] = result[i, j]
            for i in T.Parallel(8):
                stats[slot, head, part, i, 0] = maximum[i]
                stats[slot, head, part, i, 1] = normalizer[i]
        return (out, stats)

    return compute


def attention(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.kv import get

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    codec = get("fp8").bind(
        (
            p.get("pool", p.get("slots", p.get("r", 1))),
            p.get("kv_heads", 2),
            p["capacity"],
            p.get("d", 256),
        ),
        provider="cuda",
    )
    from akas.quantization._shared.access import cache_arguments, packed_arguments

    decode_scalar = cache_arguments(codec.read)
    from akas.quantization._shared.access import packed_arguments

    decode_packed = packed_arguments(codec.require("read_packed"))
    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    d, cap = (256, p["capacity"])
    qrows, kvrows, stride = (p.get("query_rows", 128), 32, 16)
    threads = p.get("threads", 256)
    joint = p.get("joint_kv", False)
    decoded = p.get("decoded_kv", False)
    value_splits = p.get("value_splits", 1)
    if value_splits not in (1, 2) or (value_splits != 1 and (not decoded)):
        raise ValueError("split value dimensions require decoded KV")
    vd = d // value_splits
    cache_dtype = "bfloat16" if decoded else "uint8"
    if qrows not in (64, 128, 256) or threads not in (128, 256, 512):
        raise ValueError("unsupported Hopper attention geometry")
    token_rows = qrows // 8
    packed = p.get("packed_loads", False)
    policy = T.GemmWarpPolicy.FullRow if qrows >= 128 else T.GemmWarpPolicy.Square
    if p.get("square_warps", False):
        policy = T.GemmWarpPolicy.Square
    if packed and cap % kvrows:
        raise ValueError("packed KV requires capacity divisible by key tile")

    @T.macro
    def algorithm(q, kc, vc, ks, vs, projection, positions, lengths, out):
        with T.Kernel(
            slots, 2, T.ceildiv(chunk, token_rows) * value_splits, threads=threads
        ) as (slot, head, group):
            tile = group // value_splits
            value_part = group % value_splits
            if tile * token_rows < lengths[slot]:
                query = T.alloc_shared((qrows, d), "bfloat16")
                key = T.alloc_shared((kvrows, d), "bfloat16")
                value = T.alloc_shared((kvrows, vd), "bfloat16")
                prob = T.alloc_shared((qrows, kvrows), "bfloat16")
                if packed and (not decoded):
                    encoded = T.alloc_shared((16, d), codec.packed_dtype)
                    scales = T.alloc_shared((16, 2), "float32")
                    if joint:
                        value_encoded = T.alloc_shared((16, d), codec.packed_dtype)
                        value_scales = T.alloc_shared((16, 2), "float32")
                scores = T.alloc_fragment((qrows, kvrows), "float32")
                result = T.alloc_fragment((qrows, vd), "float32")
                maximum = T.alloc_fragment((qrows,), "float32")
                previous = T.alloc_fragment((qrows,), "float32")
                factor = T.alloc_fragment((qrows,), "float32")
                normalizer = T.alloc_fragment((qrows,), "float32")
                total = T.alloc_fragment((qrows,), "float32")
                T.clear(result)
                T.clear(normalizer)
                T.fill(maximum, -T.infinity("float32"))
                if joint or decoded:
                    for i, j in T.Parallel(16, d):
                        key[i + 16, j] = 0
                    for i, j in T.Parallel(16, vd):
                        value[i + 16, j] = 0
                for i, j in T.Parallel(qrows, d):
                    local = tile * token_rows + i // 8
                    query[i, j] = T.if_then_else(
                        local < lengths[slot],
                        q[slot * chunk + T.min(local, chunk - 1), head * 8 + i % 8, j],
                        0,
                    )
                count = positions[slot] + T.min(
                    lengths[slot], tile * token_rows + token_rows
                )
                for step in T.serial(T.ceildiv(count, stride)):
                    if decoded:
                        start = step * stride
                        T.copy(kc[slot, head, start : start + 16, 0:d], key[0:16, 0:d])
                        T.copy(
                            vc[
                                slot,
                                head,
                                start : start + 16,
                                value_part * vd : value_part * vd + vd,
                            ],
                            value[0:16, 0:vd],
                        )
                    elif packed:
                        start = step * stride
                        keys = T.view(kc, dtype=codec.packed_dtype)
                        values = T.view(vc, dtype=codec.packed_dtype)
                        T.copy(keys[slot, head, start : start + 16, 0:d], encoded)
                        T.copy(ks[slot, head, start : start + 16, 0:2], scales)
                        if joint:
                            T.copy(
                                values[slot, head, start : start + 16, 0:d],
                                value_encoded,
                            )
                            T.copy(
                                vs[slot, head, start : start + 16, 0:2], value_scales
                            )
                            T.sync_threads()
                            for i, j in T.Parallel(16, d):
                                key[i, j] = 0
                                value[i, j] = 0
                                if start + i < count:
                                    key[i, j] = decode_packed(encoded, scales, i, j)
                                    value[i, j] = decode_packed(
                                        value_encoded, value_scales, i, j
                                    )
                        else:
                            T.sync_threads()
                            for i, j in T.Parallel(16, d):
                                key[i, j] = 0
                                if start + i < count:
                                    key[i, j] = decode_packed(encoded, scales, i, j)
                                key[i + 16, j] = 0
                            T.sync_threads()
                            T.copy(values[slot, head, start : start + 16, 0:d], encoded)
                            T.copy(vs[slot, head, start : start + 16, 0:2], scales)
                            T.sync_threads()
                            for i, j in T.Parallel(16, d):
                                value[i, j] = 0
                                if start + i < count:
                                    value[i, j] = decode_packed(encoded, scales, i, j)
                                value[i + 16, j] = 0
                        T.sync_threads()
                    else:
                        for i, j in T.Parallel(kvrows, d):
                            index = step * stride + i
                            key[i, j] = T.if_then_else(
                                (i < stride) & (index < count),
                                decode_scalar(
                                    kc, ks, slot, head, T.min(index, cap - 1), j
                                ),
                                0,
                            )
                            value[i, j] = T.if_then_else(
                                (i < stride) & (index < count),
                                decode_scalar(
                                    vc, vs, slot, head, T.min(index, cap - 1), j
                                ),
                                0,
                            )
                    T.sync_threads()
                    T.gemm(
                        query,
                        key,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=policy,
                    )
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(qrows, kvrows):
                        valid = (
                            (j < stride)
                            & (
                                step * stride + j
                                <= positions[slot] + tile * token_rows + i // 8
                            )
                            & (step * stride + j < count)
                        )
                        scores[i, j] = T.if_then_else(
                            valid, scores[i, j] / 16, -T.infinity("float32")
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(qrows):
                        factor[i] = T.exp(previous[i] - maximum[i])
                    for i, j in T.Parallel(qrows, kvrows):
                        scores[i, j] = T.exp(scores[i, j] - maximum[i])
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(qrows):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(qrows, vd):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.sync_threads()
                    T.gemm(prob, value, result, policy=policy)
                for i, j in T.Parallel(qrows, vd):
                    local = tile * token_rows + i // 8
                    if local < lengths[slot]:
                        row = slot * chunk + local
                        gate = T.cast(
                            T.cast(
                                projection[
                                    row,
                                    (head * 8 + i % 8) * 512 + d + value_part * vd + j,
                                ],
                                "bfloat16",
                            ),
                            "float32",
                        )
                        normalized = T.cast(
                            T.cast(result[i, j] / normalizer[i], "bfloat16"), "float32"
                        )
                        out[row, (head * 8 + i % 8) * d + value_part * vd + j] = (
                            normalized / (1 + T.exp(-gate))
                        )

    if decoded:

        @kernel
        def compute(
            q: Tensor((rows, 16, d), "bfloat16"),
            kc: Tensor((slots, 2, cap, d), cache_dtype),
            vc: Tensor((slots, 2, cap, d), cache_dtype),
            projection: Tensor((rows, 8192), "float32"),
            positions: Tensor((slots,), "int32"),
            lengths: Tensor((slots,), "int32"),
        ) -> Tensor((rows, 4096), "bfloat16"):
            out = output((rows, 4096), "bfloat16")
            algorithm(q, kc, vc, None, None, projection, positions, lengths, out)
            return out

        return compute

    @kernel
    def compute(
        q: Tensor((rows, 16, d), "bfloat16"),
        kc: Tensor((slots, 2, cap, d), cache_dtype),
        vc: Tensor((slots, 2, cap, d), cache_dtype),
        ks: Tensor((slots, 2, cap, 2), "float32"),
        vs: Tensor((slots, 2, cap, 2), "float32"),
        projection: Tensor((rows, 8192), "float32"),
        positions: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
    ) -> Tensor((rows, 4096), "bfloat16"):
        out = output((rows, 4096), "bfloat16")
        with T.Kernel(
            slots, 2, T.ceildiv(chunk, token_rows) * value_splits, threads=threads
        ) as (slot, head, group):
            tile = group // value_splits
            value_part = group % value_splits
            if tile * token_rows < lengths[slot]:
                query = T.alloc_shared((qrows, d), "bfloat16")
                key = T.alloc_shared((kvrows, d), "bfloat16")
                value = T.alloc_shared((kvrows, vd), "bfloat16")
                prob = T.alloc_shared((qrows, kvrows), "bfloat16")
                if packed and (not decoded):
                    encoded = T.alloc_shared((16, d), codec.packed_dtype)
                    scales = T.alloc_shared((16, 2), "float32")
                    if joint:
                        value_encoded = T.alloc_shared((16, d), codec.packed_dtype)
                        value_scales = T.alloc_shared((16, 2), "float32")
                scores = T.alloc_fragment((qrows, kvrows), "float32")
                result = T.alloc_fragment((qrows, vd), "float32")
                maximum = T.alloc_fragment((qrows,), "float32")
                previous = T.alloc_fragment((qrows,), "float32")
                factor = T.alloc_fragment((qrows,), "float32")
                normalizer = T.alloc_fragment((qrows,), "float32")
                total = T.alloc_fragment((qrows,), "float32")
                T.clear(result)
                T.clear(normalizer)
                T.fill(maximum, -T.infinity("float32"))
                if joint or decoded:
                    for i, j in T.Parallel(16, d):
                        key[i + 16, j] = 0
                    for i, j in T.Parallel(16, vd):
                        value[i + 16, j] = 0
                for i, j in T.Parallel(qrows, d):
                    local = tile * token_rows + i // 8
                    query[i, j] = T.if_then_else(
                        local < lengths[slot],
                        q[slot * chunk + T.min(local, chunk - 1), head * 8 + i % 8, j],
                        0,
                    )
                count = positions[slot] + T.min(
                    lengths[slot], tile * token_rows + token_rows
                )
                for step in T.serial(T.ceildiv(count, stride)):
                    if decoded:
                        start = step * stride
                        T.copy(kc[slot, head, start : start + 16, 0:d], key[0:16, 0:d])
                        T.copy(
                            vc[
                                slot,
                                head,
                                start : start + 16,
                                value_part * vd : value_part * vd + vd,
                            ],
                            value[0:16, 0:vd],
                        )
                    elif packed:
                        start = step * stride
                        keys = T.view(kc, dtype=codec.packed_dtype)
                        values = T.view(vc, dtype=codec.packed_dtype)
                        T.copy(keys[slot, head, start : start + 16, 0:d], encoded)
                        T.copy(ks[slot, head, start : start + 16, 0:2], scales)
                        if joint:
                            T.copy(
                                values[slot, head, start : start + 16, 0:d],
                                value_encoded,
                            )
                            T.copy(
                                vs[slot, head, start : start + 16, 0:2], value_scales
                            )
                            T.sync_threads()
                            for i, j in T.Parallel(16, d):
                                key[i, j] = 0
                                value[i, j] = 0
                                if start + i < count:
                                    key[i, j] = decode_packed(encoded, scales, i, j)
                                    value[i, j] = decode_packed(
                                        value_encoded, value_scales, i, j
                                    )
                        else:
                            T.sync_threads()
                            for i, j in T.Parallel(16, d):
                                key[i, j] = 0
                                if start + i < count:
                                    key[i, j] = decode_packed(encoded, scales, i, j)
                                key[i + 16, j] = 0
                            T.sync_threads()
                            T.copy(values[slot, head, start : start + 16, 0:d], encoded)
                            T.copy(vs[slot, head, start : start + 16, 0:2], scales)
                            T.sync_threads()
                            for i, j in T.Parallel(16, d):
                                value[i, j] = 0
                                if start + i < count:
                                    value[i, j] = decode_packed(encoded, scales, i, j)
                                value[i + 16, j] = 0
                        T.sync_threads()
                    else:
                        for i, j in T.Parallel(kvrows, d):
                            index = step * stride + i
                            key[i, j] = T.if_then_else(
                                (i < stride) & (index < count),
                                decode_scalar(
                                    kc, ks, slot, head, T.min(index, cap - 1), j
                                ),
                                0,
                            )
                            value[i, j] = T.if_then_else(
                                (i < stride) & (index < count),
                                decode_scalar(
                                    vc, vs, slot, head, T.min(index, cap - 1), j
                                ),
                                0,
                            )
                    T.sync_threads()
                    T.gemm(
                        query,
                        key,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=policy,
                    )
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(qrows, kvrows):
                        valid = (
                            (j < stride)
                            & (
                                step * stride + j
                                <= positions[slot] + tile * token_rows + i // 8
                            )
                            & (step * stride + j < count)
                        )
                        scores[i, j] = T.if_then_else(
                            valid, scores[i, j] / 16, -T.infinity("float32")
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(qrows):
                        factor[i] = T.exp(previous[i] - maximum[i])
                    for i, j in T.Parallel(qrows, kvrows):
                        scores[i, j] = T.exp(scores[i, j] - maximum[i])
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(qrows):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(qrows, vd):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.sync_threads()
                    T.gemm(prob, value, result, policy=policy)
                for i, j in T.Parallel(qrows, vd):
                    local = tile * token_rows + i // 8
                    if local < lengths[slot]:
                        row = slot * chunk + local
                        gate = T.cast(
                            T.cast(
                                projection[
                                    row,
                                    (head * 8 + i % 8) * 512 + d + value_part * vd + j,
                                ],
                                "bfloat16",
                            ),
                            "float32",
                        )
                        normalized = T.cast(
                            T.cast(result[i, j] / normalizer[i], "bfloat16"), "float32"
                        )
                        out[row, (head * 8 + i % 8) * d + value_part * vd + j] = (
                            normalized / (1 + T.exp(-gate))
                        )
        return out

    return compute


def hopper_prefill_split_partial(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.kv import get

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    codec = get("fp8").bind(
        (
            p.get("pool", p.get("slots", p.get("r", 1))),
            p.get("kv_heads", 2),
            p["capacity"],
            p.get("d", 256),
        ),
        provider="cuda",
    )
    from akas.quantization._shared.access import cache_arguments, packed_arguments

    decode_scalar = cache_arguments(codec.read)
    from akas.quantization._shared.access import packed_arguments

    decode_packed = packed_arguments(codec.require("read_packed"))
    r, chunk, cap, splits, d = (
        p["slots"],
        p["chunk"],
        p["capacity"],
        p.get("splits", 16),
        256,
    )
    rows = r * chunk
    packed = p.get("packed_loads", False)
    decoded = p.get("decoded_kv", False)
    tile_rows = p.get("key_rows", 64)
    query_tokens = p.get("query_tokens", 8)
    if not p.get("pad_queries", False):
        query_tokens = min(chunk, query_tokens)
    query_rows = max(16, query_tokens * 8)
    query_tiles = T.ceildiv(chunk, query_tokens)
    threads = p.get("threads", 128 if query_rows <= 64 else 256)
    policy = T.GemmWarpPolicy.FullRow if query_rows >= 128 else T.GemmWarpPolicy.Square
    if packed and cap % tile_rows:
        raise ValueError("packed KV requires capacity divisible by the key tile")

    @T.macro
    def algorithm(q, kc, vc, ks, vs, positions, lengths, out, stats):
        with T.Kernel(r, 2, splits * query_tiles, threads=threads) as (
            slot,
            head,
            group,
        ):
            part = group // query_tiles
            query_tile = group % query_tiles
            query = T.alloc_shared((query_rows, d), "bfloat16")
            key = T.alloc_shared((tile_rows, d), "bfloat16")
            value = T.alloc_shared((tile_rows, d), "bfloat16")
            prob = T.alloc_shared((query_rows, tile_rows), "bfloat16")
            if packed and (not decoded):
                encoded = T.alloc_shared((tile_rows, d), codec.packed_dtype)
                scales = T.alloc_shared((tile_rows, 2), "float32")
            scores = T.alloc_fragment((query_rows, tile_rows), "float32")
            result = T.alloc_fragment((query_rows, d), "float32")
            maximum = T.alloc_fragment((query_rows,), "float32")
            previous = T.alloc_fragment((query_rows,), "float32")
            factor = T.alloc_fragment((query_rows,), "float32")
            normalizer = T.alloc_fragment((query_rows,), "float32")
            total = T.alloc_fragment((query_rows,), "float32")
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            for i, j in T.Parallel(query_rows, d):
                query[i, j] = T.if_then_else(
                    query_tile * query_tokens + i // 8 < lengths[slot],
                    q[
                        slot * chunk
                        + T.min(query_tile * query_tokens + i // 8, chunk - 1),
                        head * 8 + i % 8,
                        j,
                    ],
                    0,
                )
            count = positions[slot] + lengths[slot]
            tiles = T.ceildiv(T.ceildiv(count, tile_rows), splits)
            if (lengths[slot] > query_tile * query_tokens) & (
                part * tiles * tile_rows < count
            ):
                for tile in T.serial(tiles):
                    if decoded:
                        start = (part * tiles + tile) * tile_rows
                        if start < cap:
                            T.copy(kc[slot, head, start : start + tile_rows, 0:d], key)
                            T.copy(
                                vc[slot, head, start : start + tile_rows, 0:d], value
                            )
                        else:
                            T.clear(key)
                            T.clear(value)
                    elif packed:
                        start = (part * tiles + tile) * tile_rows
                        if start < cap:
                            keys = T.view(kc, dtype=codec.packed_dtype)
                            values = T.view(vc, dtype=codec.packed_dtype)
                            T.copy(
                                keys[slot, head, start : start + tile_rows, 0:d],
                                encoded,
                            )
                            T.copy(
                                ks[slot, head, start : start + tile_rows, 0:2], scales
                            )
                            for i, j in T.Parallel(tile_rows, d):
                                key[i, j] = T.if_then_else(
                                    start + i < count,
                                    decode_packed(encoded, scales, i, j),
                                    0,
                                )
                            T.copy(
                                values[slot, head, start : start + tile_rows, 0:d],
                                encoded,
                            )
                            T.copy(
                                vs[slot, head, start : start + tile_rows, 0:2], scales
                            )
                            for i, j in T.Parallel(tile_rows, d):
                                value[i, j] = T.if_then_else(
                                    start + i < count,
                                    decode_packed(encoded, scales, i, j),
                                    0,
                                )
                        else:
                            T.clear(key)
                            T.clear(value)
                    else:
                        for i, j in T.Parallel(tile_rows, d):
                            pos = (part * tiles + tile) * tile_rows + i
                            key[i, j] = T.if_then_else(
                                pos < count,
                                decode_scalar(
                                    kc, ks, slot, head, T.min(pos, cap - 1), j
                                ),
                                0,
                            )
                            value[i, j] = T.if_then_else(
                                pos < count,
                                decode_scalar(
                                    vc, vs, slot, head, T.min(pos, cap - 1), j
                                ),
                                0,
                            )
                    T.gemm(
                        query,
                        key,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=policy,
                    )
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(query_rows, tile_rows):
                        scores[i, j] = T.if_then_else(
                            ((part * tiles + tile) * tile_rows + j < count)
                            & (
                                (part * tiles + tile) * tile_rows + j
                                <= positions[slot] + query_tile * query_tokens + i // 8
                            ),
                            scores[i, j] * d ** (-0.5),
                            -T.infinity("float32"),
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(query_rows):
                        factor[i] = T.if_then_else(
                            previous[i] > -T.infinity("float32"),
                            T.exp(previous[i] - maximum[i]),
                            0,
                        )
                    for i, j in T.Parallel(query_rows, tile_rows):
                        scores[i, j] = T.if_then_else(
                            scores[i, j] > -T.infinity("float32"),
                            T.exp(scores[i, j] - maximum[i]),
                            0,
                        )
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(query_rows):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(query_rows, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result, policy=policy)
            for i, j in T.Parallel(query_tokens * 8, d):
                if query_tile * query_tokens * 8 + i < chunk * 8:
                    out[slot, head, part, query_tile * query_tokens * 8 + i, j] = (
                        result[i, j]
                    )
            for i in T.Parallel(query_tokens * 8):
                if query_tile * query_tokens * 8 + i < chunk * 8:
                    stats[slot, head, part, query_tile * query_tokens * 8 + i, 0] = (
                        maximum[i]
                    )
                    stats[slot, head, part, query_tile * query_tokens * 8 + i, 1] = (
                        normalizer[i]
                    )

    if decoded:

        @kernel
        def compute(
            q: Tensor((rows, 16, d), "bfloat16"),
            kc: Tensor((r, 2, cap, d), "bfloat16"),
            vc: Tensor((r, 2, cap, d), "bfloat16"),
            positions: Tensor((r,), "int32"),
            lengths: Tensor((r,), "int32"),
        ) -> tuple[
            Tensor((r, 2, splits, chunk * 8, d), "float32"),
            Tensor((r, 2, splits, chunk * 8, 2), "float32"),
        ]:
            out = output((r, 2, splits, chunk * 8, d), "float32")
            stats = output((r, 2, splits, chunk * 8, 2), "float32")
            algorithm(q, kc, vc, kc, vc, positions, lengths, out, stats)
            return (out, stats)

        return compute
    if not decoded:

        @kernel
        def compute(
            q: Tensor((rows, 16, d), "bfloat16"),
            kc: Tensor((r, 2, cap, d), "bfloat16" if decoded else "uint8"),
            vc: Tensor((r, 2, cap, d), "bfloat16" if decoded else "uint8"),
            ks: Tensor((r, 2, cap, 2), "float32"),
            vs: Tensor((r, 2, cap, 2), "float32"),
            positions: Tensor((r,), "int32"),
            lengths: Tensor((r,), "int32"),
        ) -> tuple[
            Tensor((r, 2, splits, chunk * 8, d), "float32"),
            Tensor((r, 2, splits, chunk * 8, 2), "float32"),
        ]:
            out = output((r, 2, splits, chunk * 8, d), "float32")
            stats = output((r, 2, splits, chunk * 8, 2), "float32")
            with T.Kernel(r, 2, splits * query_tiles, threads=threads) as (
                slot,
                head,
                group,
            ):
                part = group // query_tiles
                query_tile = group % query_tiles
                query = T.alloc_shared((query_rows, d), "bfloat16")
                key = T.alloc_shared((tile_rows, d), "bfloat16")
                value = T.alloc_shared((tile_rows, d), "bfloat16")
                prob = T.alloc_shared((query_rows, tile_rows), "bfloat16")
                if packed and (not decoded):
                    encoded = T.alloc_shared((tile_rows, d), codec.packed_dtype)
                    scales = T.alloc_shared((tile_rows, 2), "float32")
                scores = T.alloc_fragment((query_rows, tile_rows), "float32")
                result = T.alloc_fragment((query_rows, d), "float32")
                maximum = T.alloc_fragment((query_rows,), "float32")
                previous = T.alloc_fragment((query_rows,), "float32")
                factor = T.alloc_fragment((query_rows,), "float32")
                normalizer = T.alloc_fragment((query_rows,), "float32")
                total = T.alloc_fragment((query_rows,), "float32")
                T.clear(result)
                T.clear(normalizer)
                T.fill(maximum, -T.infinity("float32"))
                for i, j in T.Parallel(query_rows, d):
                    query[i, j] = T.if_then_else(
                        query_tile * query_tokens + i // 8 < lengths[slot],
                        q[
                            slot * chunk
                            + T.min(query_tile * query_tokens + i // 8, chunk - 1),
                            head * 8 + i % 8,
                            j,
                        ],
                        0,
                    )
                count = positions[slot] + lengths[slot]
                tiles = T.ceildiv(T.ceildiv(count, tile_rows), splits)
                if (lengths[slot] > query_tile * query_tokens) & (
                    part * tiles * tile_rows < count
                ):
                    for tile in T.serial(tiles):
                        if decoded:
                            start = (part * tiles + tile) * tile_rows
                            if start < cap:
                                T.copy(
                                    kc[slot, head, start : start + tile_rows, 0:d], key
                                )
                                T.copy(
                                    vc[slot, head, start : start + tile_rows, 0:d],
                                    value,
                                )
                            else:
                                T.clear(key)
                                T.clear(value)
                        elif packed:
                            start = (part * tiles + tile) * tile_rows
                            if start < cap:
                                keys = T.view(kc, dtype=codec.packed_dtype)
                                values = T.view(vc, dtype=codec.packed_dtype)
                                T.copy(
                                    keys[slot, head, start : start + tile_rows, 0:d],
                                    encoded,
                                )
                                T.copy(
                                    ks[slot, head, start : start + tile_rows, 0:2],
                                    scales,
                                )
                                for i, j in T.Parallel(tile_rows, d):
                                    key[i, j] = T.if_then_else(
                                        start + i < count,
                                        decode_packed(encoded, scales, i, j),
                                        0,
                                    )
                                T.copy(
                                    values[slot, head, start : start + tile_rows, 0:d],
                                    encoded,
                                )
                                T.copy(
                                    vs[slot, head, start : start + tile_rows, 0:2],
                                    scales,
                                )
                                for i, j in T.Parallel(tile_rows, d):
                                    value[i, j] = T.if_then_else(
                                        start + i < count,
                                        decode_packed(encoded, scales, i, j),
                                        0,
                                    )
                            else:
                                T.clear(key)
                                T.clear(value)
                        else:
                            for i, j in T.Parallel(tile_rows, d):
                                pos = (part * tiles + tile) * tile_rows + i
                                key[i, j] = T.if_then_else(
                                    pos < count,
                                    decode_scalar(
                                        kc, ks, slot, head, T.min(pos, cap - 1), j
                                    ),
                                    0,
                                )
                                value[i, j] = T.if_then_else(
                                    pos < count,
                                    decode_scalar(
                                        vc, vs, slot, head, T.min(pos, cap - 1), j
                                    ),
                                    0,
                                )
                        T.gemm(
                            query,
                            key,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=policy,
                        )
                        T.copy(maximum, previous)
                        for i, j in T.Parallel(query_rows, tile_rows):
                            scores[i, j] = T.if_then_else(
                                ((part * tiles + tile) * tile_rows + j < count)
                                & (
                                    (part * tiles + tile) * tile_rows + j
                                    <= positions[slot]
                                    + query_tile * query_tokens
                                    + i // 8
                                ),
                                scores[i, j] * d ** (-0.5),
                                -T.infinity("float32"),
                            )
                        T.reduce_max(scores, maximum, dim=1, clear=False)
                        for i in T.Parallel(query_rows):
                            factor[i] = T.if_then_else(
                                previous[i] > -T.infinity("float32"),
                                T.exp(previous[i] - maximum[i]),
                                0,
                            )
                        for i, j in T.Parallel(query_rows, tile_rows):
                            scores[i, j] = T.if_then_else(
                                scores[i, j] > -T.infinity("float32"),
                                T.exp(scores[i, j] - maximum[i]),
                                0,
                            )
                        T.reduce_sum(scores, total, dim=1)
                        for i in T.Parallel(query_rows):
                            normalizer[i] = normalizer[i] * factor[i] + total[i]
                        for i, j in T.Parallel(query_rows, d):
                            result[i, j] *= factor[i]
                        T.copy(scores, prob)
                        T.gemm(prob, value, result, policy=policy)
                for i, j in T.Parallel(query_tokens * 8, d):
                    if query_tile * query_tokens * 8 + i < chunk * 8:
                        out[slot, head, part, query_tile * query_tokens * 8 + i, j] = (
                            result[i, j]
                        )
                for i in T.Parallel(query_tokens * 8):
                    if query_tile * query_tokens * 8 + i < chunk * 8:
                        stats[
                            slot, head, part, query_tile * query_tokens * 8 + i, 0
                        ] = maximum[i]
                        stats[
                            slot, head, part, query_tile * query_tokens * 8 + i, 1
                        ] = normalizer[i]
            return (out, stats)

        return compute


def flat(p, splits):
    import tilelang.language as T

    h, d = (p["h"], p["d"])
    stride = d + 2

    @kernel
    def compute(parts: Tensor(h * splits * stride, "float32")) -> Tensor(
        h * d, "float32"
    ):
        out = output(h * d, "float32")
        with T.Kernel(h, threads=128) as head:
            maxima = T.alloc_fragment((splits,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            sums = T.alloc_fragment((splits,), "float32")
            normalizer = T.alloc_fragment((1,), "float32")
            products = T.alloc_fragment((splits, d), "float32")
            result = T.alloc_fragment((d,), "float32")
            for i in T.Parallel(splits):
                maxima[i] = parts[(head * splits + i) * stride + d]
            T.reduce_max(maxima, maximum, dim=0)
            for i in T.Parallel(splits):
                sums[i] = parts[(head * splits + i) * stride + (d + 1)] * T.exp(
                    maxima[i] - maximum[0]
                )
            T.reduce_sum(sums, normalizer, dim=0)
            for i, j in T.Parallel(splits, d):
                products[i, j] = parts[(head * splits + i) * stride + j] * T.exp(
                    maxima[i] - maximum[0]
                )
            T.reduce_sum(products, result, dim=0)
            for j in T.Parallel(d):
                out[head * d + j] = result[j] / normalizer[0]
        return out

    return compute


def merge_partitioned_flat(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.attention.cuda import flat as merge_kernel

    p = {**spec.parameters, **schedule}
    return merge_kernel(p, p["splits"])


def merge_native(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, splits, d = (p["r"], p.get("splits", 16), 256)

    @kernel
    def compute(
        partial: Tensor((r, 2, splits, 8, d), "float32"),
        stats: Tensor((r, 2, splits, 8, 2), "float32"),
        projection: Tensor((r, 8192), "float32"),
        active: Tensor((r,), "int32"),
    ) -> Tensor((r, 4096), "bfloat16"):
        out = output((r, 4096), "bfloat16")
        with T.Kernel(r, 16, threads=256) as (slot, head):
            for j in T.Parallel(d):
                maximum = T.alloc_var("float32")
                numerator = T.alloc_var("float32")
                denominator = T.alloc_var("float32")
                maximum = -T.infinity("float32")
                for part in T.serial(splits):
                    maximum = T.max(maximum, stats[slot, head // 8, part, head % 8, 0])
                numerator = 0
                denominator = 0
                if active[slot] != 0:
                    for part in T.serial(splits):
                        factor = T.exp(
                            stats[slot, head // 8, part, head % 8, 0] - maximum
                        )
                        numerator += (
                            partial[slot, head // 8, part, head % 8, j] * factor
                        )
                        denominator += (
                            stats[slot, head // 8, part, head % 8, 1] * factor
                        )
                    gate = T.cast(
                        T.cast(projection[slot, head * 512 + d + j], "bfloat16"),
                        "float32",
                    )
                    value = T.cast(
                        T.cast(numerator / denominator, "bfloat16"), "float32"
                    )
                    out[slot, head * d + j] = value / (1 + T.exp(-gate))
                else:
                    out[slot, head * d + j] = 0
        return out

    return compute


def chunk(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, chunk, splits, d = (p["slots"], p["chunk"], p.get("splits", 16), 256)
    rows = r * chunk

    @kernel
    def compute(
        partial: Tensor((r, 2, splits, chunk * 8, d), "float32"),
        stats: Tensor((r, 2, splits, chunk * 8, 2), "float32"),
        projection: Tensor((rows, 8192), "float32"),
        active: Tensor((rows,), "int32"),
    ) -> Tensor((rows, 4096), "bfloat16"):
        out = output((rows, 4096), "bfloat16")
        with T.Kernel(rows, 16, threads=256) as (slot, head):
            for j in T.Parallel(d):
                maximum = T.alloc_var("float32")
                numerator = T.alloc_var("float32")
                denominator = T.alloc_var("float32")
                maximum = -T.infinity("float32")
                for part in T.serial(splits):
                    maximum = T.max(
                        maximum,
                        stats[
                            slot // chunk,
                            head // 8,
                            part,
                            slot % chunk * 8 + head % 8,
                            0,
                        ],
                    )
                numerator = 0
                denominator = 0
                if active[slot] != 0:
                    for part in T.serial(splits):
                        factor = T.exp(
                            stats[
                                slot // chunk,
                                head // 8,
                                part,
                                slot % chunk * 8 + head % 8,
                                0,
                            ]
                            - maximum
                        )
                        numerator += (
                            partial[
                                slot // chunk,
                                head // 8,
                                part,
                                slot % chunk * 8 + head % 8,
                                j,
                            ]
                            * factor
                        )
                        denominator += (
                            stats[
                                slot // chunk,
                                head // 8,
                                part,
                                slot % chunk * 8 + head % 8,
                                1,
                            ]
                            * factor
                        )
                    gate = T.cast(
                        T.cast(projection[slot, head * 512 + d + j], "bfloat16"),
                        "float32",
                    )
                    value = T.cast(
                        T.cast(numerator / denominator, "bfloat16"), "float32"
                    )
                    out[slot, head * d + j] = value / (1 + T.exp(-gate))
                else:
                    out[slot, head * d + j] = 0
        return out

    return compute


def mapped_fp8(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.kv import get

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    codec = get("fp8").bind(
        (
            p.get("pool", p.get("slots", p.get("r", 1))),
            p.get("kv_heads", 2),
            p["capacity"],
            p.get("d", 256),
        ),
        provider="cuda",
    )
    from akas.quantization._shared.access import packed_arguments

    decode_packed = packed_arguments(codec.require("read_packed"))
    slots, chunk, pool = (p[name] for name in ("slots", "chunk", "pool"))
    rows = slots * chunk
    heads, kh, d, cap, eps = (
        p[name] for name in ("heads", "kv_heads", "d", "capacity", "eps")
    )
    projection = 2 * heads * d + 2 * kh * d
    group = heads // kh
    qt = min(chunk, 4)
    qm = max(16, qt * group)
    blocks = T.ceildiv(chunk, qt)
    splits = p["splits"]
    shape = (slots, kh, blocks, splits, qt * group)

    @kernel
    def compute(
        partial: Tensor(shape + (d,), "float32"),
        stats: Tensor(shape + (2,), "float32"),
        proj: Tensor((rows, projection), "float32"),
        lengths: Tensor((slots,), "int32"),
    ) -> Tensor((rows, heads * d), "bfloat16"):
        out = output((rows, heads * d), "bfloat16")
        with T.Kernel(slots, heads, chunk, threads=256) as (batch, head, t):
            if t < lengths[batch]:
                maximum = T.alloc_var("float32")
                denominator = T.alloc_var("float32")
                maximum = -T.infinity("float32")
                denominator = 0
                for part in T.serial(splits):
                    maximum = T.max(
                        maximum,
                        stats[
                            batch,
                            head // group,
                            t // qt,
                            part,
                            t % qt * group + head % group,
                            0,
                        ],
                    )
                for part in T.serial(splits):
                    m = stats[
                        batch,
                        head // group,
                        t // qt,
                        part,
                        t % qt * group + head % group,
                        0,
                    ]
                    if m > -T.infinity("float32"):
                        denominator += stats[
                            batch,
                            head // group,
                            t // qt,
                            part,
                            t % qt * group + head % group,
                            1,
                        ] * T.exp(m - maximum)
                for j in T.Parallel(d):
                    value = T.alloc_var("float32")
                    value = 0
                    for part in T.serial(splits):
                        m = stats[
                            batch,
                            head // group,
                            t // qt,
                            part,
                            t % qt * group + head % group,
                            0,
                        ]
                        if m > -T.infinity("float32"):
                            value += partial[
                                batch,
                                head // group,
                                t // qt,
                                part,
                                t % qt * group + head % group,
                                j,
                            ] * T.exp(m - maximum)
                    normal = T.cast(T.cast(value / denominator, "bfloat16"), "float32")
                    gate = T.cast(
                        T.cast(
                            proj[batch * chunk + t, head * 2 * d + d + j], "bfloat16"
                        ),
                        "float32",
                    )
                    out[batch * chunk + t, head * d + j] = normal / (1 + T.exp(-gate))
        return out

    return compute


def floating(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T
    from akas.ops.attention.kernel import cache_arguments, cache_reader

    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    if d != 64:
        raise ValueError("floating attention requires 64-channel heads")
    chunk = p.get("chunk", 1)
    slots = r // chunk
    bq, bk = p.get("query_rows", 32), p.get("key_rows", 64)
    stages, threads = p.get("stages", 0), p.get("threads", 128)
    async_loads = (
        p.get("async_loads", False)
        and p["kv_dtype"] == "float16"
        and p.get("unit_scales", False)
    )
    if (
        bq not in (16, 32, 64)
        or bk not in (32, 64, 128)
        or stages not in (0, 1, 2, 3)
        or threads not in (128, 256)
    ):
        raise ValueError("unsupported floating prefill schedule")
    storage, read = cache_reader(p)
    controls = [("positions", r, "int32"), ("enabled", r, "int32")]
    if chunk == 1:
        raise ValueError("floating prefill requires a chunk")

    @kernel
    def compute(
        q: Tensor((r, h * d), "float32"),
        kc: Tensor(storage.values.shape, storage.values.dtype),
        vc: Tensor(storage.values.shape, storage.values.dtype),
        ks: Tensor(storage.metadata[0].shape, "float32"),
        vs: Tensor(storage.metadata[0].shape, "float32"),
        positions: Tensor(r, "int32"),
        enabled: Tensor(r, "int32"),
    ) -> Tensor(r * h * d, "float32"):
        out = output(r * h * d, "float32", position=5)
        with T.Kernel(T.ceildiv(chunk, bq), h, slots, threads=threads) as (
            bx,
            head,
            slot,
        ):
            query = T.alloc_shared((bq, d), "float16")
            key = T.alloc_shared((bk, d), "float16")
            value = T.alloc_shared((bk, d), "float16")
            prob = T.alloc_shared((bq, bk), "float16")
            scores = T.alloc_fragment((bq, bk), "float32")
            result = T.alloc_fragment((bq, d), "float32")
            maximum = T.alloc_fragment((bq,), "float32")
            previous = T.alloc_fragment((bq,), "float32")
            factor = T.alloc_fragment((bq,), "float32")
            normalizer = T.alloc_fragment((bq,), "float32")
            total = T.alloc_fragment((bq,), "float32")
            for i, j in T.Parallel(bq, d):
                row = slot * chunk + bx * bq + i
                query[i, j] = T.if_then_else(
                    bx * bq + i < chunk,
                    q[row, head * d + j]
                    * (ks[slot, head // (h // kh), 0, 0] if async_loads else 1),
                    0,
                )
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            limits = T.alloc_fragment((bq,), "int32")
            live = T.alloc_fragment((1,), "int32")
            for i in T.Parallel(bq):
                row = slot * chunk + T.min(bx * bq + i, chunk - 1)
                limits[i] = T.if_then_else(
                    bx * bq + i < chunk and enabled[row] != 0,
                    positions[row] + 1,
                    0,
                )
            T.reduce_max(limits, live, dim=0)
            tile_count = T.ceildiv(live[0], bk)
            pipeline_tiles = 4 if async_loads else 1
            for block in T.serial(T.ceildiv(tile_count, pipeline_tiles)):
                for local in T.Pipelined(pipeline_tiles, num_stages=stages):
                    tile = block * pipeline_tiles + local
                    if async_loads:
                        T.copy(kc[slot, head // (h // kh), tile * bk, 0], key)
                        T.copy(vc[slot, head // (h // kh), tile * bk, 0], value)
                    else:
                        for i, j in T.Parallel(bk, d):
                            key[i, j] = T.if_then_else(
                                tile * bk + i < cap,
                                read(kc, ks, slot, head // (h // kh), tile * bk + i, j),
                                0,
                            )
                            value[i, j] = T.if_then_else(
                                tile * bk + i < cap,
                                read(vc, vs, slot, head // (h // kh), tile * bk + i, j),
                                0,
                            )
                    T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(bq, bk):
                        row = slot * chunk + bx * bq + i
                        scores[i, j] = T.if_then_else(
                            bx * bq + i < chunk and tile * bk + j <= positions[row],
                            scores[i, j] * d ** (-0.5),
                            -T.infinity("float32"),
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(bq):
                        factor[i] = T.exp(previous[i] - maximum[i])
                    for i, j in T.Parallel(bq, bk):
                        scores[i, j] = T.exp(scores[i, j] - maximum[i])
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(bq):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]

                    for i, j in T.Parallel(bq, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result)
            for i, j in T.Parallel(bq, d):
                row = slot * chunk + bx * bq + i
                if bx * bq + i < chunk:
                    out[(row * h + head) * d + j] = T.if_then_else(
                        enabled[row] != 0,
                        result[i, j]
                        / normalizer[i]
                        * (vs[slot, head // (h // kh), 0, 0] if async_loads else 1),
                        0,
                    )
        return out

    return compute


def mapped_bf16(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    heads, kh, d, cap, eps = (
        p[n] for n in ("heads", "kv_heads", "d", "capacity", "eps")
    )
    group = heads // kh
    qt = min(chunk, 4)
    qm = max(16, qt * group)
    projection = 2 * heads * d + 2 * kh * d
    if p.get("kv_dtype") == "fp8":
        from akas.quantization.kv import get
        from akas.quantization._shared.access import cache_arguments

        codec = get("fp8").bind((pool, kh, cap, d), provider="cuda")
        decode_scalar = cache_arguments(codec.read)

        @kernel
        def compute(
            q: Tensor((rows, heads, d), "bfloat16"),
            kc: Tensor((pool, kh, cap, d), "uint8"),
            vc: Tensor((pool, kh, cap, d), "uint8"),
            ks: Tensor((pool, kh, cap, 2), "float32"),
            vs: Tensor((pool, kh, cap, 2), "float32"),
            proj: Tensor((rows, projection), "float32"),
            mapping: Tensor((slots,), "int32"),
            positions: Tensor((slots,), "int32"),
            lengths: Tensor((slots,), "int32"),
        ) -> Tensor((rows, heads * d), "bfloat16"):
            out = output((rows, heads * d), "bfloat16")
            with T.Kernel(slots, kh, T.ceildiv(chunk, qt), threads=128) as (
                batch,
                head,
                qblock,
            ):
                request = mapping[batch]
                if (request >= 0) & (qblock * qt < lengths[batch]):
                    query = T.alloc_shared((qm, d), "bfloat16")
                    key = T.alloc_shared((64, d), "bfloat16")
                    value = T.alloc_shared((64, d), "bfloat16")
                    prob = T.alloc_shared((qm, 64), "bfloat16")
                    scores = T.alloc_fragment((qm, 64), "float32")
                    result = T.alloc_fragment((qm, d), "float32")
                    maximum = T.alloc_fragment((qm,), "float32")
                    previous = T.alloc_fragment((qm,), "float32")
                    factor = T.alloc_fragment((qm,), "float32")
                    normalizer = T.alloc_fragment((qm,), "float32")
                    total = T.alloc_fragment((qm,), "float32")
                    T.clear(result)
                    T.clear(normalizer)
                    T.fill(maximum, -T.infinity("float32"))
                    for i, j in T.Parallel(qm, d):
                        query[i, j] = T.if_then_else(
                            (i < qt * group)
                            & (qblock * qt + i // group < lengths[batch]),
                            q[
                                batch * chunk
                                + T.min(qblock * qt + i // group, chunk - 1),
                                head * group + i % group,
                                j,
                            ],
                            0,
                        )
                    count = positions[batch] + T.min((qblock + 1) * qt, lengths[batch])
                    for tile in T.serial(T.ceildiv(count, 64)):
                        for i, j in T.Parallel(64, d):
                            pos = tile * 64 + i
                            key[i, j] = T.if_then_else(
                                pos < count,
                                decode_scalar(
                                    kc, ks, request, head, T.min(pos, cap - 1), j
                                ),
                                0,
                            )
                            value[i, j] = T.if_then_else(
                                pos < count,
                                decode_scalar(
                                    vc, vs, request, head, T.min(pos, cap - 1), j
                                ),
                                0,
                            )
                        T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                        T.copy(maximum, previous)
                        for i, j in T.Parallel(qm, 64):
                            scores[i, j] = T.if_then_else(
                                (tile * 64 + j < count)
                                & (
                                    tile * 64 + j
                                    <= positions[batch] + qblock * qt + i // group
                                ),
                                scores[i, j] * d ** (-0.5),
                                -T.infinity("float32"),
                            )
                        T.reduce_max(scores, maximum, dim=1, clear=False)
                        for i in T.Parallel(qm):
                            factor[i] = T.if_then_else(
                                previous[i] > -T.infinity("float32"),
                                T.exp(previous[i] - maximum[i]),
                                0,
                            )
                        for i, j in T.Parallel(qm, 64):
                            scores[i, j] = T.if_then_else(
                                scores[i, j] > -T.infinity("float32"),
                                T.exp(scores[i, j] - maximum[i]),
                                0,
                            )
                        T.reduce_sum(scores, total, dim=1)
                        for i in T.Parallel(qm):
                            normalizer[i] = normalizer[i] * factor[i] + total[i]
                        for i, j in T.Parallel(qm, d):
                            result[i, j] *= factor[i]
                        T.copy(scores, prob)
                        T.gemm(prob, value, result)
                    for i, j in T.Parallel(qt * group, d):
                        row = batch * chunk + qblock * qt + i // group
                        if qblock * qt + i // group < lengths[batch]:
                            gate = T.cast(
                                T.cast(
                                    proj[
                                        row,
                                        (head * group + i % group) * 2 * d + d + j,
                                    ],
                                    "bfloat16",
                                ),
                                "float32",
                            )
                            value_out = T.cast(
                                T.cast(result[i, j] / normalizer[i], "bfloat16"),
                                "float32",
                            )
                            out[row, (head * group + i % group) * d + j] = value_out / (
                                1 + T.exp(-gate)
                            )
            return out

        return compute

    @kernel
    def compute(
        q: Tensor((rows, heads, d), "bfloat16"),
        kc: Tensor((pool, kh, cap, d), "bfloat16"),
        vc: Tensor((pool, kh, cap, d), "bfloat16"),
        proj: Tensor((rows, projection), "float32"),
        mapping: Tensor((slots,), "int32"),
        positions: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
    ) -> Tensor((rows, heads * d), "bfloat16"):
        out = output((rows, heads * d), "bfloat16")
        with T.Kernel(slots, kh, T.ceildiv(chunk, qt), threads=128) as (
            batch,
            head,
            qblock,
        ):
            request = mapping[batch]
            if (request >= 0) & (qblock * qt < lengths[batch]):
                query = T.alloc_shared((qm, d), "bfloat16")
                key = T.alloc_shared((64, d), "bfloat16")
                value = T.alloc_shared((64, d), "bfloat16")
                prob = T.alloc_shared((qm, 64), "bfloat16")
                scores = T.alloc_fragment((qm, 64), "float32")
                result = T.alloc_fragment((qm, d), "float32")
                maximum = T.alloc_fragment((qm,), "float32")
                previous = T.alloc_fragment((qm,), "float32")
                factor = T.alloc_fragment((qm,), "float32")
                normalizer = T.alloc_fragment((qm,), "float32")
                total = T.alloc_fragment((qm,), "float32")
                T.clear(result)
                T.clear(normalizer)
                T.fill(maximum, -T.infinity("float32"))
                for i, j in T.Parallel(qm, d):
                    query[i, j] = T.if_then_else(
                        (i < qt * group) & (qblock * qt + i // group < lengths[batch]),
                        q[
                            batch * chunk + T.min(qblock * qt + i // group, chunk - 1),
                            head * group + i % group,
                            j,
                        ],
                        0,
                    )
                count = positions[batch] + T.min((qblock + 1) * qt, lengths[batch])
                for tile in T.serial(T.ceildiv(count, 64)):
                    for i, j in T.Parallel(64, d):
                        pos = tile * 64 + i
                        key[i, j] = T.if_then_else(
                            pos < count, kc[request, head, T.min(pos, cap - 1), j], 0
                        )
                        value[i, j] = T.if_then_else(
                            pos < count, vc[request, head, T.min(pos, cap - 1), j], 0
                        )
                    T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(qm, 64):
                        scores[i, j] = T.if_then_else(
                            (tile * 64 + j < count)
                            & (
                                tile * 64 + j
                                <= positions[batch] + qblock * qt + i // group
                            ),
                            scores[i, j] * d ** (-0.5),
                            -T.infinity("float32"),
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(qm):
                        factor[i] = T.if_then_else(
                            previous[i] > -T.infinity("float32"),
                            T.exp(previous[i] - maximum[i]),
                            0,
                        )
                    for i, j in T.Parallel(qm, 64):
                        scores[i, j] = T.if_then_else(
                            scores[i, j] > -T.infinity("float32"),
                            T.exp(scores[i, j] - maximum[i]),
                            0,
                        )
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(qm):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(qm, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result)
                for i, j in T.Parallel(qt * group, d):
                    row = batch * chunk + qblock * qt + i // group
                    if qblock * qt + i // group < lengths[batch]:
                        gate = T.cast(
                            T.cast(
                                proj[row, (head * group + i % group) * 2 * d + d + j],
                                "bfloat16",
                            ),
                            "float32",
                        )
                        value_out = T.cast(
                            T.cast(result[i, j] / normalizer[i], "bfloat16"), "float32"
                        )
                        out[row, (head * group + i % group) * d + j] = value_out / (
                            1 + T.exp(-gate)
                        )
        return out

    return compute


def native(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    d, cap = (256, p["capacity"])
    qrows, kvrows = (64, 32)

    @kernel
    def compute(
        q: Tensor((rows, 16, d), "bfloat16"),
        kc: Tensor((slots, 2, cap, d), "bfloat16"),
        vc: Tensor((slots, 2, cap, d), "bfloat16"),
        projection: Tensor((rows, 8192), "float32"),
        positions: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
    ) -> Tensor((rows, 4096), "bfloat16"):
        out = output((rows, 4096), "bfloat16")
        with T.Kernel(slots, 2, T.ceildiv(chunk, 8), threads=256) as (slot, head, tile):
            if tile * 8 < lengths[slot]:
                query = T.alloc_shared((qrows, d), "bfloat16")
                key = T.alloc_shared((kvrows, d), "bfloat16")
                value = T.alloc_shared((kvrows, d), "bfloat16")
                prob = T.alloc_shared((qrows, kvrows), "bfloat16")
                scores = T.alloc_fragment((qrows, kvrows), "float32")
                result = T.alloc_fragment((qrows, d), "float32")
                maximum = T.alloc_fragment((qrows,), "float32")
                previous = T.alloc_fragment((qrows,), "float32")
                factor = T.alloc_fragment((qrows,), "float32")
                normalizer = T.alloc_fragment((qrows,), "float32")
                total = T.alloc_fragment((qrows,), "float32")
                T.clear(result)
                T.clear(normalizer)
                T.fill(maximum, -T.infinity("float32"))
                for i, j in T.Parallel(qrows, d):
                    local = tile * 8 + i // 8
                    query[i, j] = T.if_then_else(
                        local < lengths[slot],
                        q[slot * chunk + T.min(local, chunk - 1), head * 8 + i % 8, j],
                        0,
                    )
                count = positions[slot] + T.min(lengths[slot], tile * 8 + 8)
                for step in T.serial(T.ceildiv(count, kvrows)):
                    for i, j in T.Parallel(kvrows, d):
                        index = step * kvrows + i
                        key[i, j] = T.if_then_else(
                            index < count, kc[slot, head, T.min(index, cap - 1), j], 0
                        )
                        value[i, j] = T.if_then_else(
                            index < count, vc[slot, head, T.min(index, cap - 1), j], 0
                        )
                    T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(qrows, kvrows):
                        valid = (
                            step * kvrows + j <= positions[slot] + tile * 8 + i // 8
                        ) & (step * kvrows + j < count)
                        scores[i, j] = T.if_then_else(
                            valid, scores[i, j] / 16, -T.infinity("float32")
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(qrows):
                        factor[i] = T.exp(previous[i] - maximum[i])
                    for i, j in T.Parallel(qrows, kvrows):
                        scores[i, j] = T.exp(scores[i, j] - maximum[i])
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(qrows):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(qrows, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result)
                for i, j in T.Parallel(qrows, d):
                    local = tile * 8 + i // 8
                    if local < lengths[slot]:
                        row = slot * chunk + local
                        gate = T.cast(
                            T.cast(
                                projection[row, (head * 8 + i % 8) * 512 + d + j],
                                "bfloat16",
                            ),
                            "float32",
                        )
                        normalized = T.cast(
                            T.cast(result[i, j] / normalizer[i], "bfloat16"), "float32"
                        )
                        out[row, (head * 8 + i % 8) * d + j] = normalized / (
                            1 + T.exp(-gate)
                        )
        return out

    return compute


def quantized(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.kv import get

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    format = get(p.get("kv_dtype", "fp8"))
    codec = format.bind(
        (
            p.get("pool", p.get("slots", p.get("r", 1))),
            p.get("kv_heads", 2),
            p["capacity"],
            p.get("d", 256),
        ),
        provider="cuda",
    )
    from akas.quantization._shared.access import cache_arguments, packed_arguments

    decode = cache_arguments(codec.read)
    packed_dtype = codec.packed_dtype
    decode_packed = (
        packed_arguments(codec.require("read_packed"))
        if p.get("packed_loads", False) and packed_dtype
        else None
    )
    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    d, cap = (256, p["capacity"])
    qrows, kvrows = (p.get("query_rows", 64), p.get("key_rows", 32))
    token_rows = qrows // 8
    packed = p.get("packed_loads", False)
    policy = T.GemmWarpPolicy.FullRow if qrows >= 128 else T.GemmWarpPolicy.Square
    if packed and cap % kvrows:
        raise ValueError("packed KV requires capacity divisible by key tile")
    storage = codec.storage
    if len(storage.metadata) != 1:
        raise ValueError("this attention ABI requires one metadata buffer per cache")
    cache_shape, cache_dtype = (storage.values.shape, storage.values.dtype)
    metadata_shape, metadata_dtype = (
        storage.metadata[0].shape,
        storage.metadata[0].dtype,
    )
    metadata_columns = metadata_shape[-1]
    if packed and (
        not packed_dtype or cache_shape[-1] != d or len(metadata_shape) != 4
    ):
        raise ValueError("KV codec does not support this packed attention load")

    @kernel
    def compute(
        q: Tensor((rows, 16, d), "bfloat16"),
        kc: Tensor(cache_shape, cache_dtype),
        vc: Tensor(cache_shape, cache_dtype),
        ks: Tensor(metadata_shape, metadata_dtype),
        vs: Tensor(metadata_shape, metadata_dtype),
        projection: Tensor((rows, 8192), "float32"),
        positions: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
    ) -> Tensor((rows, 4096), "bfloat16"):
        out = output((rows, 4096), "bfloat16")
        with T.Kernel(slots, 2, T.ceildiv(chunk, token_rows), threads=256) as (
            slot,
            head,
            tile,
        ):
            if tile * token_rows < lengths[slot]:
                query = T.alloc_shared((qrows, d), "bfloat16")
                key = T.alloc_shared((kvrows, d), "bfloat16")
                value = T.alloc_shared((kvrows, d), "bfloat16")
                prob = T.alloc_shared((qrows, kvrows), "bfloat16")
                if packed:
                    encoded = T.alloc_shared((kvrows, d), packed_dtype)
                    scales = T.alloc_shared((kvrows, metadata_columns), metadata_dtype)
                scores = T.alloc_fragment((qrows, kvrows), "float32")
                result = T.alloc_fragment((qrows, d), "float32")
                maximum = T.alloc_fragment((qrows,), "float32")
                previous = T.alloc_fragment((qrows,), "float32")
                factor = T.alloc_fragment((qrows,), "float32")
                normalizer = T.alloc_fragment((qrows,), "float32")
                total = T.alloc_fragment((qrows,), "float32")
                T.clear(result)
                T.clear(normalizer)
                T.fill(maximum, -T.infinity("float32"))
                for i, j in T.Parallel(qrows, d):
                    local = tile * token_rows + i // 8
                    query[i, j] = T.if_then_else(
                        local < lengths[slot],
                        q[slot * chunk + T.min(local, chunk - 1), head * 8 + i % 8, j],
                        0,
                    )
                count = positions[slot] + T.min(
                    lengths[slot], tile * token_rows + token_rows
                )
                for step in T.serial(T.ceildiv(count, kvrows)):
                    if packed:
                        start = step * kvrows
                        keys = T.view(kc, dtype=packed_dtype)
                        values = T.view(vc, dtype=packed_dtype)
                        T.copy(keys[slot, head, start : start + kvrows, 0:d], encoded)
                        T.copy(
                            ks[slot, head, start : start + kvrows, 0:metadata_columns],
                            scales,
                        )
                        for i, j in T.Parallel(kvrows, d):
                            key[i, j] = T.if_then_else(
                                start + i < count,
                                decode_packed(encoded, scales, i, j),
                                0,
                            )
                        T.copy(values[slot, head, start : start + kvrows, 0:d], encoded)
                        T.copy(
                            vs[slot, head, start : start + kvrows, 0:metadata_columns],
                            scales,
                        )
                        for i, j in T.Parallel(kvrows, d):
                            value[i, j] = T.if_then_else(
                                start + i < count,
                                decode_packed(encoded, scales, i, j),
                                0,
                            )
                    else:
                        for i, j in T.Parallel(kvrows, d):
                            index = step * kvrows + i
                            key[i, j] = T.if_then_else(
                                index < count,
                                decode(kc, ks, slot, head, T.min(index, cap - 1), j),
                                0,
                            )
                            value[i, j] = T.if_then_else(
                                index < count,
                                decode(vc, vs, slot, head, T.min(index, cap - 1), j),
                                0,
                            )
                    T.gemm(
                        query,
                        key,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=policy,
                    )
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(qrows, kvrows):
                        valid = (
                            step * kvrows + j
                            <= positions[slot] + tile * token_rows + i // 8
                        ) & (step * kvrows + j < count)
                        scores[i, j] = T.if_then_else(
                            valid, scores[i, j] / 16, -T.infinity("float32")
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(qrows):
                        factor[i] = T.exp(previous[i] - maximum[i])
                    for i, j in T.Parallel(qrows, kvrows):
                        scores[i, j] = T.exp(scores[i, j] - maximum[i])
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(qrows):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(qrows, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result, policy=policy)
                for i, j in T.Parallel(qrows, d):
                    local = tile * token_rows + i // 8
                    if local < lengths[slot]:
                        row = slot * chunk + local
                        gate = T.cast(
                            T.cast(
                                projection[row, (head * 8 + i % 8) * 512 + d + j],
                                "bfloat16",
                            ),
                            "float32",
                        )
                        normalized = T.cast(
                            T.cast(result[i, j] / normalizer[i], "bfloat16"), "float32"
                        )
                        out[row, (head * 8 + i % 8) * d + j] = normalized / (
                            1 + T.exp(-gate)
                        )
        return out

    return compute


def partial(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.kv import get

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    codec = get("fp8").bind(
        (
            p.get("pool", p.get("slots", p.get("r", 1))),
            p.get("kv_heads", 2),
            p["capacity"],
            p.get("d", 256),
        ),
        provider="cuda",
    )
    from akas.quantization._shared.access import cache_arguments, packed_arguments

    decode_scalar = cache_arguments(codec.read)
    from akas.quantization._shared.access import packed_arguments

    decode_packed = packed_arguments(codec.require("read_packed"))
    r, chunk, cap, splits, d = (
        p["slots"],
        p["chunk"],
        p["capacity"],
        p.get("splits", 16),
        256,
    )
    rows = r * chunk
    packed = p.get("packed_loads", False)
    tile_rows = p.get("key_rows", 64)
    query_rows = max(16, chunk * 8)
    if packed and cap % tile_rows:
        raise ValueError("packed KV requires capacity divisible by the key tile")

    @kernel
    def compute(
        q: Tensor((rows, 16, d), "bfloat16"),
        kc: Tensor((r, 2, cap, d), "uint8"),
        vc: Tensor((r, 2, cap, d), "uint8"),
        ks: Tensor((r, 2, cap, 2), "float32"),
        vs: Tensor((r, 2, cap, 2), "float32"),
        positions: Tensor((r,), "int32"),
        lengths: Tensor((r,), "int32"),
    ) -> tuple[
        Tensor((r, 2, splits, chunk * 8, d), "float32"),
        Tensor((r, 2, splits, chunk * 8, 2), "float32"),
    ]:
        out = output((r, 2, splits, chunk * 8, d), "float32")
        stats = output((r, 2, splits, chunk * 8, 2), "float32")
        with T.Kernel(r, 2, splits, threads=128) as (slot, head, part):
            query = T.alloc_shared((query_rows, d), "bfloat16")
            key = T.alloc_shared((tile_rows, d), "bfloat16")
            value = T.alloc_shared((tile_rows, d), "bfloat16")
            prob = T.alloc_shared((query_rows, tile_rows), "bfloat16")
            if packed:
                encoded = T.alloc_shared((tile_rows, d), codec.packed_dtype)
                scales = T.alloc_shared((tile_rows, 2), "float32")
            scores = T.alloc_fragment((query_rows, tile_rows), "float32")
            result = T.alloc_fragment((query_rows, d), "float32")
            maximum = T.alloc_fragment((query_rows,), "float32")
            previous = T.alloc_fragment((query_rows,), "float32")
            factor = T.alloc_fragment((query_rows,), "float32")
            normalizer = T.alloc_fragment((query_rows,), "float32")
            total = T.alloc_fragment((query_rows,), "float32")
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            for i, j in T.Parallel(query_rows, d):
                query[i, j] = T.if_then_else(
                    i < lengths[slot] * 8,
                    q[slot * chunk + T.min(i // 8, chunk - 1), head * 8 + i % 8, j],
                    0,
                )
            count = positions[slot] + lengths[slot]
            tiles = T.ceildiv(T.ceildiv(count, tile_rows), splits)
            if (lengths[slot] > 0) & (part * tiles * tile_rows < count):
                for tile in T.serial(tiles):
                    if packed:
                        start = (part * tiles + tile) * tile_rows
                        if start < cap:
                            keys = T.view(kc, dtype=codec.packed_dtype)
                            values = T.view(vc, dtype=codec.packed_dtype)
                            T.copy(
                                keys[slot, head, start : start + tile_rows, 0:d],
                                encoded,
                            )
                            T.copy(
                                ks[slot, head, start : start + tile_rows, 0:2], scales
                            )
                            for i, j in T.Parallel(tile_rows, d):
                                key[i, j] = T.if_then_else(
                                    start + i < count,
                                    decode_packed(encoded, scales, i, j),
                                    0,
                                )
                            T.copy(
                                values[slot, head, start : start + tile_rows, 0:d],
                                encoded,
                            )
                            T.copy(
                                vs[slot, head, start : start + tile_rows, 0:2], scales
                            )
                            for i, j in T.Parallel(tile_rows, d):
                                value[i, j] = T.if_then_else(
                                    start + i < count,
                                    decode_packed(encoded, scales, i, j),
                                    0,
                                )
                        else:
                            T.clear(key)
                            T.clear(value)
                    else:
                        for i, j in T.Parallel(tile_rows, d):
                            pos = (part * tiles + tile) * tile_rows + i
                            key[i, j] = T.if_then_else(
                                pos < count,
                                decode_scalar(
                                    kc, ks, slot, head, T.min(pos, cap - 1), j
                                ),
                                0,
                            )
                            value[i, j] = T.if_then_else(
                                pos < count,
                                decode_scalar(
                                    vc, vs, slot, head, T.min(pos, cap - 1), j
                                ),
                                0,
                            )
                    T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(query_rows, tile_rows):
                        scores[i, j] = T.if_then_else(
                            ((part * tiles + tile) * tile_rows + j < count)
                            & (
                                (part * tiles + tile) * tile_rows + j
                                <= positions[slot] + i // 8
                            ),
                            scores[i, j] * d ** (-0.5),
                            -T.infinity("float32"),
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(query_rows):
                        factor[i] = T.if_then_else(
                            previous[i] > -T.infinity("float32"),
                            T.exp(previous[i] - maximum[i]),
                            0,
                        )
                    for i, j in T.Parallel(query_rows, tile_rows):
                        scores[i, j] = T.if_then_else(
                            scores[i, j] > -T.infinity("float32"),
                            T.exp(scores[i, j] - maximum[i]),
                            0,
                        )
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(query_rows):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(query_rows, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result)
            for i, j in T.Parallel(chunk * 8, d):
                out[slot, head, part, i, j] = result[i, j]
            for i in T.Parallel(chunk * 8):
                stats[slot, head, part, i, 0] = maximum[i]
                stats[slot, head, part, i, 1] = normalizer[i]
        return (out, stats)

    return compute


def prefill_split_mapped_fp8(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.kv import get

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    codec = get("fp8").bind(
        (
            p.get("pool", p.get("slots", p.get("r", 1))),
            p.get("kv_heads", 2),
            p["capacity"],
            p.get("d", 256),
        ),
        provider="cuda",
    )
    from akas.quantization._shared.access import packed_arguments

    decode_packed = packed_arguments(codec.require("read_packed"))
    slots, chunk, pool = (p[name] for name in ("slots", "chunk", "pool"))
    rows = slots * chunk
    heads, kh, d, cap, eps = (
        p[name] for name in ("heads", "kv_heads", "d", "capacity", "eps")
    )
    projection = 2 * heads * d + 2 * kh * d
    group = heads // kh
    qt = min(chunk, 4)
    qm = max(16, qt * group)
    blocks = T.ceildiv(chunk, qt)
    splits = p["splits"]
    shape = (slots, kh, blocks, splits, qt * group)

    @kernel
    def compute(
        q: Tensor((rows, heads, d), "bfloat16"),
        kc: Tensor((pool, kh, cap, d), "uint8"),
        vc: Tensor((pool, kh, cap, d), "uint8"),
        ks: Tensor((pool, kh, cap, 2), "float32"),
        vs: Tensor((pool, kh, cap, 2), "float32"),
        mapping: Tensor((slots,), "int32"),
        positions: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
    ) -> tuple[Tensor(shape + (d,), "float32"), Tensor(shape + (2,), "float32")]:
        out = output(shape + (d,), "float32")
        stats = output(shape + (2,), "float32")
        with T.Kernel(slots, kh, blocks * splits, threads=128) as (batch, head, which):
            qblock = which // splits
            part = which % splits
            request = mapping[batch]
            query = T.alloc_shared((qm, d), "bfloat16")
            key = T.alloc_shared((64, d), "bfloat16")
            value = T.alloc_shared((64, d), "bfloat16")
            encoded = T.alloc_shared((64, d), codec.packed_dtype)
            scales = T.alloc_shared((64, 2), "float32")
            prob = T.alloc_shared((qm, 64), "bfloat16")
            scores = T.alloc_fragment((qm, 64), "float32")
            result = T.alloc_fragment((qm, d), "float32")
            maximum = T.alloc_fragment((qm,), "float32")
            previous = T.alloc_fragment((qm,), "float32")
            factor = T.alloc_fragment((qm,), "float32")
            normalizer = T.alloc_fragment((qm,), "float32")
            total = T.alloc_fragment((qm,), "float32")
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            count = positions[batch] + T.min((qblock + 1) * qt, lengths[batch])
            tiles = T.ceildiv(T.ceildiv(count, 64), splits)
            if (
                (request >= 0)
                & (qblock * qt < lengths[batch])
                & (part * tiles * 64 < count)
            ):
                for i, j in T.Parallel(qm, d):
                    query[i, j] = T.if_then_else(
                        (i < qt * group) & (qblock * qt + i // group < lengths[batch]),
                        q[
                            batch * chunk + T.min(qblock * qt + i // group, chunk - 1),
                            head * group + i % group,
                            j,
                        ],
                        0,
                    )
                for tile in T.serial(tiles):
                    start = (part * tiles + tile) * 64
                    if start < cap:
                        keys = T.view(kc, dtype=codec.packed_dtype)
                        values = T.view(vc, dtype=codec.packed_dtype)
                        T.copy(keys[request, head, start : start + 64, :], encoded)
                        T.copy(ks[request, head, start : start + 64, :], scales)
                        for i, j in T.Parallel(64, d):
                            key[i, j] = T.if_then_else(
                                start + i < count,
                                decode_packed(encoded, scales, i, j),
                                0,
                            )
                        T.copy(values[request, head, start : start + 64, :], encoded)
                        T.copy(vs[request, head, start : start + 64, :], scales)
                        for i, j in T.Parallel(64, d):
                            value[i, j] = T.if_then_else(
                                start + i < count,
                                decode_packed(encoded, scales, i, j),
                                0,
                            )
                    else:
                        T.clear(key)
                        T.clear(value)
                    T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(qm, 64):
                        scores[i, j] = T.if_then_else(
                            (start + j < count)
                            & (
                                start + j <= positions[batch] + qblock * qt + i // group
                            ),
                            scores[i, j] * d ** (-0.5),
                            -T.infinity("float32"),
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(qm):
                        factor[i] = T.if_then_else(
                            previous[i] > -T.infinity("float32"),
                            T.exp(previous[i] - maximum[i]),
                            0,
                        )
                    for i, j in T.Parallel(qm, 64):
                        scores[i, j] = T.if_then_else(
                            scores[i, j] > -T.infinity("float32"),
                            T.exp(scores[i, j] - maximum[i]),
                            0,
                        )
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(qm):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(qm, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result)
            for i, j in T.Parallel(qt * group, d):
                out[batch, head, qblock, part, i, j] = result[i, j]
            for i in T.Parallel(qt * group):
                stats[batch, head, qblock, part, i, 0] = maximum[i]
                stats[batch, head, qblock, part, i, 1] = normalizer[i]
        return (out, stats)

    return compute


def qkv_mapped_bf16(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    heads, kh, d, cap, eps, theta = (
        p[n] for n in ("heads", "kv_heads", "d", "capacity", "eps", "theta")
    )
    projection = 2 * heads * d + 2 * kh * d

    @kernel
    def compute(
        proj: Tensor((rows, projection), "float32"),
        qw: Tensor((d,), "bfloat16"),
        kw: Tensor((d,), "bfloat16"),
        mapping: Tensor((slots,), "int32"),
        positions: Tensor((rows,), "int32"),
        active: Tensor((rows,), "int32"),
        kc: Tensor((pool, kh, cap, d), "bfloat16"),
        vc: Tensor((pool, kh, cap, d), "bfloat16"),
    ) -> tuple[
        Tensor((pool, kh, cap, d), "bfloat16"),
        Tensor((pool, kh, cap, d), "bfloat16"),
        Tensor((rows, heads, d), "bfloat16"),
    ]:
        q = output((rows, heads, d), "bfloat16")
        with T.Kernel(rows, heads + kh, threads=256) as (row, head):
            if active[row] != 0:
                values = T.alloc_shared((d,), "bfloat16")
                squares = T.alloc_fragment((d,), "float32")
                total = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(d):
                    raw = T.cast(
                        T.cast(
                            proj[
                                row,
                                T.if_then_else(
                                    head < heads,
                                    head * 2 * d,
                                    2 * heads * d + (head - heads) * d,
                                )
                                + j,
                            ],
                            "bfloat16",
                        ),
                        "float32",
                    )
                    squares[j] = raw * raw
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(d):
                    raw = T.cast(
                        T.cast(
                            proj[
                                row,
                                T.if_then_else(
                                    head < heads,
                                    head * 2 * d,
                                    2 * heads * d + (head - heads) * d,
                                )
                                + j,
                            ],
                            "bfloat16",
                        ),
                        "float32",
                    )
                    weight = T.if_then_else(head < heads, qw[j], kw[j])
                    values[j] = (
                        raw
                        * T.rsqrt(total[0] / d + eps)
                        * (1 + T.cast(weight, "float32"))
                    )
                for j in T.Parallel(d):
                    value = T.alloc_var("float32")
                    value = T.cast(values[j], "float32")
                    if j < 64:
                        angle = positions[row] * T.exp(
                            T.cast(-2 * (j % 32), "float32")
                            / 64
                            * T.log(T.float32(theta))
                        )
                        pair = T.cast(values[(j + 32) % 64], "float32")
                        value = value * T.cos(angle) + T.if_then_else(
                            j < 32, -pair, pair
                        ) * T.sin(angle)
                    if head < heads:
                        q[row, head, j] = value
                    else:
                        kc[mapping[row // chunk], head - heads, positions[row], j] = (
                            value
                        )
                        vc[mapping[row // chunk], head - heads, positions[row], j] = (
                            T.cast(
                                proj[
                                    row, 2 * heads * d + kh * d + (head - heads) * d + j
                                ],
                                "bfloat16",
                            )
                        )
        return (kc, vc, q)

    return compute


def query(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, d, eps, cap = (p["r"], 256, p["eps"], p["capacity"])
    theta = p["theta"]
    qkind = True
    heads = 16 if qkind else 2

    @kernel
    def compute(
        projection: Tensor((r, 8192), "float32"),
        w: Tensor((d,), "bfloat16"),
        positions: Tensor((r,), "int32"),
        active: Tensor((r,), "int32"),
    ) -> Tensor((r, heads, d), "bfloat16"):
        q = output((r, heads, d), "bfloat16")
        with T.Kernel(r, heads, threads=256) as (slot, head):
            squares = T.alloc_fragment((d,), "float32")
            total = T.alloc_fragment((1,), "float32")
            normal = T.alloc_shared((d,), "bfloat16")
            for j in T.Parallel(d):
                value = T.cast(
                    T.cast(projection[slot, head * 512 + j], "bfloat16"), "float32"
                )
                squares[j] = value * value
            T.reduce_sum(squares, total, dim=0)
            for j in T.Parallel(d):
                value = T.cast(
                    T.cast(projection[slot, head * 512 + j], "bfloat16"), "float32"
                )
                normal[j] = (
                    value * T.rsqrt(total[0] / d + eps) * (1 + T.cast(w[j], "float32"))
                )
            for j in T.Parallel(d):
                value = T.alloc_var("float32")
                value = T.cast(normal[j], "float32")
                if j < 64:
                    angle = positions[slot] * T.exp(
                        T.cast(-2 * (j % 32), "float32") / 64 * T.log(T.float32(theta))
                    )
                    pair = T.cast(normal[(j + 32) % 64], "float32")
                    value = value * T.cos(angle) + T.if_then_else(
                        j < 32, -pair, pair
                    ) * T.sin(angle)
                q[slot, head, j] = T.if_then_else(active[slot] != 0, value, 0)
        return q

    return compute


def kv(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, d, eps, cap = (p["r"], 256, p["eps"], p["capacity"])
    theta = p["theta"]
    qkind = False
    heads = 16 if qkind else 2

    @kernel
    def compute(
        projection: Tensor((r, 512), "float32"),
        values: Tensor((r, 512), "float32"),
        w: Tensor((d,), "bfloat16"),
        positions: Tensor((r,), "int32"),
        active: Tensor((r,), "int32"),
        kc: Tensor((r, heads, cap, d), "bfloat16"),
        vc: Tensor((r, heads, cap, d), "bfloat16"),
    ) -> tuple[
        Tensor((r, heads, cap, d), "bfloat16"), Tensor((r, heads, cap, d), "bfloat16")
    ]:
        with T.Kernel(r, heads, threads=256) as (slot, head):
            if active[slot] != 0:
                squares = T.alloc_fragment((d,), "float32")
                total = T.alloc_fragment((1,), "float32")
                normal = T.alloc_shared((d,), "bfloat16")
                for j in T.Parallel(d):
                    value = T.cast(
                        T.cast(projection[slot, head * d + j], "bfloat16"), "float32"
                    )
                    squares[j] = value * value
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(d):
                    value = T.cast(
                        T.cast(projection[slot, head * d + j], "bfloat16"), "float32"
                    )
                    normal[j] = (
                        value
                        * T.rsqrt(total[0] / d + eps)
                        * (1 + T.cast(w[j], "float32"))
                    )
                for j in T.Parallel(d):
                    value = T.alloc_var("float32")
                    value = T.cast(normal[j], "float32")
                    if j < 64:
                        angle = positions[slot] * T.exp(
                            T.cast(-2 * (j % 32), "float32")
                            / 64
                            * T.log(T.float32(theta))
                        )
                        pair = T.cast(normal[(j + 32) % 64], "float32")
                        value = value * T.cos(angle) + T.if_then_else(
                            j < 32, -pair, pair
                        ) * T.sin(angle)
                    kc[slot, head, positions[slot], j] = value
                    vc[slot, head, positions[slot], j] = T.cast(
                        values[slot, head * d + j], "bfloat16"
                    )
        return (kc, vc)

    return compute


def kv_chunk(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    d, cap, eps, theta = (256, p["capacity"], p["eps"], p["theta"])

    @kernel
    def compute(
        projection: Tensor((rows, 512), "float32"),
        values: Tensor((rows, 512), "float32"),
        w: Tensor((d,), "bfloat16"),
        positions: Tensor((rows,), "int32"),
        active: Tensor((rows,), "int32"),
        kc: Tensor((slots, 2, cap, d), "bfloat16"),
        vc: Tensor((slots, 2, cap, d), "bfloat16"),
    ) -> tuple[
        Tensor((slots, 2, cap, d), "bfloat16"), Tensor((slots, 2, cap, d), "bfloat16")
    ]:
        with T.Kernel(rows, 2, threads=256) as (row, head):
            if active[row] != 0:
                squares = T.alloc_fragment((d,), "float32")
                total = T.alloc_fragment((1,), "float32")
                normal = T.alloc_shared((d,), "bfloat16")
                for j in T.Parallel(d):
                    value = T.cast(
                        T.cast(projection[row, head * d + j], "bfloat16"), "float32"
                    )
                    squares[j] = value * value
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(d):
                    value = T.cast(
                        T.cast(projection[row, head * d + j], "bfloat16"), "float32"
                    )
                    normal[j] = (
                        value
                        * T.rsqrt(total[0] / d + eps)
                        * (1 + T.cast(w[j], "float32"))
                    )
                for j in T.Parallel(d):
                    value = T.alloc_var("float32")
                    value = T.cast(normal[j], "float32")
                    if j < 64:
                        angle = positions[row] * T.exp(
                            T.cast(-2 * (j % 32), "float32")
                            / 64
                            * T.log(T.float32(theta))
                        )
                        pair = T.cast(normal[(j + 32) % 64], "float32")
                        value = value * T.cos(angle) + T.if_then_else(
                            j < 32, -pair, pair
                        ) * T.sin(angle)
                    kc[row // chunk, head, positions[row], j] = value
                    vc[row // chunk, head, positions[row], j] = T.cast(
                        values[row, head * d + j], "bfloat16"
                    )
        return (kc, vc)

    return compute


def kv_quantized(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.kv import get

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    format = get(p.get("kv_dtype", "fp8"))
    codec = format.bind(
        (
            p.get("pool", p.get("slots", p.get("r", 1))),
            p.get("kv_heads", 2),
            p["capacity"],
            p.get("d", 256),
        ),
        provider="cuda",
    )
    slots = p.get("slots", p.get("r"))
    chunk = p.get("chunk", 1)
    rows = slots * chunk
    d, cap, eps, theta = (256, p["capacity"], p["eps"], p["theta"])
    from akas.quantization._shared.access import pair_arguments

    encode_pair = pair_arguments(codec.require("write_pair"))
    storage = codec.storage
    if len(storage.metadata) != 1:
        raise ValueError("this attention ABI requires one metadata buffer per cache")
    cache_shape, cache_dtype = (storage.values.shape, storage.values.dtype)
    metadata_shape, metadata_dtype = (
        storage.metadata[0].shape,
        storage.metadata[0].dtype,
    )
    metadata_columns = metadata_shape[-1]

    @kernel
    def compute(
        projection: Tensor((rows, 512), "float32"),
        values: Tensor((rows, 512), "float32"),
        w: Tensor((d,), "bfloat16"),
        positions: Tensor((rows,), "int32"),
        active: Tensor((rows,), "int32"),
        kc: Tensor(cache_shape, cache_dtype),
        vc: Tensor(cache_shape, cache_dtype),
        ks: Tensor(metadata_shape, metadata_dtype),
        vs: Tensor(metadata_shape, metadata_dtype),
    ) -> tuple[
        Tensor(cache_shape, cache_dtype),
        Tensor(cache_shape, cache_dtype),
        Tensor(metadata_shape, metadata_dtype),
        Tensor(metadata_shape, metadata_dtype),
    ]:
        with T.Kernel(rows, 2, threads=256) as (row, head):
            if active[row] != 0:
                squares = T.alloc_fragment((d,), "float32")
                total = T.alloc_fragment((1,), "float32")
                normal = T.alloc_shared((d,), "bfloat16")
                rotated = T.alloc_shared((d,), "bfloat16")
                for j in T.Parallel(d):
                    value = T.cast(
                        T.cast(projection[row, head * d + j], "bfloat16"), "float32"
                    )
                    squares[j] = value * value
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(d):
                    value = T.cast(
                        T.cast(projection[row, head * d + j], "bfloat16"), "float32"
                    )
                    normal[j] = (
                        value
                        * T.rsqrt(total[0] / d + eps)
                        * (1 + T.cast(w[j], "float32"))
                    )
                for j in T.Parallel(d):
                    value = T.alloc_var("float32")
                    value = T.cast(normal[j], "float32")
                    if j < 64:
                        angle = positions[row] * T.exp(
                            T.cast(-2 * (j % 32), "float32")
                            / 64
                            * T.log(T.float32(theta))
                        )
                        pair = T.cast(normal[(j + 32) % 64], "float32")
                        value = value * T.cos(angle) + T.if_then_else(
                            j < 32, -pair, pair
                        ) * T.sin(angle)
                    rotated[j] = value
                encode_pair(
                    rotated,
                    values,
                    row,
                    head,
                    row // chunk,
                    positions[row],
                    kc,
                    vc,
                    ks,
                    vs,
                )
        return (kc, vc, ks, vs)

    return compute


def qkv_mapped_fp8(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.kv import get

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    codec = get("fp8").bind(
        (
            p.get("pool", p.get("slots", p.get("r", 1))),
            p.get("kv_heads", 2),
            p["capacity"],
            p.get("d", 256),
        ),
        provider="cuda",
    )
    from akas.quantization._shared.access import packed_arguments

    decode_packed = packed_arguments(codec.require("read_packed"))
    slots, chunk, pool = (p[name] for name in ("slots", "chunk", "pool"))
    rows = slots * chunk
    heads, kh, d, cap, eps = (
        p[name] for name in ("heads", "kv_heads", "d", "capacity", "eps")
    )
    projection = 2 * heads * d + 2 * kh * d
    theta = p["theta"]
    from akas.quantization._shared.access import pair_arguments

    codec = get("fp8").bind(
        (
            p.get("pool", p.get("slots", p.get("r", 1))),
            p.get("kv_heads", 2),
            p["capacity"],
            p.get("d", 256),
        ),
        provider="cuda",
        access={"value_offset": 2 * heads * d + kh * d},
    )
    encode_pair = pair_arguments(codec.require("write_pair"))

    @kernel
    def compute(
        proj: Tensor((rows, projection), "float32"),
        qw: Tensor((d,), "bfloat16"),
        kw: Tensor((d,), "bfloat16"),
        mapping: Tensor((slots,), "int32"),
        positions: Tensor((rows,), "int32"),
        active: Tensor((rows,), "int32"),
        kc: Tensor((pool, kh, cap, d), "uint8"),
        vc: Tensor((pool, kh, cap, d), "uint8"),
        ks: Tensor((pool, kh, cap, 2), "float32"),
        vs: Tensor((pool, kh, cap, 2), "float32"),
    ) -> tuple[
        Tensor((pool, kh, cap, d), "uint8"),
        Tensor((pool, kh, cap, d), "uint8"),
        Tensor((pool, kh, cap, 2), "float32"),
        Tensor((pool, kh, cap, 2), "float32"),
        Tensor((rows, heads, d), "bfloat16"),
    ]:
        q = output((rows, heads, d), "bfloat16")
        with T.Kernel(rows, heads + kh, threads=256) as (row, head):
            if active[row] != 0:
                normal = T.alloc_shared((d,), "bfloat16")
                rotated = T.alloc_shared((d,), "bfloat16")
                squares = T.alloc_fragment((d,), "float32")
                total = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(d):
                    offset = T.if_then_else(
                        head < heads, head * 2 * d, 2 * heads * d + (head - heads) * d
                    )
                    value = T.cast(T.cast(proj[row, offset + j], "bfloat16"), "float32")
                    squares[j] = value * value
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(d):
                    offset = T.if_then_else(
                        head < heads, head * 2 * d, 2 * heads * d + (head - heads) * d
                    )
                    value = T.cast(T.cast(proj[row, offset + j], "bfloat16"), "float32")
                    weight = T.if_then_else(head < heads, qw[j], kw[j])
                    normal[j] = (
                        value
                        * T.rsqrt(total[0] / d + eps)
                        * (1 + T.cast(weight, "float32"))
                    )
                for j in T.Parallel(d):
                    value = T.alloc_var("float32")
                    value = T.cast(normal[j], "float32")
                    if j < 64:
                        angle = positions[row] * T.exp(
                            T.cast(-2 * (j % 32), "float32")
                            / 64
                            * T.log(T.float32(theta))
                        )
                        pair = T.cast(normal[(j + 32) % 64], "float32")
                        value = value * T.cos(angle) + T.if_then_else(
                            j < 32, -pair, pair
                        ) * T.sin(angle)
                    rotated[j] = value
                if head < heads:
                    for j in T.Parallel(d):
                        q[row, head, j] = rotated[j]
                else:
                    encode_pair(
                        rotated,
                        proj,
                        row,
                        head - heads,
                        mapping[row // chunk],
                        positions[row],
                        kc,
                        vc,
                        ks,
                        vs,
                    )
        return (kc, vc, ks, vs, q)

    return compute


def decode(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.kv import get

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    codec = get("fp8").bind(
        (
            p.get("pool", p.get("slots", p.get("r", 1))),
            p.get("kv_heads", 2),
            p["capacity"],
            p.get("d", 256),
        ),
        provider="cuda",
    )
    from akas.quantization._shared.access import cache_arguments, packed_arguments

    decode_scalar = cache_arguments(codec.read)
    from akas.quantization._shared.access import packed_arguments

    decode_packed = packed_arguments(codec.require("read_packed"))
    slots, cap = (p["slots"], p["capacity"])
    d = 256
    block = 128
    if cap % block:
        raise ValueError("attention workspace requires capacity divisible by 128")

    @kernel
    def compute(
        kc: Tensor((slots, 2, cap, d), "uint8"),
        vc: Tensor((slots, 2, cap, d), "uint8"),
        ks: Tensor((slots, 2, cap, 2), "float32"),
        vs: Tensor((slots, 2, cap, 2), "float32"),
        positions: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
    ) -> tuple[
        Tensor((slots, 2, cap, d), "bfloat16"), Tensor((slots, 2, cap, d), "bfloat16")
    ]:
        decoded_k = output((slots, 2, cap, d), "bfloat16")
        decoded_v = output((slots, 2, cap, d), "bfloat16")
        with T.Kernel(slots, 2, T.ceildiv(cap, block), threads=256) as (
            slot,
            head,
            tile,
        ):
            count = positions[slot] + lengths[slot]
            if lengths[slot] > 0 and tile * block < count:
                ek = T.alloc_shared((block, d), codec.packed_dtype)
                ev = T.alloc_shared((block, d), codec.packed_dtype)
                sk = T.alloc_shared((block, 2), "float32")
                sv = T.alloc_shared((block, 2), "float32")
                keys = T.view(kc, dtype=codec.packed_dtype)
                values = T.view(vc, dtype=codec.packed_dtype)
                T.copy(keys[slot, head, tile * block : tile * block + block, 0:d], ek)
                T.copy(values[slot, head, tile * block : tile * block + block, 0:d], ev)
                T.copy(ks[slot, head, tile * block : tile * block + block, 0:2], sk)
                T.copy(vs[slot, head, tile * block : tile * block + block, 0:2], sv)
                T.sync_threads()
                for i, j in T.Parallel(block, d):
                    decoded_k[slot, head, tile * block + i, j] = 0
                    decoded_v[slot, head, tile * block + i, j] = 0
                    if tile * block + i < count:
                        decoded_k[slot, head, tile * block + i, j] = decode_packed(
                            ek, sk, i, j
                        )
                        decoded_v[slot, head, tile * block + i, j] = decode_packed(
                            ev, sv, i, j
                        )
        return (decoded_k, decoded_v)

    return compute


from .kernel import (
    attention as shared_attention,
    attention_flat,
    cache_extent,
    cache_pad,
    cache_write,
    chunk_active,
    merge_floating,
    partial_floating,
    prepare_flat,
    rope,
)
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "cuda"))


def floating_ordered(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T
    from akas.ops.attention.kernel import cache_arguments, cache_reader

    policy = T.GemmWarpPolicy.FullRow
    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    if d != 64:
        raise ValueError("floating attention requires 64-channel heads")
    chunk = p.get("chunk", 1)
    slots = r // chunk
    bq, bk = p.get("query_rows", 64), p.get("key_rows", 64)
    stages, threads = p.get("stages", 1), p.get("threads", 128)
    async_loads = (
        p.get("async_loads", True)
        and p["kv_dtype"] == "float16"
        and p.get("unit_scales", False)
    )
    if (
        bq not in (16, 32, 64, 128)
        or bk not in (32, 64, 128)
        or stages not in (0, 1, 2, 3)
        or threads not in (128, 256, 512)
    ):
        raise ValueError("unsupported floating prefill schedule")
    if bq != 64 or bk != 64 or threads != 128:
        raise ValueError("ordered warp layout requires Q64/K64/128 threads")
    if not async_loads or stages != 1 or cap % 256 or not p.get("staged_kv", False):
        raise ValueError(
            "ordered attention requires staged F16 KV, unit scales and 256-token capacity alignment"
        )
    storage, read = cache_reader(p)
    controls = [("positions", r, "int32"), ("enabled", r, "int32")]
    if chunk == 1:
        raise ValueError("floating prefill requires a chunk")

    @kernel
    def compute(
        q: Tensor((r, h * d), "float32"),
        kc: Tensor(storage.values.shape, storage.values.dtype),
        vc: Tensor(storage.values.shape, storage.values.dtype),
        ks: Tensor(storage.metadata[0].shape, "float32"),
        vs: Tensor(storage.metadata[0].shape, "float32"),
        positions: Tensor(r, "int32"),
        enabled: Tensor(r, "int32"),
    ) -> Tensor(r * h * d, "float32"):
        out = output(r * h * d, "float32", position=5)
        with T.Kernel(T.ceildiv(chunk, bq), h, slots, threads=threads) as (
            bx,
            head,
            slot,
        ):
            query = T.alloc_shared((bq, d), "float16")
            key = T.alloc_shared((bk, d), "float16")
            value = T.alloc_shared((bk, d), "float16")
            prob = T.alloc_shared((bq, bk), "float16")
            scores = T.alloc_fragment((bq, bk), "float32")
            result = T.alloc_fragment((bq, d), "float32")
            maximum = T.alloc_fragment((bq,), "float32")
            previous = T.alloc_fragment((bq,), "float32")
            factor = T.alloc_fragment((bq,), "float32")
            normalizer = T.alloc_fragment((bq,), "float32")
            total = T.alloc_fragment((bq,), "float32")
            lower = T.alloc_fragment((bq, 4), "float32")
            upper = T.alloc_fragment((bq, 4), "float32")
            paired = T.alloc_fragment((bq, 4), "float32")
            local_sum_layout = T.Fragment(
                (bq, 4),
                forward_thread_fn=lambda row, lane: row // 16 * 32 + row % 8 * 4 + lane,
                forward_index_fn=lambda row, lane: row % 16 // 8,
            )
            T.annotate_layout(
                {
                    lower: local_sum_layout,
                    upper: local_sum_layout,
                    paired: local_sum_layout,
                }
            )
            for i, j in T.Parallel(bq, d):
                row = slot * chunk + bx * bq + i
                query[i, j] = T.if_then_else(
                    bx * bq + i < chunk,
                    q[row, head * d + j]
                    * (ks[slot, head // (h // kh), 0, 0] if async_loads else 1),
                    0,
                )
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            limits = T.alloc_fragment((bq,), "int32")
            live = T.alloc_fragment((1,), "int32")
            for i in T.Parallel(bq):
                row = slot * chunk + T.min(bx * bq + i, chunk - 1)
                limits[i] = T.if_then_else(
                    bx * bq + i < chunk and enabled[row] != 0,
                    positions[row] + 1,
                    0,
                )
            T.reduce_max(limits, live, dim=0)
            tile_count = T.ceildiv(live[0], bk)
            pipeline_tiles = 4 if async_loads else 1
            for block in T.serial(T.ceildiv(tile_count, pipeline_tiles)):
                for local in T.Pipelined(pipeline_tiles, num_stages=stages):
                    tile = block * pipeline_tiles + local
                    if async_loads:
                        T.copy(kc[slot, head // (h // kh), tile * bk, 0], key)
                        T.copy(vc[slot, head // (h // kh), tile * bk, 0], value)
                    else:
                        for i, j in T.Parallel(bk, d):
                            key[i, j] = T.if_then_else(
                                tile * bk + i < cap,
                                read(kc, ks, slot, head // (h // kh), tile * bk + i, j),
                                0,
                            )
                            value[i, j] = T.if_then_else(
                                tile * bk + i < cap,
                                read(vc, vs, slot, head // (h // kh), tile * bk + i, j),
                                0,
                            )
                    T.gemm(
                        query,
                        key,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=policy,
                    )
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(bq, bk):
                        row = slot * chunk + bx * bq + i
                        scores[i, j] = T.if_then_else(
                            bx * bq + i < chunk and tile * bk + j <= positions[row],
                            scores[i, j] * d ** (-0.5),
                            -T.infinity("float32"),
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(bq):
                        factor[i] = T.exp(previous[i] - maximum[i])
                    for i, j in T.Parallel(bq, bk):
                        scores[i, j] = T.exp(scores[i, j] - maximum[i])
                    # Match the old two-column-warp order without a block barrier:
                    # sum each 32-key half in its original per-lane order, add the
                    # two lane partials, then reduce the same four column lanes.
                    for i, lane in T.Parallel(bq, 4):
                        lower[i, lane] = 0
                        for element in T.unroll(8):
                            column = element % 4 * 8 + lane * 2 + element // 4
                            lower[i, lane] += scores[i, column]
                    for i, lane in T.Parallel(bq, 4):
                        upper[i, lane] = 0
                        for element in T.unroll(8):
                            column = element % 4 * 8 + lane * 2 + element // 4 + 32
                            upper[i, lane] += scores[i, column]
                    for i, lane in T.Parallel(bq, 4):
                        paired[i, lane] = lower[i, lane] + upper[i, lane]
                    T.reduce_sum(paired, total, dim=1)
                    for i in T.Parallel(bq):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]

                    for i, j in T.Parallel(bq, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result, policy=policy)
            for i, j in T.Parallel(bq, d):
                row = slot * chunk + bx * bq + i
                if bx * bq + i < chunk:
                    out[(row * h + head) * d + j] = T.if_then_else(
                        enabled[row] != 0,
                        result[i, j]
                        / normalizer[i]
                        * (vs[slot, head // (h // kh), 0, 0] if async_loads else 1),
                        0,
                    )
        return out

    return compute


def window_attention(spec, schedule):
    """Tensor-core causal GQA with absolute positions and optional circular sliding KV."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, h, kh, d, cap, chunk, window, splits = (
        p[k] for k in ("r", "h", "kh", "d", "capacity", "chunk", "window", "splits")
    )
    qm, bk = p.get("query_rows", 16), p.get("key_rows", 64)
    shape = (r // chunk, kh, cap, d)
    query_dtype = p.get("query_dtype", "float32")

    @kernel
    def compute(
        q: Tensor((r, h * d), query_dtype),
        kc: Tensor(shape, "float16"),
        vc: Tensor(shape, "float16"),
        positions: Tensor((r,), "int32"),
        enabled: Tensor((r,), "int32"),
    ) -> Tensor((r * h * splits * (d + 2),), "float32"):
        out = output((r * h * splits * (d + 2),), "float32")
        with T.Kernel(T.ceildiv(chunk, qm), h, (r // chunk) * splits, threads=128) as (
            block,
            head,
            z,
        ):
            slot, part = z // splits, z % splits
            row0 = slot * chunk + block * qm
            query = T.alloc_shared((qm, d), "float16")
            key = T.alloc_shared((bk, d), "float16")
            value = T.alloc_shared((bk, d), "float16")
            prob = T.alloc_shared((qm, bk), "float16")
            scores = T.alloc_fragment((qm, bk), "float32")
            result = T.alloc_fragment((qm, d), "float32")
            maximum = T.alloc_fragment((qm,), "float32")
            previous = T.alloc_fragment((qm,), "float32")
            factor = T.alloc_fragment((qm,), "float32")
            normalizer = T.alloc_fragment((qm,), "float32")
            total = T.alloc_fragment((qm,), "float32")
            first = T.max(0, positions[row0] - window + 1) if window else 0
            last = T.alloc_var("int32")
            last = first
            for i in T.unroll(qm):
                if block * qm + i < chunk:
                    if enabled[row0 + i] != 0:
                        last = positions[row0 + i] + 1
            tiles = T.ceildiv(last - first, bk)
            per_part = T.ceildiv(tiles, splits)
            for i, j in T.Parallel(qm, d):
                row = row0 + i
                query[i, j] = T.if_then_else(
                    i + block * qm < chunk and enabled[T.min(row, r - 1)] != 0,
                    q[T.min(row, r - 1), head * d + j],
                    0,
                )
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -1e30)
            for tile in T.serial(part * per_part, T.min((part + 1) * per_part, tiles)):
                start = first + tile * bk
                if p.get("contiguous", False) and start + bk <= last:
                    T.copy(kc[slot, head // (h // kh), start : start + bk, :], key)
                    T.copy(vc[slot, head // (h // kh), start : start + bk, :], value)
                else:
                    for i, j in T.Parallel(bk, d):
                        key[i, j] = T.if_then_else(
                            start + i < last,
                            kc[slot, head // (h // kh), (start + i) % cap, j],
                            0,
                        )
                        value[i, j] = T.if_then_else(
                            start + i < last,
                            vc[slot, head // (h // kh), (start + i) % cap, j],
                            0,
                        )
                T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                T.copy(maximum, previous)
                for i, j in T.Parallel(qm, bk):
                    row = T.min(row0 + i, r - 1)
                    valid = (
                        block * qm + i < chunk
                        and enabled[row] != 0
                        and start + j <= positions[row]
                        and (window == 0 or start + j > positions[row] - window)
                    )
                    scores[i, j] = T.if_then_else(valid, scores[i, j] * d**-0.5, -1e30)
                T.reduce_max(scores, maximum, dim=1, clear=False)
                for i in T.Parallel(qm):
                    factor[i] = T.exp(previous[i] - maximum[i])
                for i, j in T.Parallel(qm, bk):
                    scores[i, j] = T.if_then_else(
                        scores[i, j] > -1e29, T.exp(scores[i, j] - maximum[i]), 0
                    )
                T.reduce_sum(scores, total, dim=1)
                for i in T.Parallel(qm):
                    normalizer[i] = normalizer[i] * factor[i] + total[i]
                for i, j in T.Parallel(qm, d):
                    result[i, j] *= factor[i]
                T.copy(scores, prob)
                T.gemm(prob, value, result)
            for i, j in T.Parallel(qm, d):
                if block * qm + i < chunk:
                    out[
                        ((row0 + i) * h + head) * splits * (d + 2) + part * (d + 2) + j
                    ] = result[i, j]
            for i in T.Parallel(qm):
                if block * qm + i < chunk:
                    out[
                        ((row0 + i) * h + head) * splits * (d + 2) + part * (d + 2) + d
                    ] = maximum[i]
                    out[
                        ((row0 + i) * h + head) * splits * (d + 2)
                        + part * (d + 2)
                        + d
                        + 1
                    ] = normalizer[i]
        return out

    return compute


def window_grouped(spec, schedule):
    """Share each K/V tile across all GQA heads during split decode."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, h, kh, d, cap, window, splits = (
        p[n] for n in ("r", "h", "kh", "d", "capacity", "window", "splits")
    )
    group = h // kh
    qm, bk = 16, p.get("key_rows", 64)
    if p["chunk"] != 1 or h % kh or not 1 <= group <= qm:
        raise ValueError(
            "grouped window attention requires decode and at most 16 query heads per KV head"
        )
    shape = (r, kh, cap, d)

    @kernel
    def compute(
        q: Tensor((r, h * d), "float32"),
        kc: Tensor(shape, "float16"),
        vc: Tensor(shape, "float16"),
        positions: Tensor((r,), "int32"),
        enabled: Tensor((r,), "int32"),
    ) -> Tensor((r * h * splits * (d + 2),), "float32"):
        out = output((r * h * splits * (d + 2),), "float32")
        with T.Kernel(kh, r * splits, threads=128) as (head, z):
            slot, part = z // splits, z % splits
            query = T.alloc_shared((qm, d), "float16")
            key = T.alloc_shared((bk, d), "float16")
            value = T.alloc_shared((bk, d), "float16")
            prob = T.alloc_shared((qm, bk), "float16")
            scores = T.alloc_fragment((qm, bk), "float32")
            result = T.alloc_fragment((qm, d), "float32")
            maximum = T.alloc_fragment((qm,), "float32")
            previous = T.alloc_fragment((qm,), "float32")
            factor = T.alloc_fragment((qm,), "float32")
            normalizer = T.alloc_fragment((qm,), "float32")
            total = T.alloc_fragment((qm,), "float32")
            first = T.max(0, positions[slot] - window + 1) if window else 0
            last = T.if_then_else(enabled[slot] != 0, positions[slot] + 1, first)
            tiles = T.ceildiv(last - first, bk)
            per_part = T.ceildiv(tiles, splits)
            for i, j in T.Parallel(qm, d):
                query[i, j] = T.if_then_else(
                    i < group and enabled[slot] != 0,
                    q[slot, (head * group + T.min(i, group - 1)) * d + j],
                    0,
                )
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -1e30)
            for tile in T.serial(part * per_part, T.min((part + 1) * per_part, tiles)):
                start = first + tile * bk
                for i, j in T.Parallel(bk, d):
                    key[i, j] = T.if_then_else(
                        start + i < last, kc[slot, head, (start + i) % cap, j], 0
                    )
                    value[i, j] = T.if_then_else(
                        start + i < last, vc[slot, head, (start + i) % cap, j], 0
                    )
                T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                T.copy(maximum, previous)
                for i, j in T.Parallel(qm, bk):
                    scores[i, j] = T.if_then_else(
                        i < group and start + j < last, scores[i, j] * d**-0.5, -1e30
                    )
                T.reduce_max(scores, maximum, dim=1, clear=False)
                for i in T.Parallel(qm):
                    factor[i] = T.exp(previous[i] - maximum[i])
                for i, j in T.Parallel(qm, bk):
                    scores[i, j] = T.if_then_else(
                        scores[i, j] > -1e29, T.exp(scores[i, j] - maximum[i]), 0
                    )
                T.reduce_sum(scores, total, dim=1)
                for i in T.Parallel(qm):
                    normalizer[i] = normalizer[i] * factor[i] + total[i]
                for i, j in T.Parallel(qm, d):
                    result[i, j] *= factor[i]
                T.copy(scores, prob)
                T.gemm(prob, value, result)
            for i, j in T.Parallel(group, d):
                out[((slot * h + head * group + i) * splits + part) * (d + 2) + j] = (
                    result[i, j]
                )
            for i in T.Parallel(group):
                out[((slot * h + head * group + i) * splits + part) * (d + 2) + d] = (
                    maximum[i]
                )
                out[
                    ((slot * h + head * group + i) * splits + part) * (d + 2) + d + 1
                ] = normalizer[i]
        return out

    return compute
