from akas.kernels.kernel import Tensor,kernel,output
def factory(spec,schedule):
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
        or qm not in (64, 128, 256)
        or bk not in (64, 128)
        or stages not in (1, 2)
        or threads not in (128, 256, 512)
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

