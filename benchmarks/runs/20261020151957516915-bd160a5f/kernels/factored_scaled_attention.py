from akas.kernels.kernel import Tensor,kernel,output
def factory(spec, schedule):
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
        key_scale,
        value_scale,
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
                scores[i, j] * 0.125 * key_scale[j, 0],
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
        for i, j in T.Parallel(qm, bk):
            prob[i,j] = scores[i,j] * value_scale[j,0] * 128
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
                            key[i, j] = T.cast(encoded_k[i,j], "float16")
                            value[i, j] = T.cast(encoded_v[i,j], "float16")
                    consume(
                        start,
                        length,
                        query,
                        key,
                        value,
                        scale_k,
                        scale_v,
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
                        T.cast(keys[slot, head, T.min(start + i, cap - 1), j], "float16"),
                        0,
                    )
                    value_tail[i, j] = T.if_then_else(
                        start + i < length,
                        T.cast(values[slot, head, T.min(start + i, cap - 1), j], "float16"),
                        0,
                    )
                for i in T.Parallel(bk):
                    scale_k[i,0] = T.if_then_else(start+i<length,ks[slot,head,T.min(start+i,cap-1),0],1)
                    scale_v[i,0] = T.if_then_else(start+i<length,vs[slot,head,T.min(start+i,cap-1),0],1)
                consume(
                    start,
                    length,
                    query,
                    key_tail,
                    value_tail,
                    scale_k,
                    scale_v,
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
                        result[i, j] * (vs[slot, head, 0, 0] if native_half else 0.0078125)
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

