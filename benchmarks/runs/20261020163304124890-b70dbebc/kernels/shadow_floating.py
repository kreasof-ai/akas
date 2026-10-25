from akas.kernels.kernel import Tensor,kernel,output
def factory(spec, schedule):
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
                row = slot*chunk+T.min(bx*bq+i,chunk-1)
                limits[i] = T.if_then_else(bx*bq+i<chunk and enabled[row]!=0, positions[row]+1, 0)
            T.reduce_max(limits,live,dim=0)
            tile_count = T.ceildiv(live[0],bk)
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
