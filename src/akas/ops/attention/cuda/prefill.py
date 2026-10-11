"""Operation kernels for attention.cuda.prefill."""


def make_kernel(kind, p):
    import tilelang.language as T

    slots, chunk = p["slots"], p["chunk"]
    rows = slots * chunk
    if kind == "attention_kv":
        d, cap, eps, theta = 256, p["capacity"], p["eps"], p["theta"]

        @T.prim_func
        def kernel(
            projection: T.Tensor((rows, 512), "float32"),
            values: T.Tensor((rows, 512), "float32"),
            w: T.Tensor((d,), "bfloat16"),
            positions: T.Tensor((rows,), "int32"),
            active: T.Tensor((rows,), "int32"),
            kc: T.Tensor((slots, 2, cap, d), "bfloat16"),
            vc: T.Tensor((slots, 2, cap, d), "bfloat16"),
        ):
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

        return kernel
    import tilelang.language as T

    slots, chunk = p["slots"], p["chunk"]
    rows = slots * chunk
    if kind == "attention":
        d, cap = 256, p["capacity"]
        qrows, kvrows = 64, 32

        @T.prim_func
        def kernel(
            q: T.Tensor((rows, 16, d), "bfloat16"),
            kc: T.Tensor((slots, 2, cap, d), "bfloat16"),
            vc: T.Tensor((slots, 2, cap, d), "bfloat16"),
            projection: T.Tensor((rows, 8192), "float32"),
            positions: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            out: T.Tensor((rows, 4096), "bfloat16"),
        ):
            with T.Kernel(slots, 2, T.ceildiv(chunk, 8), threads=256) as (
                slot,
                head,
                tile,
            ):
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
                            q[
                                slot * chunk + T.min(local, chunk - 1),
                                head * 8 + i % 8,
                                j,
                            ],
                            0,
                        )
                    # Causality bounds the read to the last query in this tile.
                    count = positions[slot] + T.min(lengths[slot], tile * 8 + 8)
                    for step in T.serial(T.ceildiv(count, kvrows)):
                        for i, j in T.Parallel(kvrows, d):
                            index = step * kvrows + i
                            key[i, j] = T.if_then_else(
                                index < count,
                                kc[slot, head, T.min(index, cap - 1), j],
                                0,
                            )
                            value[i, j] = T.if_then_else(
                                index < count,
                                vc[slot, head, T.min(index, cap - 1), j],
                                0,
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
                                T.cast(result[i, j] / normalizer[i], "bfloat16"),
                                "float32",
                            )
                            out[row, (head * 8 + i % 8) * d + j] = normalized / (
                                1 + T.exp(-gate)
                            )

        return kernel
    raise ValueError("unsupported operation: " + kind)
