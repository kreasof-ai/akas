"""Operation kernels for attention.cuda.bf16_batch."""


def make_kernel(kind, p):
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = p.get("slots", r), p.get("chunk", 1), p.get("pool", 256)
    rows = slots * chunk
    if kind == "attention_qkv":
        heads, kh, d, cap, eps, theta = (
            p[n] for n in ("heads", "kv_heads", "d", "capacity", "eps", "theta")
        )
        projection = 2 * heads * d + 2 * kh * d

        @T.prim_func
        def kernel(
            proj: T.Tensor((rows, projection), "float32"),
            qw: T.Tensor((d,), "bfloat16"),
            kw: T.Tensor((d,), "bfloat16"),
            mapping: T.Tensor((slots,), "int32"),
            positions: T.Tensor((rows,), "int32"),
            active: T.Tensor((rows,), "int32"),
            kc: T.Tensor((pool, kh, cap, d), "bfloat16"),
            vc: T.Tensor((pool, kh, cap, d), "bfloat16"),
            q: T.Tensor((rows, heads, d), "bfloat16"),
        ):
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
                            kc[
                                mapping[row // chunk], head - heads, positions[row], j
                            ] = value
                            vc[
                                mapping[row // chunk], head - heads, positions[row], j
                            ] = T.cast(
                                proj[
                                    row, 2 * heads * d + kh * d + (head - heads) * d + j
                                ],
                                "bfloat16",
                            )

        return kernel
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = p.get("slots", r), p.get("chunk", 1), p.get("pool", 256)
    rows = slots * chunk
    if kind == "attention":
        heads, kh, d, cap, eps = (
            p[n] for n in ("heads", "kv_heads", "d", "capacity", "eps")
        )
        group = heads // kh
        qt = min(chunk, 4)
        qm = max(16, qt * group)
        projection = 2 * heads * d + 2 * kh * d

        @T.prim_func
        def kernel(
            q: T.Tensor((rows, heads, d), "bfloat16"),
            kc: T.Tensor((pool, kh, cap, d), "bfloat16"),
            vc: T.Tensor((pool, kh, cap, d), "bfloat16"),
            proj: T.Tensor((rows, projection), "float32"),
            mapping: T.Tensor((slots,), "int32"),
            positions: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            out: T.Tensor((rows, heads * d), "bfloat16"),
        ):
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
                                kc[request, head, T.min(pos, cap - 1), j],
                                0,
                            )
                            value[i, j] = T.if_then_else(
                                pos < count,
                                vc[request, head, T.min(pos, cap - 1), j],
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
                                scores[i, j] * d**-0.5,
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
                                        row, (head * group + i % group) * 2 * d + d + j
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

        return kernel
    raise ValueError("unsupported operation: " + kind)
