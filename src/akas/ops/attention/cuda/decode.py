"""Operation kernels for attention.cuda.decode."""


def make_kernel(kind, p):
    import tilelang.language as T

    if kind in ("attention_q", "attention_kv"):
        r, d, eps, cap = p["r"], 256, p["eps"], p["capacity"]
        theta = p["theta"]
        from tensor.compiler.frontend.source import primitive

        qkind = kind == "attention_q"
        heads = 16 if qkind else 2
        arguments = (
            [("projection", (r, 8192), "float32")]
            if qkind
            else [("projection", (r, 512), "float32"), ("values", (r, 512), "float32")]
        ) + [
            ("w", (d,), "bfloat16"),
            ("positions", (r,), "int32"),
            ("active", (r,), "int32"),
        ]
        arguments += (
            [("q", (r, heads, d), "bfloat16")]
            if qkind
            else [
                ("kc", (r, heads, cap, d), "bfloat16"),
                ("vc", (r, heads, cap, d), "bfloat16"),
            ]
        )

        @T.macro
        def query(projection, w, positions, active, q):
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
                    q[slot, head, j] = T.if_then_else(active[slot] != 0, value, 0)

        @T.macro
        def keyvalue(projection, values, w, positions, active, kc, vc):
            with T.Kernel(r, heads, threads=256) as (slot, head):
                if active[slot] != 0:
                    squares = T.alloc_fragment((d,), "float32")
                    total = T.alloc_fragment((1,), "float32")
                    normal = T.alloc_shared((d,), "bfloat16")
                    for j in T.Parallel(d):
                        value = T.cast(
                            T.cast(projection[slot, head * d + j], "bfloat16"),
                            "float32",
                        )
                        squares[j] = value * value
                    T.reduce_sum(squares, total, dim=0)
                    for j in T.Parallel(d):
                        value = T.cast(
                            T.cast(projection[slot, head * d + j], "bfloat16"),
                            "float32",
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

        return primitive(arguments, query if qkind else keyvalue)
    import tilelang.language as T

    if kind == "attention_partial":
        r, cap, splits, d = p["r"], p["capacity"], p.get("splits", 16), 256

        @T.prim_func
        def kernel(
            q: T.Tensor((r, 16, d), "bfloat16"),
            kc: T.Tensor((r, 2, cap, d), "bfloat16"),
            vc: T.Tensor((r, 2, cap, d), "bfloat16"),
            positions: T.Tensor((r,), "int32"),
            active: T.Tensor((r,), "int32"),
            out: T.Tensor((r, 2, splits, 8, d), "float32"),
            stats: T.Tensor((r, 2, splits, 8, 2), "float32"),
        ):
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
                                scores[i, j] * d**-0.5,
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

        return kernel
    import tilelang.language as T

    if kind == "attention_merge":
        r, splits, d = p["r"], p.get("splits", 16), 256

        @T.prim_func
        def kernel(
            partial: T.Tensor((r, 2, splits, 8, d), "float32"),
            stats: T.Tensor((r, 2, splits, 8, 2), "float32"),
            projection: T.Tensor((r, 8192), "float32"),
            active: T.Tensor((r,), "int32"),
            out: T.Tensor((r, 4096), "bfloat16"),
        ):
            with T.Kernel(r, 16, threads=256) as (slot, head):
                for j in T.Parallel(d):
                    maximum = T.alloc_var("float32")
                    numerator = T.alloc_var("float32")
                    denominator = T.alloc_var("float32")
                    maximum = -T.infinity("float32")
                    for part in T.serial(splits):
                        maximum = T.max(
                            maximum, stats[slot, head // 8, part, head % 8, 0]
                        )
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
                        # The attention output is BF16 before its gate, as in
                        # prefill and the checkpoint's BF16 inference path.
                        value = T.cast(
                            T.cast(numerator / denominator, "bfloat16"), "float32"
                        )
                        out[slot, head * d + j] = value / (1 + T.exp(-gate))
                    else:
                        out[slot, head * d + j] = 0

        return kernel
    raise ValueError("unsupported operation: " + kind)
