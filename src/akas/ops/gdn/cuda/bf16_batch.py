"""Operation kernels for gdn.cuda.bf16_batch."""


def make_kernel(kind, p):
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = p.get("slots", r), p.get("chunk", 1), p.get("pool", 256)
    rows = slots * chunk
    if kind == "gdn_conv":
        c, projection = p["channels"], p["projection"]

        @T.prim_func
        def kernel(
            x: T.Tensor((rows, projection), "float32"),
            w: T.Tensor((c, 1, 4), "bfloat16"),
            mapping: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            state: T.Tensor((pool, c, 3), "float32"),
            out: T.Tensor((rows, c), "float32"),
        ):
            with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
                request = mapping[batch]
                if request >= 0:
                    for j in T.Parallel(256):
                        col = tile * 256 + j
                        if col < c:
                            h0 = T.alloc_var("float32")
                            h1 = T.alloc_var("float32")
                            h2 = T.alloc_var("float32")
                            h0 = state[request, col, 0]
                            h1 = state[request, col, 1]
                            h2 = state[request, col, 2]
                            for t in T.serial(chunk):
                                if t < lengths[batch]:
                                    current = T.cast(
                                        T.cast(x[batch * chunk + t, col], "bfloat16"),
                                        "float32",
                                    )
                                    value = h0 * T.cast(
                                        w[col, 0, 0], "float32"
                                    ) + h1 * T.cast(w[col, 0, 1], "float32")
                                    value += h2 * T.cast(
                                        w[col, 0, 2], "float32"
                                    ) + current * T.cast(w[col, 0, 3], "float32")
                                    out[batch * chunk + t, col] = T.cast(
                                        value / (1 + T.exp(-value)), "bfloat16"
                                    )
                                    h0 = h1
                                    h1 = h2
                                    h2 = current
                                else:
                                    out[batch * chunk + t, col] = 0
                            if p.get("commit", True):
                                state[request, col, 0] = h0
                                state[request, col, 1] = h1
                                state[request, col, 2] = h2

        return kernel
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = p.get("slots", r), p.get("chunk", 1), p.get("pool", 256)
    rows = slots * chunk
    if kind == "gdn_prepare":
        heads, kh, d, channels, projection = (
            p[n] for n in ("heads", "key_heads", "d", "channels", "projection")
        )
        value_width = heads * d

        @T.prim_func
        def kernel(
            conv: T.Tensor((rows, channels), "float32"),
            proj: T.Tensor((rows, projection), "float32"),
            dt: T.Tensor((heads,), "bfloat16"),
            A: T.Tensor((heads,), "float32"),
            q: T.Tensor((rows, heads, d), "float32"),
            k: T.Tensor((rows, heads, d), "float32"),
            v: T.Tensor((rows, heads, d), "float32"),
            g: T.Tensor((rows, heads), "float32"),
            beta: T.Tensor((rows, heads), "float32"),
        ):
            with T.Kernel(rows, heads, threads=128) as (row, head):
                qq = T.alloc_fragment((d,), "float32")
                kk = T.alloc_fragment((d,), "float32")
                qs = T.alloc_fragment((d,), "float32")
                ks = T.alloc_fragment((d,), "float32")
                qt = T.alloc_fragment((1,), "float32")
                kt = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(d):
                    qq[j] = conv[row, head // (heads // kh) * d + j]
                    kk[j] = conv[row, kh * d + head // (heads // kh) * d + j]
                    qs[j] = qq[j] * qq[j]
                    ks[j] = kk[j] * kk[j]
                T.reduce_sum(qs, qt, dim=0)
                T.reduce_sum(ks, kt, dim=0)
                for j in T.Parallel(d):
                    q[row, head, j] = qq[j] * T.rsqrt(qt[0] + T.float32(1e-6)) * d**-0.5
                    k[row, head, j] = kk[j] * T.rsqrt(kt[0] + T.float32(1e-6))
                    v[row, head, j] = conv[row, 2 * kh * d + head * d + j]
                for j in T.Parallel(1):
                    a = T.cast(
                        T.cast(proj[row, channels + value_width + head], "bfloat16"),
                        "float32",
                    ) + T.cast(dt[head], "float32")
                    g[row, head] = -T.exp(A[head]) * (
                        T.max(a, 0) + T.log(1 + T.exp(-T.abs(a)))
                    )
                    beta[row, head] = 1 / (
                        1
                        + T.exp(
                            -T.cast(
                                T.cast(
                                    proj[row, channels + value_width + heads + head],
                                    "bfloat16",
                                ),
                                "float32",
                            )
                        )
                    )

        return kernel
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = p.get("slots", r), p.get("chunk", 1), p.get("pool", 256)
    rows = slots * chunk
    if kind == "gdn_scan":
        heads, d, tile = p["heads"], p["d"], p.get("tile", 32)

        @T.prim_func
        def kernel(
            q: T.Tensor((rows, heads, d), "float32"),
            k: T.Tensor((rows, heads, d), "float32"),
            v: T.Tensor((rows, heads, d), "float32"),
            g: T.Tensor((rows, heads), "float32"),
            beta: T.Tensor((rows, heads), "float32"),
            mapping: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            state: T.Tensor((pool, heads, d, d), "float32"),
            out: T.Tensor((rows, heads, d), "float32"),
        ):
            with T.Kernel(slots, heads, d // tile, threads=128) as (batch, head, part):
                request = mapping[batch]
                if request >= 0:
                    matrix = T.alloc_fragment((tile, d), "float32")
                    prediction = T.alloc_fragment((tile, d), "float32")
                    inner = T.alloc_fragment((tile,), "float32")
                    result = T.alloc_fragment((tile,), "float32")
                    qq = T.alloc_shared((d,), "float32")
                    kk = T.alloc_shared((d,), "float32")
                    T.copy(
                        state[request, head, part * tile : part * tile + tile, :],
                        matrix,
                    )
                    for t in T.serial(chunk):
                        row = batch * chunk + t
                        if t < lengths[batch]:
                            for j in T.Parallel(d):
                                qq[j] = q[row, head, j]
                                kk[j] = k[row, head, j]
                            for i, j in T.Parallel(tile, d):
                                matrix[i, j] *= T.exp(g[row, head])
                                prediction[i, j] = matrix[i, j] * kk[j]
                            T.reduce_sum(prediction, inner, dim=1)
                            for i, j in T.Parallel(tile, d):
                                matrix[i, j] += (
                                    kk[j]
                                    * beta[row, head]
                                    * (v[row, head, part * tile + i] - inner[i])
                                )
                                prediction[i, j] = matrix[i, j] * qq[j]
                            T.reduce_sum(prediction, result, dim=1)
                            for i in T.Parallel(tile):
                                out[row, head, part * tile + i] = result[i]
                        else:
                            for i in T.Parallel(tile):
                                out[row, head, part * tile + i] = 0
                    if p.get("commit", True):
                        T.copy(
                            matrix,
                            state[request, head, part * tile : part * tile + tile, :],
                        )

        return kernel
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = p.get("slots", r), p.get("chunk", 1), p.get("pool", 256)
    rows = slots * chunk
    if kind == "gdn_norm":
        heads, d, channels, projection, eps = (
            p[n] for n in ("heads", "d", "channels", "projection", "eps")
        )

        @T.prim_func
        def kernel(
            x: T.Tensor((rows, heads, d), "float32"),
            proj: T.Tensor((rows, projection), "float32"),
            w: T.Tensor((d,), "float32"),
            out: T.Tensor((rows, heads * d), "bfloat16"),
        ):
            with T.Kernel(rows, heads, threads=128) as (row, head):
                values = T.alloc_fragment((d,), "float32")
                squares = T.alloc_fragment((d,), "float32")
                total = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(d):
                    values[j] = T.cast(T.cast(x[row, head, j], "bfloat16"), "float32")
                    squares[j] = values[j] * values[j]
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(d):
                    gate = T.cast(
                        T.cast(proj[row, channels + head * d + j], "bfloat16"),
                        "float32",
                    )
                    out[row, head * d + j] = (
                        values[j]
                        * T.rsqrt(total[0] / d + eps)
                        * w[j]
                        * gate
                        / (1 + T.exp(-gate))
                    )

        return kernel
    raise ValueError("unsupported operation: " + kind)
