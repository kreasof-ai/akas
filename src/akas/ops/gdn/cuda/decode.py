"""Operation kernels for gdn.cuda.decode."""


def make_kernel(kind, p):
    import tilelang.language as T

    if kind == "gdn_conv":
        r, channels = p["r"], 8192

        @T.prim_func
        def kernel(
            x: T.Tensor((r, channels), "float32"),
            w: T.Tensor((channels, 1, 4), "bfloat16"),
            active: T.Tensor((r,), "int32"),
            state: T.Tensor((r, channels, 3), "float32"),
            out: T.Tensor((r, channels), "float32"),
        ):
            with T.Kernel(r, channels // 256, threads=256) as (b, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    current = T.cast(T.cast(x[b, col], "bfloat16"), "float32")
                    value = T.alloc_var("float32")
                    value = current * T.cast(w[col, 0, 3], "float32")
                    for t in T.unroll(3):
                        value += state[b, col, t] * T.cast(w[col, 0, t], "float32")
                    if active[b] != 0:
                        state[b, col, 0] = state[b, col, 1]
                        state[b, col, 1] = state[b, col, 2]
                        state[b, col, 2] = current
                        out[b, col] = T.cast(value / (1 + T.exp(-value)), "bfloat16")
                    else:
                        out[b, col] = 0

        return kernel
    import tilelang.language as T

    if kind == "gdn_prepare":
        r, heads, d = p["r"], 32, 128

        @T.prim_func
        def kernel(
            qkv: T.Tensor((r, 8192), "float32"),
            a: T.Tensor((r, heads), "float32"),
            b: T.Tensor((r, heads), "float32"),
            dt: T.Tensor((heads,), "bfloat16"),
            A: T.Tensor((heads,), "float32"),
            q: T.Tensor((r, heads, d), "float32"),
            k: T.Tensor((r, heads, d), "float32"),
            v: T.Tensor((r, heads, d), "float32"),
            g: T.Tensor((r, heads), "float32"),
            beta: T.Tensor((r, heads), "float32"),
        ):
            with T.Kernel(r, heads, threads=128) as (slot, head):
                qs = T.alloc_fragment((d,), "float32")
                ks = T.alloc_fragment((d,), "float32")
                qq = T.alloc_fragment((d,), "float32")
                kk = T.alloc_fragment((d,), "float32")
                qtotal = T.alloc_fragment((1,), "float32")
                ktotal = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(d):
                    qq[j] = qkv[slot, head // 2 * d + j]
                    kk[j] = qkv[slot, 2048 + head // 2 * d + j]
                    qs[j] = qq[j] * qq[j]
                    ks[j] = kk[j] * kk[j]
                T.reduce_sum(qs, qtotal, dim=0)
                T.reduce_sum(ks, ktotal, dim=0)
                for j in T.Parallel(d):
                    q[slot, head, j] = (
                        qq[j] * T.rsqrt(qtotal[0] + T.float32(1e-6)) * d**-0.5
                    )
                    k[slot, head, j] = kk[j] * T.rsqrt(ktotal[0] + T.float32(1e-6))
                    v[slot, head, j] = qkv[slot, 4096 + head * d + j]
                for j in T.Parallel(1):
                    value = T.cast(
                        T.cast(a[slot, head], "bfloat16"), "float32"
                    ) + T.cast(dt[head], "float32")
                    softplus = T.max(value, 0) + T.log(1 + T.exp(-T.abs(value)))
                    g[slot, head] = -T.exp(A[head]) * softplus
                    beta[slot, head] = 1 / (
                        1 + T.exp(-T.cast(T.cast(b[slot, head], "bfloat16"), "float32"))
                    )

        return kernel
    import tilelang.language as T

    if kind == "gdn_norm":
        r, heads, d, eps = p["r"], 32, 128, p["eps"]

        @T.prim_func
        def kernel(
            x: T.Tensor((r, heads, d), "float32"),
            z: T.Tensor((r, 4096), "float32"),
            w: T.Tensor((d,), "float32"),
            out: T.Tensor((r, 4096), "bfloat16"),
        ):
            with T.Kernel(r, heads, threads=128) as (slot, head):
                values = T.alloc_fragment((d,), "float32")
                squares = T.alloc_fragment((d,), "float32")
                total = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(d):
                    values[j] = T.cast(T.cast(x[slot, head, j], "bfloat16"), "float32")
                    squares[j] = values[j] * values[j]
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(d):
                    gate = T.cast(T.cast(z[slot, head * d + j], "bfloat16"), "float32")
                    out[slot, head * d + j] = (
                        values[j]
                        * T.rsqrt(total[0] / d + eps)
                        * w[j]
                        * gate
                        / (1 + T.exp(-gate))
                    )

        return kernel
    import tilelang.language as T

    if kind == "gdn_recurrent":
        slots, heads, key, value = (p[n] for n in ("slots", "heads", "key", "value"))
        threads = p.get("threads", 128)
        tile = p.get("value_tile", 32)
        if (
            any(
                type(v) is not int or v <= 0
                for v in (slots, heads, key, value, tile, threads)
            )
            or key & (key - 1)
            or tile & (tile - 1)
            or value % tile
            or threads not in (64, 128, 256)
        ):
            raise ValueError("invalid gated delta schedule")

        # Each request's compact Q/K is already expanded to value heads and
        # normalized. log_decay is -exp(A_log)*softplus(a+dt_bias); beta=sigmoid(b).
        @T.prim_func
        def kernel(
            q: T.Tensor((slots, heads, key), "float32"),
            k: T.Tensor((slots, heads, key), "float32"),
            v: T.Tensor((slots, heads, value), "float32"),
            log_decay: T.Tensor((slots, heads), "float32"),
            beta: T.Tensor((slots, heads), "float32"),
            active: T.Tensor((slots,), "int32"),
            state: T.Tensor((slots, heads, value, key), "float32"),
            out: T.Tensor((slots, heads, value), "float32"),
        ):
            with T.Kernel(slots, heads, value // tile, threads=threads) as (
                slot,
                head,
                part,
            ):
                if active[slot] != 0:
                    matrix = T.alloc_fragment((tile, key), "float32")
                    prediction = T.alloc_fragment((tile, key), "float32")
                    inner = T.alloc_fragment((tile,), "float32")
                    result = T.alloc_fragment((tile,), "float32")
                    q_shared = T.alloc_shared((key,), "float32")
                    k_shared = T.alloc_shared((key,), "float32")
                    for j in T.Parallel(key):
                        q_shared[j] = q[slot, head, j]
                        k_shared[j] = k[slot, head, j]
                    for i, j in T.Parallel(tile, key):
                        matrix[i, j] = state[slot, head, part * tile + i, j] * T.exp(
                            log_decay[slot, head]
                        )
                        prediction[i, j] = matrix[i, j] * k_shared[j]
                    T.reduce_sum(prediction, inner, dim=1)
                    for i, j in T.Parallel(tile, key):
                        matrix[i, j] += (
                            k_shared[j]
                            * beta[slot, head]
                            * (v[slot, head, part * tile + i] - inner[i])
                        )
                        state[slot, head, part * tile + i, j] = matrix[i, j]
                        prediction[i, j] = matrix[i, j] * q_shared[j]
                    T.reduce_sum(prediction, result, dim=1)
                    for i in T.Parallel(tile):
                        out[slot, head, part * tile + i] = result[i]
                else:
                    for i in T.Parallel(tile):
                        out[slot, head, part * tile + i] = 0

        return kernel
    raise ValueError("unsupported operation: " + kind)
