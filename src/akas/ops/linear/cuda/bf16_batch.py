"""Operation kernels for linear.cuda.bf16_batch."""


def make_kernel(kind, p):
    import tilelang.language as T

    r = p.get("r", 1)
    if kind == "linear":
        k, o = p["k"], p["o"]
        m, n, bk = p.get("m", 16), p.get("n", 64), p.get("bk", 64)

        @T.prim_func
        def kernel(
            x: T.Tensor((r, k), "bfloat16"),
            w: T.Tensor((o, k), "bfloat16"),
            out: T.Tensor((r, o), "float32"),
        ):
            with T.Kernel(T.ceildiv(o, n), T.ceildiv(r, m), threads=128) as (bx, by):
                a = T.alloc_shared((m, bk), "bfloat16")
                b = T.alloc_shared((n, bk), "bfloat16")
                acc = T.alloc_fragment((m, n), "float32")
                T.clear(acc)
                for tile in T.Pipelined(
                    T.ceildiv(k, bk), num_stages=p.get("stages", 2)
                ):
                    for i, j in T.Parallel(m, bk):
                        a[i, j] = T.if_then_else(
                            (by * m + i < r) & (tile * bk + j < k),
                            x[T.min(by * m + i, r - 1), T.min(tile * bk + j, k - 1)],
                            0,
                        )
                    for i, j in T.Parallel(n, bk):
                        b[i, j] = T.if_then_else(
                            (bx * n + i < o) & (tile * bk + j < k),
                            w[T.min(bx * n + i, o - 1), T.min(tile * bk + j, k - 1)],
                            0,
                        )
                    T.gemm(a, b, acc, transpose_B=True)
                for i, j in T.Parallel(m, n):
                    if (by * m + i < r) & (bx * n + j < o):
                        out[by * m + i, bx * n + j] = acc[i, j]

        return kernel
    import tilelang.language as T

    r = p.get("r", 1)
    if kind == "gemv":
        k, o, cols = p["k"], p["o"], p.get("columns", 16)

        @T.prim_func
        def kernel(
            x: T.Tensor((r, k), "bfloat16"),
            w: T.Tensor((o, k), "bfloat16"),
            out: T.Tensor((r, o), "float32"),
        ):
            with T.Kernel(T.ceildiv(o, cols), r, threads=256) as (block, row):
                sums = T.alloc_fragment((cols, 32), "float32")
                total = T.alloc_fragment((cols,), "float32")
                T.clear(sums)
                for tile in T.serial(T.ceildiv(k, 32)):
                    for i, j in T.Parallel(cols, 32):
                        if (block * cols + i < o) & (tile * 32 + j < k):
                            sums[i, j] = T.ieee_fmaf(
                                T.cast(x[row, tile * 32 + j], "float32"),
                                T.cast(w[block * cols + i, tile * 32 + j], "float32"),
                                sums[i, j],
                            )
                T.reduce_sum(sums, total, dim=1)
                for i in T.Parallel(cols):
                    if block * cols + i < o:
                        out[row, block * cols + i] = total[i]

        return kernel
    raise ValueError("unsupported operation: " + kind)
