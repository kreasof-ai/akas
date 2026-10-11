"""Native SM89 FP8 tensor-core projections with block-128 dynamic scales."""


def bf16_head_kernel(p):
    """Vector-loaded BF16 tensor-core projection for the large vocabulary head."""
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    r, k, o = p["r"], p["k"], p["o"]
    m = 16
    n, depth, threads = p.get("columns", 64), p.get("depth", 128), p.get("threads", 128)
    parts = p.get("partitions", 1)
    if o % n or k % (depth * parts):
        raise ValueError("invalid BF16 projection partition")

    @T.macro
    def algorithm(x, w, out):
        with T.Kernel(o // n, T.ceildiv(r, m), parts, threads=threads) as (
            bx,
            by,
            part,
        ):
            lhs = T.alloc_shared((m, depth), "bfloat16")
            rhs = T.alloc_shared((n, depth), "bfloat16")
            accum = T.alloc_fragment((m, n), "float32")
            T.clear(accum)
            for tile in T.Pipelined(k // (depth * parts), num_stages=2):
                start = (part * (k // (depth * parts)) + tile) * depth
                for i, j in T.Parallel(m, depth):
                    lhs[i, j] = T.if_then_else(
                        by * m + i < r, x[T.min(by * m + i, r - 1), start + j], 0
                    )
                T.copy(w[bx * n : bx * n + n, start : start + depth], rhs)
                T.gemm(lhs, rhs, accum, transpose_B=True)
            for i, j in T.Parallel(m, n):
                if by * m + i < r:
                    if parts > 1:
                        out[by * m + i, part, bx * n + j] = accum[i, j]
                    else:
                        out[by * m + i, bx * n + j] = accum[i, j]

    shape = (r, parts, o) if parts > 1 else (r, o)
    return primitive(
        [
            ("x", (r, k), "bfloat16"),
            ("w", (o, k), "bfloat16"),
            ("out", shape, "float32"),
        ],
        algorithm,
    )
