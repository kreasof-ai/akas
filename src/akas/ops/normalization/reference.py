"""Portable per-row RMS normalization with F32 accumulation."""


def make_kernel(kind, p):
    import tilelang.language as T
    from akas.ops.reductions.webgpu import tree_reduce

    r, c, eps = p["r"], p["c"], p["eps"]
    reduce = tree_reduce(64)

    @T.prim_func
    def kernel(
        x: T.Tensor((r * c,), "float32"),
        w: T.Tensor((c,), "float32"),
        out: T.Tensor((r * c,), "float32"),
    ):
        with T.Kernel(r, threads=64) as row:
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((64,), "float32")
            acc = T.alloc_var("float32")
            acc = 0
            for tile in T.serial(T.ceildiv(c, 64)):
                if tile * 64 + tx < c:
                    acc += x[row * c + tile * 64 + tx] * x[row * c + tile * 64 + tx]
            scratch[tx] = acc
            T.sync_threads()
            reduce(scratch, tx)
            for tile in T.serial(T.ceildiv(c, 64)):
                if tile * 64 + tx < c:
                    out[row * c + tile * 64 + tx] = (
                        x[row * c + tile * 64 + tx]
                        * T.rsqrt(scratch[0] / c + eps)
                        * w[tile * 64 + tx]
                    )

    return kernel
