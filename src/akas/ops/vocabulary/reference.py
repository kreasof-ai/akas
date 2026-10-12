"""Portable floating embedding lookup and greedy sampling."""


def make_kernel(kind, p):
    import tilelang.language as T
    from akas.ops.reductions.webgpu import tree_reduce

    r, c = p["r"], p["c"]
    if kind == "embedding":
        v = p["v"]
        start = p.get("start", 0)

        @T.prim_func
        def kernel(
            tokens: T.Tensor((r,), "int32"),
            w: T.Tensor((v * c,), p.get("weight_dtype", "float32")),
            out: T.Tensor((r * c,), "float32"),
        ):
            with T.Kernel(T.ceildiv(c, 64), r, threads=64) as (block, row):
                tx = T.get_thread_binding()
                col = block * 64 + tx
                if col < c and tokens[row] >= start and tokens[row] < start + v:
                    out[row * c + col] = w[(tokens[row] - start) * c + col]

        return kernel
    if kind == "argmax":
        maximum = tree_reduce(64, "max")

        @T.prim_func
        def kernel(
            x: T.Tensor((r * c,), "float32"),
            out: T.Tensor((r,), "int32"),
            enabled: T.Tensor((r,), "int32"),
        ):
            with T.Kernel(r, threads=64) as row:
                tx = T.get_thread_binding()
                scratch = T.alloc_shared((64,), "float32")
                best = T.alloc_var("float32")
                index = T.alloc_var("int32")
                best = -T.infinity("float32")
                index = c
                for tile in T.serial(T.ceildiv(c, 64)):
                    if tile * 64 + tx < c:
                        if x[row * c + tile * 64 + tx] > best:
                            best = x[row * c + tile * 64 + tx]
                            index = tile * 64 + tx
                scratch[tx] = best
                T.sync_threads()
                maximum(scratch, tx)
                # Ties choose the lowest token ID deterministically.
                value = T.alloc_var("float32")
                value = scratch[0]
                T.sync_threads()
                scratch[tx] = T.cast(
                    T.if_then_else(best == value, -index, -c), "float32"
                )
                T.sync_threads()
                maximum(scratch, tx)
                if tx == 0 and enabled[row] != 0:
                    out[row] = T.cast(-scratch[0], "int32")

        return kernel
    raise ValueError(kind)
