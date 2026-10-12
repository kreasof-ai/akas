"""Portable F32 projection baseline; one reduction per output and batch row."""


def make_kernel(kind, p):
    import tilelang.language as T
    from akas.ops.reductions.webgpu import tree_reduce

    r, k, o = p["r"], p["k"], p["o"]
    threads = p.get("threads", 64)
    if threads not in (32, 64, 128, 256):
        raise ValueError("reference projection threads must be 32/64/128/256")
    stride, offset = p.get("output_stride", o), p.get("output_offset", 0)
    out_shape = (r, stride) if p.get("matrix_output", False) else (r * stride,)
    reduce = tree_reduce(threads)

    @T.prim_func
    def kernel(
        x: T.Tensor((r * k,), "float32"),
        w: T.Tensor((o * k,), p.get("weight_dtype", "float32")),
        out: T.Tensor(out_shape, "float32"),
    ):
        with T.Kernel(T.ceildiv(o, 4), r, threads=threads) as (block, row):
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((threads,), "float32")
            acc = T.alloc_var("float32")
            for group in T.serial(4):
                col = block * 4 + group
                if col < o:
                    acc = 0
                    for tile in T.serial(T.ceildiv(k, threads)):
                        if tile * threads + tx < k:
                            acc += (
                                x[row * k + tile * threads + tx]
                                * w[col * k + tile * threads + tx]
                            )
                    scratch[tx] = acc
                    T.sync_threads()
                    reduce(scratch, tx)
                    if tx == 0:
                        if p.get("matrix_output", False):
                            out[row, offset + col] = scratch[0]
                        else:
                            out[row * stride + offset + col] = scratch[0]
                    T.sync_threads()

    return kernel
