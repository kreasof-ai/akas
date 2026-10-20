"""Expose independent router columns to more workgroups; retain F32 arithmetic."""
from akas.kernels.kernel import Tensor,kernel,output

def factory(spec, schedule):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    r, k, o = (p["r"], p["k"], p["o"])
    threads = p.get("threads", 64)
    columns_per_group = p.get("columns_per_group", 4)
    if columns_per_group not in (1, 2, 4):
        raise ValueError("unsupported projection output grouping")
    if threads not in (32, 64, 128, 256):
        raise ValueError("reference projection threads must be 32/64/128/256")
    stride, offset = (p.get("output_stride", o), p.get("output_offset", 0))
    out_shape = (r, stride) if p.get("matrix_output", False) else (r * stride,)
    reduce = tree_reduce(threads)

    @kernel
    def compute(
        x: Tensor((r * k,), "float32"),
        w: Tensor((o * k,), p.get("weight_dtype", "float32")),
    ) -> Tensor(out_shape, "float32"):
        out = output(out_shape, "float32")
        with T.Kernel(T.ceildiv(o, columns_per_group), r, threads=threads) as (block, row):
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((threads,), "float32")
            acc = T.alloc_var("float32")
            for group in T.serial(columns_per_group):
                col = block * columns_per_group + group
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
        return out

    return compute
