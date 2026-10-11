"""Portable F32 projection without subgroup, FP16, or barrier requirements.

Each lane owns an output and accumulates channels serially. The reduction-based
reference remains a tuning candidate; this baseline also runs efficiently on
software adapters used for functional qualification.
"""


def make_kernel(kind, p):
    import tilelang.language as T

    r, k, o = p["r"], p["k"], p["o"]
    threads = p.get("threads", 64)
    if threads not in (32, 64, 128, 256):
        raise ValueError("reference projection threads must be 32/64/128/256")
    stride, offset = p.get("output_stride", o), p.get("output_offset", 0)
    matrix = p.get("matrix_output", False)
    shape = (r, stride) if matrix else (r * stride,)

    @T.prim_func
    def kernel(
        x: T.Tensor((r * k,), "float32"),
        w: T.Tensor((o * k,), "float32"),
        out: T.Tensor(shape, "float32"),
    ):
        with T.Kernel(T.ceildiv(o, threads), r, threads=threads) as (block, row):
            tx = T.get_thread_binding()
            col = block * threads + tx
            if col < o:
                accum = T.alloc_var("float32")
                accum = 0
                for channel in T.serial(k):
                    accum += x[row * k + channel] * w[col * k + channel]
                if matrix:
                    out[row, offset + col] = accum
                else:
                    out[row * stride + offset + col] = accum

    return kernel
