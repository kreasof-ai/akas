"""Operation kernels for residual.cuda.packed_baseline."""


def make_kernel(kind, p):
    import tilelang.language as T

    c = p.get("c")
    r = p.get("r", 1)
    if kind == "add":
        n = r * c

        @T.prim_func
        def kernel(
            x: T.Tensor((n,), "float32"),
            y: T.Tensor((n,), "float32"),
            out: T.Tensor((n,), "float32"),
        ):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < n:
                        out[i] = x[i] + y[i]

        return kernel
    raise ValueError("unsupported operation: " + kind)
