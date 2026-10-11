"""Operation kernels for sequence.cuda.packed_baseline."""


def make_kernel(kind, p):
    import tilelang.language as T

    c = p.get("c")
    r = p.get("r", 1)
    if kind == "last":

        @T.prim_func
        def kernel(
            x: T.Tensor((r * c,), "float32"),
            out: T.Tensor((c,), "float32"),
            pos: T.Tensor((2,), "int32"),
        ):
            with T.Kernel(T.ceildiv(c, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    col = block * 256 + lane
                    if col < c:
                        out[col] = x[(pos[1] - 1) * c + col]

        return kernel
    import tilelang.language as T

    if kind == "advance":

        @T.prim_func
        def kernel(pos: T.Tensor((2,), "int32")):
            with T.Kernel(1, threads=32):
                for i in T.Parallel(1):
                    pos[0] += pos[1]

        return kernel
    raise ValueError("unsupported operation: " + kind)
