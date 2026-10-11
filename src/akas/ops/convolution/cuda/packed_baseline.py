"""Operation kernels for convolution.cuda.packed_baseline."""


def make_kernel(kind, p):
    import tilelang.language as T

    c = p.get("c")
    r = p.get("r", 1)
    if kind == "conv":

        @T.prim_func
        def kernel(
            x: T.Tensor((r * 3 * c,), "float32"),
            w: T.Tensor((c * 3,), "float32"),
            state: T.Tensor((2 * c,), "float32"),
            out: T.Tensor((r * c,), "float32"),
            pos: T.Tensor((2,), "int32"),
        ):
            with T.Kernel(T.ceildiv(c, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    col = block * 256 + lane
                    if col < c:
                        left = T.alloc_var("float32")
                        right = T.alloc_var("float32")
                        current = T.alloc_var("float32")
                        left = state[col]
                        right = state[c + col]
                        for row in T.serial(r):
                            if row < pos[1]:
                                current = (
                                    x[row * (3 * c) + col]
                                    * x[row * (3 * c) + 2 * c + col]
                                )
                                out[row * c + col] = x[row * (3 * c) + c + col] * (
                                    left * w[col * 3]
                                    + right * w[col * 3 + 1]
                                    + current * w[col * 3 + 2]
                                )
                                left = right
                                right = current
                            else:
                                out[row * c + col] = 0
                        state[col] = left
                        state[c + col] = right

        return kernel
    raise ValueError("unsupported operation: " + kind)
