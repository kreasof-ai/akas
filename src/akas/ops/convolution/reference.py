"""Gated causal short convolution with independent, masked request state."""


def make_kernel(kind, p):
    import tilelang.language as T

    r, c, length = p["r"], p["c"], p.get("length", 3)

    @T.prim_func
    def kernel(
        x: T.Tensor((r * 3 * c,), "float32"),
        w: T.Tensor((c * length,), "float32"),
        state: T.Tensor((r, length - 1, c), "float32"),
        out: T.Tensor((r * c,), "float32"),
        enabled: T.Tensor((r,), "int32"),
    ):
        with T.Kernel(T.ceildiv(c, 64), r, threads=64) as (block, row):
            tx = T.get_thread_binding()
            col = block * 64 + tx
            if col < c:
                if enabled[row] != 0:
                    current = T.alloc_var("float32")
                    acc = T.alloc_var("float32")
                    current = x[row * 3 * c + col] * x[row * 3 * c + 2 * c + col]
                    acc = current * w[col * length + length - 1]
                    for i in T.serial(length - 1):
                        acc += state[row, i, col] * w[col * length + i]
                    out[row * c + col] = x[row * 3 * c + c + col] * acc
                    for i in T.serial(length - 2):
                        state[row, i, col] = state[row, i + 1, col]
                    state[row, length - 2, col] = current
                else:
                    out[row * c + col] = 0

    return kernel
