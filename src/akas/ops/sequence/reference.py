"""Advance only live request positions after a full forward step."""


def make_kernel(kind, p):
    import tilelang.language as T

    r = p["r"]
    if kind == "next_tokens":

        @T.prim_func
        def copy(sampled: T.Tensor((r,), "int32"), tokens: T.Tensor((r,), "int32")):
            with T.Kernel(T.ceildiv(r, 64), threads=64) as block:
                tx = T.get_thread_binding()
                row = block * 64 + tx
                if row < r:
                    tokens[row] = sampled[row]

        return copy

    @T.prim_func
    def kernel(positions: T.Tensor((r,), "int32"), enabled: T.Tensor((r,), "int32")):
        with T.Kernel(T.ceildiv(r, 64), threads=64) as block:
            tx = T.get_thread_binding()
            row = block * 64 + tx
            if row < r:
                positions[row] += T.if_then_else(enabled[row] != 0, 1, 0)

    return kernel
