"""Operation kernels for sequence.cuda.packed."""


def make_kernel(kind, p):
    import tilelang.language as T

    if kind == "prefill_tail":
        r, c, t = p["r"], p["c"], p["t"]

        @T.prim_func
        def kernel(
            x: T.Tensor((r * c,), "float32"),
            out: T.Tensor((t * c,), "float32"),
            pos: T.Tensor((2,), "int32"),
            tail_pos: T.Tensor((2,), "int32"),
        ):
            with T.Kernel(T.ceildiv(t * c, 256), threads=256) as bx:
                for lane in T.Parallel(256):
                    i = bx * 256 + lane
                    if i < t * c:
                        out[i] = T.if_then_else(
                            i // c < T.min(pos[1], t),
                            x[(T.max(pos[1] - t, 0) + i // c) * c + i % c],
                            0,
                        )
                    if (bx == 0) & (lane == 0):
                        tail_pos[0] = pos[0] + T.max(pos[1] - t, 0)
                        tail_pos[1] = T.min(pos[1], t)

        return kernel
    raise ValueError("unsupported operation: " + kind)
