"""Operation kernels for sequence.webgpu.packed."""


def make_kernel(kind, p):
    import tilelang.language as T

    r = p.get("r", 1)
    if kind == "prefill_tail":
        c, t = p["c"], p["t"]

        @T.prim_func
        def kernel(
            x: T.Tensor((r * c,), "float32"),
            out: T.Tensor((t * c,), "float32"),
            control: T.Tensor((2,), "int32"),
            tail_control: T.Tensor((2,), "int32"),
        ):
            with T.Kernel(T.ceildiv(t * c, 256), threads=256) as block:
                tx = T.get_thread_binding()
                index = block * 256 + tx
                start = T.max(control[1] - t, 0)
                count = T.min(control[1], t)
                if index < t * c:
                    out[index] = T.if_then_else(
                        index // c < count, x[start * c + index], 0
                    )
                if index == 0:
                    tail_control[0] = control[0] + start
                    tail_control[1] = count

        return kernel
    raise ValueError("unsupported operation: " + kind)
