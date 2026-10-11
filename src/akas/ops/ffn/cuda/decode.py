"""Operation kernels for ffn.cuda.decode."""

from akas.ops.ffn.activation import swiglu_kernel


def make_kernel(kind, p):
    if kind == "swiglu":
        return swiglu_kernel(dict(p, output_dtype="bfloat16", round_inputs=True))
    import tilelang.language as T

    if kind == "swiglu_experts":
        r, top, c = p["r"], p["top"], p["c"]

        @T.prim_func
        def kernel(
            g: T.Tensor((r, top, c), "float32"),
            u: T.Tensor((r, top, c), "float32"),
            out: T.Tensor((r, top, c), "bfloat16"),
        ):
            with T.Kernel(r, top, T.ceildiv(c, 256), threads=256) as (b, rank, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        gate = T.cast(T.cast(g[b, rank, col], "bfloat16"), "float32")
                        up = T.cast(T.cast(u[b, rank, col], "bfloat16"), "float32")
                        out[b, rank, col] = gate / (1 + T.exp(-gate)) * up

        return kernel
    raise ValueError("unsupported operation: " + kind)
