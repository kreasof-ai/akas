"""Operation kernels for linear.cuda.packed_baseline."""

from akas.quantization.weights.ggml import weight_storage
from akas.quantization.weights.ggml import weight_decoder

from akas.checkpoints.gguf import TYPES


def make_kernel(kind, p):
    import tilelang.language as T

    r = p.get("r", 1)
    if kind == "linear":
        k, o, q = (p["k"], p["o"], p["type"])
        count, dtype = weight_storage(q, k * o)
        read_weight = weight_decoder(q, k)
        _, block, size = TYPES[q]
        if r == 1:

            @T.prim_func
            def kernel(
                x: T.Tensor((r * k,), "float32"),
                w: T.Tensor((count,), dtype),
                out: T.Tensor((r * o,), "float32"),
            ):
                with T.Kernel(T.ceildiv(o, 4), threads=128) as bx:
                    accum = T.alloc_fragment((4, 32), "float32")
                    total = T.alloc_fragment((4,), "float32")
                    T.clear(accum)
                    for tile in T.serial(k // 32):
                        for row, lane in T.Parallel(4, 32):
                            accum[row, lane] += x[tile * 32 + lane] * read_weight(
                                w, bx * 4 + row, tile * 32 + lane
                            )
                    T.reduce_sum(accum, total, dim=1)
                    for row in T.Parallel(4):
                        if bx * 4 + row < o:
                            out[bx * 4 + row] = total[row]

            return kernel

        @T.prim_func
        def kernel(
            x: T.Tensor((r * k,), "float32"),
            w: T.Tensor((count,), dtype),
            out: T.Tensor((r * o,), "float32"),
        ):
            with T.Kernel(T.ceildiv(r, 32), T.ceildiv(o, 64), threads=128) as (by, bx):
                lhs = T.alloc_shared((32, 32), "float16")
                rhs = T.alloc_shared((64, 32), "float16")
                accum = T.alloc_fragment((32, 64), "float32")
                T.clear(accum)
                for tile in T.Pipelined(k // 32, num_stages=2):
                    for i, j in T.Parallel(32, 32):
                        lhs[i, j] = T.if_then_else(
                            by * 32 + i < r, x[(by * 32 + i) * k + tile * 32 + j], 0
                        )
                    for i, j in T.Parallel(64, 32):
                        if bx * 64 + i < o:
                            rhs[i, j] = read_weight(w, bx * 64 + i, tile * 32 + j)
                        else:
                            rhs[i, j] = 0
                    T.gemm(lhs, rhs, accum, transpose_B=True)
                for i, j in T.Parallel(32, 64):
                    if (by * 32 + i < r) & (bx * 64 + j < o):
                        out[(by * 32 + i) * o + bx * 64 + j] = accum[i, j]

        return kernel
    raise ValueError("unsupported operation: " + kind)
