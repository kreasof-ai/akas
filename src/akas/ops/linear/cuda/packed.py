"""Packed projections, attention, normalization, and reductions for CUDA.

Factories are producer-only; consumers execute Tensor artifacts."""

from akas.profiles.independent.factories import (
    packed_baseline_factory as baseline_kernel,
)
from akas.ops.quantization.ggml import weight_decoder
from akas.ops.quantization.ggml import weight_storage
from akas.checkpoints.gguf import TYPES
from akas.ops.quantization.cuda.ggml import packed_dot
from akas.ops.quantization.cuda.ggml import pair_decoder


def projection_abi(kind, p, algorithm):
    """Select the fused operator ABI without generating Python signatures."""
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    r, k, o, q = (p[n] for n in ("r", "k", "o", "type"))
    count, dtype = weight_storage(q, k * o)
    if kind == "ffn":

        @T.macro
        def call(x, w, w2, out):
            algorithm(x, w, out, w2)

        return primitive(
            [
                ("x", r * k, "float32"),
                ("w", count, dtype),
                ("w2", count, dtype),
                ("out", r * o, "float32"),
            ],
            call,
        )
    if kind == "linear_add":

        @T.macro
        def call(x, w, residual, out):
            algorithm(x, w, out, None, residual)

        return primitive(
            [
                ("x", r * k, "float32"),
                ("w", count, dtype),
                ("residual", r * o, "float32"),
                ("out", r * o, "float32"),
            ],
            call,
        )
    return primitive(
        [("x", r * k, "float32"), ("w", count, dtype), ("out", r * o, "float32")],
        algorithm,
    )


def gemv_kernel(kind, p):
    import tilelang.language as T

    k, o, q = p["k"], p["o"], p["type"]
    if q not in (0, 1, 2, 12, 14) or k % 256:
        return fused_kernel(kind, p)
    threads, unroll, width = (
        p.get("threads", 128),
        p.get("unroll", 4),
        p.get("f16_values", 4),
    )
    if (
        threads not in (64, 128, 256)
        or unroll not in (1, 2, 4, 8)
        or width not in (4, 8, 16)
        or k % (32 * width)
    ):
        raise ValueError("unsupported CUDA packed GEMV schedule")
    _, block, size = TYPES[q]
    row_bytes = k * (4 if q == 0 else 2) if q in (0, 1) else k // block * size
    tiles = k // (32 * width) if q in (0, 1) else k // 256
    paired = kind == "ffn"
    dot_product = packed_dot(q, k, row_bytes, width)

    @T.macro
    def gemv(x, w, out, w2=None, residual=None):
        with T.Kernel(T.ceildiv(o, threads // 32), threads=threads) as bx:
            tx = T.get_thread_binding()
            lane = tx % 32
            row = bx * (threads // 32) + tx // 32
            sums = T.alloc_local((unroll,), "float32")
            if paired:
                ups = T.alloc_local((unroll,), "float32")
            for slot in T.unroll(unroll):
                sums[slot] = 0
                if paired:
                    ups[slot] = 0
            if row < o:
                for chunk in T.serial(T.ceildiv(tiles, unroll)):
                    for slot in T.unroll(unroll):
                        tile = chunk * unroll + slot
                        if tile < tiles:
                            dot = dot_product(x, w, tile, lane, row)
                            sums[slot] = sums[slot] + dot
                            if paired:
                                dot2 = dot_product(x, w2, tile, lane, row)
                                ups[slot] = ups[slot] + dot2
            total = T.alloc_var("float32")
            total = 0
            if paired:
                up = T.alloc_var("float32")
                up = 0
            for slot in T.unroll(unroll):
                total = total + sums[slot]
                if paired:
                    up = up + ups[slot]
            for step in T.unroll(5):
                total = total + T.shfl_down(total, 16 >> step)
                if paired:
                    up = up + T.shfl_down(up, 16 >> step)
            if (lane == 0) & (row < o):
                if paired:
                    out[row] = total / (1 + T.exp(-total)) * up
                elif kind == "linear_add":
                    out[row] = residual[row] + total
                else:
                    out[row] = total

    return projection_abi(kind, p, gemv)


def prefill_kernel(kind, p):
    """FP16 tensor-core operands, FP32 accumulation, and paired FFN tiles."""
    import tilelang.language as T

    r, k, o, q = (p[n] for n in ("r", "k", "o", "type"))
    bm, bn, bk = p.get("block_m", 32), p.get("block_n", 64), p.get("block_k", 32)
    stages, threads = p.get("stages", 2), p.get("threads", 128)
    paired, packed = kind == "ffn", p.get("packed_pairs", False)
    read_weight = weight_decoder(q, k)
    read_pair = pair_decoder(q, k) if packed else None

    @T.macro
    def load_rhs(w, rhs, bx, tile, i, j):
        if packed:
            if bx * bn + i < o:
                bits = read_pair(w, bx * bn + i, tile * bk + j * 2)
                rhs[i, j * 2] = T.reinterpret(
                    "float16", T.cast(bits & T.uint32(65535), "uint16")
                )
                rhs[i, j * 2 + 1] = T.reinterpret(
                    "float16", T.cast(bits >> 16, "uint16")
                )
            else:
                rhs[i, j * 2] = 0
                rhs[i, j * 2 + 1] = 0
        else:
            if bx * bn + i < o:
                rhs[i, j] = read_weight(w, bx * bn + i, tile * bk + j)
            else:
                rhs[i, j] = 0

    @T.macro
    def matmul(x, w, out, w2=None, residual=None):
        with T.Kernel(T.ceildiv(r, bm), T.ceildiv(o, bn), threads=threads) as (by, bx):
            lhs = T.alloc_shared((bm, bk), "float16")
            rhs = T.alloc_shared((bn, bk), "float16")
            accum = T.alloc_fragment((bm, bn), "float32")
            T.clear(accum)
            if paired:
                rhs2 = T.alloc_shared((bn, bk), "float16")
                accum2 = T.alloc_fragment((bm, bn), "float32")
                T.clear(accum2)
            for tile in T.Pipelined(k // bk, num_stages=stages):
                for i, j in T.Parallel(bm, bk):
                    lhs[i, j] = T.if_then_else(
                        by * bm + i < r, x[(by * bm + i) * k + tile * bk + j], 0
                    )
                # Keep paired FFN operands in the same staging iteration. Splitting
                # the loops changes the pipelined loader's layout and scheduling.
                for i, j in T.Parallel(bn, bk // 2 if packed else bk):
                    load_rhs(w, rhs, bx, tile, i, j)
                    if paired:
                        load_rhs(w2, rhs2, bx, tile, i, j)
                T.gemm(lhs, rhs, accum, transpose_B=True)
                if paired:
                    T.gemm(lhs, rhs2, accum2, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                if (by * bm + i < r) & (bx * bn + j < o):
                    if paired:
                        out[(by * bm + i) * o + bx * bn + j] = (
                            accum[i, j] / (1 + T.exp(-accum[i, j])) * accum2[i, j]
                        )
                    elif kind == "linear_add":
                        out[(by * bm + i) * o + bx * bn + j] = (
                            residual[(by * bm + i) * o + bx * bn + j] + accum[i, j]
                        )
                    else:
                        out[(by * bm + i) * o + bx * bn + j] = accum[i, j]

    return projection_abi(kind, p, matmul)


def fused_kernel(kind, p):
    import tilelang.language as T

    if kind == "linear":
        return baseline_kernel("linear", p)
    if p["r"] > 1:
        return prefill_kernel(
            kind,
            {
                **p,
                "block_m": 32,
                "block_n": 64,
                "block_k": 32,
                "stages": 2,
                "threads": 128,
                "packed_pairs": False,
            },
        )
    k, o, q = p["k"], p["o"], p["type"]
    paired = kind == "ffn"
    read_weight = weight_decoder(q, k)

    @T.macro
    def gemv(x, w, out, w2=None, residual=None):
        with T.Kernel(T.ceildiv(o, 4), threads=128) as bx:
            accum = T.alloc_fragment((4, 32), "float32")
            if paired:
                accum2 = T.alloc_fragment((4, 32), "float32")
            total = T.alloc_fragment((4,), "float32")
            if paired:
                total2 = T.alloc_fragment((4,), "float32")
            T.clear(accum)
            if paired:
                T.clear(accum2)
            for tile in T.serial(k // 32):
                for row, lane in T.Parallel(4, 32):
                    accum[row, lane] += x[tile * 32 + lane] * read_weight(
                        w, bx * 4 + row, tile * 32 + lane
                    )
                    if paired:
                        accum2[row, lane] += x[tile * 32 + lane] * read_weight(
                            w2, bx * 4 + row, tile * 32 + lane
                        )
            T.reduce_sum(accum, total, dim=1)
            if paired:
                T.reduce_sum(accum2, total2, dim=1)
            for row in T.Parallel(4):
                if bx * 4 + row < o:
                    if paired:
                        out[bx * 4 + row] = (
                            total[row] / (1 + T.exp(-total[row])) * total2[row]
                        )
                    else:
                        out[bx * 4 + row] = residual[bx * 4 + row] + total[row]

    return projection_abi(kind, p, gemv)


def make_kernel(kind, p):
    if kind in ("linear", "linear_add", "ffn"):
        if p["r"] == 1:
            return gemv_kernel(kind, p)
        return prefill_kernel(kind, p) if "block_m" in p else fused_kernel(kind, p)
    raise ValueError("unsupported operation: " + kind)
