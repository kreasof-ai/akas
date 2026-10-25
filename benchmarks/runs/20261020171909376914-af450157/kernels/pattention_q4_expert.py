"""Experimental full expert fusion; keep the FFN intermediate in shared SRAM.

Inspired by kreasof-ai/foveal-sparse-indexer pattention_triton.py. Original
expert routing, Q4 payloads, F16 operands, and F32 accumulators are preserved.
Output tiling deliberately exposes the recomputation cost for measurement.
"""
from akas.kernels.kernel import Tensor, kernel, output


def factory(p):
    import tilelang.language as T
    from akas.quantization.weights import bind_parts

    rows, top, experts, depth, hidden, columns = (
        p[n] for n in ("r", "top", "experts", "k", "hidden", "o")
    )
    bm, bn, bk, bd = (p[n] for n in ("block_m", "block_n", "block_k", "block_d"))
    capacity = (rows * top + 3) // 4 + experts
    groups = (rows * top + bm - 1) // bm + experts
    layout = bm.bit_length() - 3
    projection = bind_parts(p["weight_parts"], depth, provider="cuda")
    down = bind_parts(p["down_parts"], hidden, provider="cuda")
    gate_storage, down_storage = projection.storage.values, down.storage.values
    load_gate, load_down = projection.require("read_tile"), down.require("read_tile")

    @kernel
    def compute(
        x: Tensor((rows, depth), "float32"),
        w: Tensor(gate_storage.shape, gate_storage.dtype),
        w2: Tensor(gate_storage.shape, gate_storage.dtype),
        wd: Tensor(down_storage.shape, down_storage.dtype),
        counts: Tensor((experts,), "int32"),
        routes: Tensor((experts, rows), "int32"),
        tile_experts: Tensor((5, capacity), "int32"),
        tile_rows: Tensor((5, capacity), "int32"),
    ) -> Tensor((rows, top, columns), "float32"):
        out = output((rows, top, columns), "float32")
        with T.Kernel(T.ceildiv(columns, bd), groups, threads=p["threads"]) as (bx, group):
            expert = tile_experts[layout, group]
            by = tile_rows[layout, group]
            if expert >= 0 and by * bm < counts[expert]:
                lhs = T.alloc_shared((bm, bk), "float16")
                gate = T.alloc_shared((bn, bk), "float16")
                up = T.alloc_shared((bn, bk), "float16")
                activation = T.alloc_shared((bm, bn), "float16")
                rhs_down = T.alloc_shared((bd, bn), "float16")
                g = T.alloc_fragment((bm, bn), "float32")
                u = T.alloc_fragment((bm, bn), "float32")
                accum = T.alloc_fragment((bm, bd), "float32")
                T.clear(accum)
                for hidden_tile in T.serial(T.ceildiv(hidden, bn)):
                    T.clear(g)
                    T.clear(u)
                    for tile in T.Pipelined(T.ceildiv(depth, bk), num_stages=1):
                        for i, j in T.Parallel(bm, bk):
                            index = by * bm + i
                            route = routes[expert, T.min(index, rows - 1)]
                            lhs[i, j] = T.if_then_else(
                                index < counts[expert] and tile * bk + j < depth,
                                x[T.max(route, 0) // top, T.min(tile * bk + j, depth - 1)],
                                0,
                            )
                        load_gate((w,), gate, expert * hidden + hidden_tile * bn, tile * bk)
                        load_gate((w2,), up, expert * hidden + hidden_tile * bn, tile * bk)
                        T.gemm(lhs, gate, g, transpose_B=True)
                        T.gemm(lhs, up, u, transpose_B=True)
                    for i, j in T.Parallel(bm, bn):
                        activation[i, j] = T.if_then_else(
                            hidden_tile * bn + j < hidden,
                            g[i, j] / (1 + T.exp(-g[i, j])) * u[i, j],
                            0,
                        )
                    load_down((wd,), rhs_down, expert * columns + bx * bd, hidden_tile * bn)
                    T.gemm(activation, rhs_down, accum, transpose_B=True)
                for i, j in T.Parallel(bm, bd):
                    index = by * bm + i
                    if index < counts[expert] and bx * bd + j < columns:
                        route = routes[expert, index]
                        out[route // top, route % top, bx * bd + j] = accum[i, j]
        return out

    return compute
