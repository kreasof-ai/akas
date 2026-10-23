from akas.kernels.kernel import Tensor,kernel,output
def staged_experts(spec, schedule):
    """Grouped tensor-core projection dispatched through compact expert tiles."""
    import tilelang.language as T
    from experimental_staged_store64_codec import bind as bind_parts

    p = {**spec.parameters, **schedule}
    rows, top, experts, columns, depth = (
        p[n] for n in ("r", "top", "experts", "o", "k")
    )
    bm, bn, bk = p.get("block_m", 16), p.get("columns", 64), p.get("block_k", 32)
    if bm not in (16, 32, 64) or bn not in (32, 64, 128) or bk not in (32, 64, 128, 256, 512):
        raise ValueError("invalid compact CUDA expert tile")
    threads = p.get("threads", 128)
    if threads not in (128, 256):
        raise ValueError("compact CUDA expert tiles require 128 or 256 threads")
    capacity = (rows * top + 3) // 4 + experts
    groups = (rows * top + bm - 1) // bm + experts
    layout = bm.bit_length() - 3
    binding = bind_parts(p["weight_parts"], depth, provider="cuda")
    storage, read_tile = binding.storage.values, binding.require("read_tile")
    routed = p.get("routed_input", False)
    shape = (rows, top, depth) if routed else (rows, depth)
    input_dtype = p.get("input_dtype", "float32")

    @T.macro
    def activation(x, route, channel):
        if routed:
            return x[route // top, route % top, channel]
        else:
            return x[route // top, channel]

    @kernel
    def compute(
        x: Tensor(shape, input_dtype),
        w: Tensor(storage.shape, storage.dtype),
        counts: Tensor((experts,), "int32"),
        routes: Tensor((experts, rows), "int32"),
        tile_experts: Tensor((5, capacity), "int32"),
        tile_rows: Tensor((5, capacity), "int32"),
    ) -> Tensor((rows, top, columns), "float32"):
        out = output((rows, top, columns), "float32")
        with T.Kernel(T.ceildiv(columns, bn), groups, threads=threads) as (bx, group):
            expert = tile_experts[layout, group]
            by = tile_rows[layout, group]
            if expert >= 0 and by * bm < counts[expert]:
                lhs = T.alloc_shared((bm, bk), "float16")
                rhs = T.alloc_shared((bn, bk), "float16")
                accum = T.alloc_fragment((bm, bn), "float32")
                T.clear(accum)
                for tile in T.Pipelined(T.ceildiv(depth, bk), num_stages=p.get("stages", 2)):
                    for i, j in T.Parallel(bm, bk):
                        index = by * bm + i
                        route = routes[expert, T.min(index, rows - 1)]
                        lhs[i, j] = T.if_then_else(
                            index < counts[expert] and tile * bk + j < depth,
                            activation(
                                x, T.max(route, 0), T.min(tile * bk + j, depth - 1)
                            ),
                            0,
                        )
                    read_tile((w,), rhs, expert * columns + bx * bn, tile * bk)
                    T.gemm(lhs, rhs, accum, transpose_B=True)
                for i, j in T.Parallel(bm, bn):
                    index = by * bm + i
                    if index < counts[expert] and bx * bn + j < columns:
                        route = routes[expert, index]
                        out[route // top, route % top, bx * bn + j] = accum[i, j]
        return out

    return compute

def make(spec,schedule):
 from akas.kernels.definition import KernelSpec
 return staged_experts(KernelSpec(**spec),schedule)
