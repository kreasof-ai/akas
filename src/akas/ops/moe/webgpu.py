"""WebGPU specializations and shared bindings for moe."""

from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))


def tiled_experts(spec, schedule):
    """Reuse each decoded weight tile across compact rows of the same expert."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.quantization.weights import bind_parts

    p = {**spec.parameters, **schedule}
    rows, top, experts, columns, depth = (
        p[n] for n in ("r", "top", "experts", "o", "k")
    )
    bm, bn, bk = p.get("block_m", 8), p.get("columns", 32), p.get("block_k", 32)
    pad = p.get("rhs_pad", 1)
    if (
        bm not in (8, 16)
        or bn not in (32, 64)
        or bk not in (32, 64)
        or pad not in (0, 1, 4)
    ):
        raise ValueError("invalid grouped WebGPU expert tile")
    binding = bind_parts(p["weight_parts"], depth, provider="webgpu")
    storage, read = binding.storage.values, binding.read
    routed = p.get("routed_input", False)
    shape = (rows, top, depth) if routed else (rows, depth)

    @kernel
    def compute(
        x: Tensor(shape, "float32"),
        w: Tensor(storage.shape, storage.dtype),
        counts: Tensor((experts,), "int32"),
        routes: Tensor((experts, rows), "int32"),
    ) -> Tensor((rows, top, columns), "float32"):
        out = output((rows, top, columns), "float32")
        with T.Kernel(
            T.ceildiv(columns, bn), T.ceildiv(rows, bm), experts, threads=128
        ) as (bx, by, expert):
            if by * bm < counts[expert]:
                lhs = T.alloc_shared((bm, bk), "float32")
                rhs = T.alloc_shared((bk, bn + pad), "float32")
                accum = T.alloc_fragment((bm, bn), "float32")
                T.clear(accum)
                for tile in T.serial(T.ceildiv(depth, bk)):
                    for i, j in T.Parallel(bm, bk):
                        index = by * bm + i
                        route = T.max(routes[expert, T.min(index, rows - 1)], 0)
                        source = T.alloc_var("float32")
                        if routed:
                            source = x[
                                route // top,
                                route % top,
                                T.min(tile * bk + j, depth - 1),
                            ]
                        else:
                            source = x[route // top, T.min(tile * bk + j, depth - 1)]
                        lhs[i, j] = T.if_then_else(
                            index < counts[expert] and tile * bk + j < depth, source, 0
                        )
                    for i, j in T.Parallel(bn, bk):
                        rhs[j, i] = T.if_then_else(
                            bx * bn + i < columns and tile * bk + j < depth,
                            read(
                                (w,),
                                expert * columns + T.min(bx * bn + i, columns - 1),
                                T.min(tile * bk + j, depth - 1),
                            ),
                            0,
                        )
                    T.sync_threads()
                    for kk in T.serial(bk):
                        for i, j in T.Parallel(bm, bn):
                            accum[i, j] += lhs[i, kk] * rhs[kk, j]
                    T.sync_threads()
                for i, j in T.Parallel(bm, bn):
                    index = by * bm + i
                    if index < counts[expert] and bx * bn + j < columns:
                        route = routes[expert, index]
                        out[route // top, route % top, bx * bn + j] = accum[i, j]
        return out

    return compute


def warp_experts(spec, schedule):
    """Decode on logical 32-lane subgroups, with a shared-reduction fallback."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.quantization.weights import bind_parts

    p = {**spec.parameters, **schedule}
    top, experts, columns, depth = (p[n] for n in ("top", "experts", "o", "k"))
    binding = bind_parts(p["weight_parts"], depth, provider="webgpu")
    storage, read = binding.storage.values, binding.read
    routed = p.get("routed_input", False)
    shape = (1, top, depth) if routed else (1, depth)

    @kernel
    def compute(
        x: Tensor(shape, "float32"),
        w: Tensor(storage.shape, storage.dtype),
        ids: Tensor((1, top), "int32"),
        counts: Tensor((experts,), "int32"),
    ) -> Tensor((1, top, columns), "float32"):
        out = output((1, top, columns), "float32")
        with T.Kernel(T.ceildiv(columns, 4), top, threads=128) as (bx, rank):
            tx = T.get_thread_binding()
            lane, warp = tx % 32, tx // 32
            scratch = T.alloc_shared((128,), "float32")
            accum = T.alloc_var("float32")
            accum = 0
            for tile in T.serial(T.ceildiv(depth, 32)):
                channel = tile * 32 + lane
                if channel < depth and bx * 4 + warp < columns:
                    value = T.alloc_var("float32")
                    if routed:
                        value = x[0, rank, channel]
                    else:
                        value = x[0, channel]
                    accum += value * read(
                        (w,), ids[0, rank] * columns + bx * 4 + warp, channel
                    )
            if T.call_extern("uint32", "tensor_subgroup_size") >= 32:
                for step in T.unroll(5):
                    accum += T.call_extern(
                        "float32", "subgroupShuffleXor", accum, T.uint32(16 >> step)
                    )
            else:
                scratch[tx] = accum
                T.sync_threads()
                for step in T.unroll(5):
                    if lane < 16 >> step:
                        scratch[tx] += scratch[tx + (16 >> step)]
                    T.sync_threads()
                accum = scratch[tx]
            if lane == 0 and bx * 4 + warp < columns:
                out[0, rank, bx * 4 + warp] = T.if_then_else(
                    counts[ids[0, rank]] > 0, accum, 0
                )
        return out

    return compute
