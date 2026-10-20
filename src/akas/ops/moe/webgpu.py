"""WebGPU specializations and shared bindings for moe."""

from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))


def expert_swiglu(spec, schedule):
    from .kernel import _expert_swiglu as portable

    return portable(spec, schedule, provider="webgpu")


def register_experts(spec, schedule):
    """Grouped expert tiles with warp-aligned columns and private accumulators."""
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import registers, vector
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.quantization.weights import bind_parts

    p = {**spec.parameters, **schedule}
    rows, top, experts, columns, depth = (
        p[n] for n in ("r", "top", "experts", "o", "k")
    )
    bm, bn, bk = p.get("block_m", 16), p.get("columns", 32), p.get("block_k", 32)
    pad = p.get("rhs_pad", 1)
    dot_width = p.get("dot_width", 4)
    micro_cols = p.get("micro_cols", 2)
    if bm < 128 // (bn // micro_cols):
        micro_cols = 1
    nr = bn // micro_cols
    row_stride = 128 // nr
    transpose = p.get("lhs_transpose", True)
    lhs_pad = p.get("lhs_pad", 1) if transpose else 0
    if (
        bn not in (32, 64)
        or bm not in (4, 8, 16, 32)
        or bm % row_stride
        or bk not in (32, 64)
    ):
        raise ValueError("invalid register expert tile")
    if (
        micro_cols not in (1, 2)
        or lhs_pad not in (0, 1)
        or pad not in (0, 1, 4)
        or 4 * ((bm + lhs_pad) * bk + bk * (bn + pad)) > 32768
    ):
        raise ValueError("register expert tile exceeds portable workgroup storage")
    if dot_width not in (1, 4):
        raise ValueError("register expert dot width must be one or four")
    micro_rows = bm // row_stride
    groups = (rows * top + bm - 1) // bm + experts
    capacity = (rows * top + 3) // 4 + experts
    layout = bm.bit_length() - 3
    binding = bind_parts(p["weight_parts"], depth, provider="webgpu")
    storage, read = binding.storage.values, binding.read
    routed = p.get("routed_input", False)
    shape = (rows, top, depth) if routed else (rows, depth)
    lhs_shape = (bk, bm + lhs_pad) if transpose else (bm, bk)

    @T.macro
    def update(acc: T.Ref, left, right):
        if dot_width == 1:
            acc = acc + left * right
        else:
            acc = acc + T.call_extern("float32", "dot", left, right)

    def accumulate(lhs, rhs, accum, kk, tx):
        right = [
            T.bind(
                vector(
                    [
                        rhs[kk * dot_width + lane, tx % nr + j * nr]
                        for lane in range(dot_width)
                    ]
                )
            )
            for j in range(micro_cols)
        ]
        left = [
            T.bind(
                vector(
                    [
                        lhs[kk * dot_width + lane, tx // nr + i * row_stride]
                        if transpose
                        else lhs[tx // nr + i * row_stride, kk * dot_width + lane]
                        for lane in range(dot_width)
                    ]
                )
            )
            for i in range(micro_rows)
        ]
        for i in range(micro_rows):
            for j in range(micro_cols):
                update(accum[i, j][0], left[i], right[j])

    @T.macro
    def store(out, routes, counts, value, expert, bx, by, tx, i, j):
        index = by * bm + tx // nr + i * row_stride
        if index < counts[expert] and bx * bn + tx % nr + j * nr < columns:
            route = routes[expert, index]
            out[route // top, route % top, bx * bn + tx % nr + j * nr] = value

    def stores(out, routes, counts, accum, expert, bx, by, tx):
        for i in range(micro_rows):
            for j in range(micro_cols):
                store(out, routes, counts, accum[i, j][0], expert, bx, by, tx, i, j)

    @kernel
    def compute(
        x: Tensor(shape, "float32"),
        w: Tensor(storage.shape, storage.dtype),
        counts: Tensor((experts,), "int32"),
        routes: Tensor((experts, rows), "int32"),
        tile_experts: Tensor((5, capacity), "int32"),
        tile_rows: Tensor((5, capacity), "int32"),
    ) -> Tensor((rows, top, columns), "float32"):
        out = output((rows, top, columns), "float32")
        with T.Kernel(T.ceildiv(columns, bn), groups, threads=128) as (bx, group):
            expert = tile_experts[layout, group]
            by = tile_rows[layout, group]
            if expert >= 0 and by * bm < counts[expert]:
                tx = T.get_thread_binding()
                lhs = T.alloc_shared(lhs_shape, "float32")
                rhs = T.alloc_shared((bk, bn + pad), "float32")
                accum = registers((micro_rows, micro_cols))
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
                        value = T.if_then_else(
                            index < counts[expert] and tile * bk + j < depth, source, 0
                        )
                        if transpose:
                            lhs[j, i] = value
                        else:
                            lhs[i, j] = value
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
                    for kk in T.serial(bk // dot_width):
                        accumulate(lhs, rhs, accum, kk, tx)
                    T.sync_threads()
                stores(out, routes, counts, accum, expert, bx, by, tx)
        return out

    return compute


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
        bm not in (4, 8, 16, 32)
        or bn not in (32, 64)
        or bk not in (32, 64)
        or pad not in (0, 1, 4)
    ):
        raise ValueError("invalid grouped WebGPU expert tile")
    # Portable lowering budgets fragment storage alongside the shared tiles.
    workgroup_bytes = 4 * (bm * bk + bk * (bn + pad) + bm * bn)
    if workgroup_bytes > 32768:
        raise ValueError("grouped WebGPU expert tile exceeds 32 KiB workgroup storage")
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


def packed_float_experts(spec, schedule):
    """Decode through the codec F32 lane-product capability."""
    from .kernel import _packed_float_experts

    return _packed_float_experts(spec, schedule, provider="webgpu")
