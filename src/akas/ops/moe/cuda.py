"""CUDA specializations and shared bindings for moe."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def quantized_experts(spec, schedule):
    """Packed expert GEMV for decode; grouped tensor-core GEMM for prefill."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.quantization.weights import bind_parts

    p = {**spec.parameters, **schedule}
    rows, top, experts, columns, depth = (
        p[name] for name in ("r", "top", "experts", "o", "k")
    )
    binding = bind_parts(p["weight_parts"], depth, provider="cuda")
    storage, read = (binding.storage.values, binding.read)
    routed = p.get("routed_input", False)
    shape = (rows, top, depth) if routed else (rows, depth)
    input_dtype = p.get("input_dtype", "bfloat16")
    bm, bn, bk = (
        p.get("block_m", 16),
        p.get("columns", 64),
        p.get("block_k", 32),
    )

    @T.macro
    def activation(x, route, channel):
        if routed:
            return x[route // top, route % top, channel]
        else:
            return x[route // top, channel]

    if rows == 1:

        @kernel
        def compute(
            x: Tensor(shape, input_dtype),
            w: Tensor(storage.shape, storage.dtype),
            ids: Tensor((rows, top), "int32"),
            counts: Tensor((experts,), "int32"),
        ) -> Tensor((rows, top, columns), "float32"):
            out = output((rows, top, columns), "float32")
            with T.Kernel(T.ceildiv(columns, 4), top, threads=128) as (bx, rank):
                expert = ids[0, rank]
                tx = T.get_thread_binding()
                warp, lane = (tx // 32, tx % 32)
                source = T.alloc_shared((128,), "float16")
                scratch = T.alloc_shared((128,), "float32")
                accum = T.alloc_var("float32")
                accum = 0
                if p.get("warp_shuffle", False):
                    for tile in T.serial(T.ceildiv(depth, 32)):
                        channel = tile * 32 + lane
                        if channel < depth and bx * 4 + warp < columns:
                            coefficient = T.cast(
                                read((w,), expert * columns + bx * 4 + warp, channel),
                                "float16",
                            )
                            accum += T.cast(
                                T.cast(activation(x, rank, channel), "float16"),
                                "float32",
                            ) * T.cast(coefficient, "float32")
                    for step in T.unroll(5):
                        accum += T.shfl_down(accum, 16 >> step)
                else:
                    for tile in T.serial(T.ceildiv(depth, 128)):
                        source[tx] = T.if_then_else(
                            tile * 128 + tx < depth,
                            activation(x, rank, tile * 128 + tx),
                            0,
                        )
                        T.sync_threads()
                        for step in T.unroll(4):
                            channel = tile * 128 + step * 32 + lane
                            if channel < depth and bx * 4 + warp < columns:
                                coefficient = T.cast(
                                    read(
                                        (w,), expert * columns + bx * 4 + warp, channel
                                    ),
                                    "float16",
                                )
                                accum += T.cast(
                                    source[step * 32 + lane], "float32"
                                ) * T.cast(coefficient, "float32")
                        T.sync_threads()
                    scratch[tx] = accum
                    T.sync_threads()
                    for step in T.unroll(5):
                        if lane < 16 >> step:
                            scratch[tx] += scratch[tx + (16 >> step)]
                        T.sync_threads()
                    accum = scratch[tx]
                if lane == 0 and bx * 4 + warp < columns:
                    out[0, rank, bx * 4 + warp] = T.if_then_else(
                        counts[expert] > 0, accum, 0
                    )
            return out
    else:

        @kernel
        def compute(
            x: Tensor(shape, input_dtype),
            w: Tensor(storage.shape, storage.dtype),
            counts: Tensor((experts,), "int32"),
            routes: Tensor((experts, rows), "int32"),
        ) -> Tensor((rows, top, columns), "float32"):
            out = output((rows, top, columns), "float32")
            with T.Kernel(
                T.ceildiv(columns, bn), T.ceildiv(rows, bm), experts, threads=128
            ) as (bx, by, expert):
                if by * bm < counts[expert]:
                    lhs = T.alloc_shared((bm, bk), "float16")
                    rhs = T.alloc_shared((bn, bk), "float16")
                    accum = T.alloc_fragment((bm, bn), "float32")
                    T.clear(accum)
                    for tile in T.Pipelined(T.ceildiv(depth, bk), num_stages=2):
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
                        for i, j in T.Parallel(bn, bk):
                            rhs[i, j] = T.if_then_else(
                                bx * bn + i < columns and tile * bk + j < depth,
                                read(
                                    (w,),
                                    expert * columns + T.min(bx * bn + i, columns - 1),
                                    T.min(tile * bk + j, depth - 1),
                                ),
                                0,
                            )
                        T.gemm(lhs, rhs, accum, transpose_B=True)
                    for i, j in T.Parallel(bm, bn):
                        index = by * bm + i
                        if index < counts[expert] and bx * bn + j < columns:
                            route = routes[expert, index]
                            out[route // top, route % top, bx * bn + j] = accum[i, j]
            return out

    return compute


def compact_experts(spec, schedule):
    """Grouped tensor-core projection dispatched through compact expert tiles."""
    import tilelang.language as T
    from akas.quantization.weights import bind_parts

    p = {**spec.parameters, **schedule}
    rows, top, experts, columns, depth = (
        p[n] for n in ("r", "top", "experts", "o", "k")
    )
    bm, bn, bk = p.get("block_m", 16), p.get("columns", 64), p.get("block_k", 32)
    if bm not in (16, 32, 64) or bn not in (32, 64, 128) or bk not in (32, 64, 128):
        raise ValueError("invalid compact CUDA expert tile")
    capacity = (rows * top + 3) // 4 + experts
    groups = (rows * top + bm - 1) // bm + experts
    layout = bm.bit_length() - 3
    binding = bind_parts(p["weight_parts"], depth, provider="cuda")
    storage, read = binding.storage.values, binding.read
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
        with T.Kernel(T.ceildiv(columns, bn), groups, threads=128) as (bx, group):
            expert = tile_experts[layout, group]
            by = tile_rows[layout, group]
            if expert >= 0 and by * bm < counts[expert]:
                lhs = T.alloc_shared((bm, bk), "float16")
                rhs = T.alloc_shared((bn, bk), "float16")
                accum = T.alloc_fragment((bm, bn), "float32")
                T.clear(accum)
                for tile in T.Pipelined(T.ceildiv(depth, bk), num_stages=2):
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
                    for i, j in T.Parallel(bn, bk):
                        rhs[i, j] = T.if_then_else(
                            bx * bn + i < columns and tile * bk + j < depth,
                            read(
                                (w,),
                                expert * columns + T.min(bx * bn + i, columns - 1),
                                T.min(tile * bk + j, depth - 1),
                            ),
                            0,
                        )
                    T.gemm(lhs, rhs, accum, transpose_B=True)
                for i, j in T.Parallel(bm, bn):
                    index = by * bm + i
                    if index < counts[expert] and bx * bn + j < columns:
                        route = routes[expert, index]
                        out[route // top, route % top, bx * bn + j] = accum[i, j]
        return out

    return compute


def quantized_experts_warp(spec: KernelSpec, schedule: Mapping[str, object]):
    """Reuse each activation across several output columns within a warp."""
    import tilelang.language as T
    from akas.quantization.weights import bind_parts

    p = {**spec.parameters, **schedule}
    rows, top, experts, columns, depth = (
        p[n] for n in ("r", "top", "experts", "o", "k")
    )
    width, threads, unroll = (
        p.get("columns_per_warp", 2),
        p.get("threads", 128),
        p.get("unroll", 2),
    )
    if (
        rows != 1
        or width not in (1, 2, 4)
        or threads not in (128, 256)
        or unroll not in (1, 2, 4)
    ):
        raise ValueError("unsupported expert decode schedule")
    binding = bind_parts(p["weight_parts"], depth, provider="cuda")
    storage, read = binding.storage.values, binding.read
    routed = p.get("routed_input", False)
    shape = (rows, top, depth) if routed else (rows, depth)
    input_dtype = p.get("input_dtype", "bfloat16")
    warps = threads // 32

    @kernel
    def compute(
        x: Tensor(shape, input_dtype),
        w: Tensor(storage.shape, storage.dtype),
        ids: Tensor((rows, top), "int32"),
        counts: Tensor((experts,), "int32"),
    ) -> Tensor((rows, top, columns), "float32"):
        out = output((rows, top, columns), "float32")
        with T.Kernel(T.ceildiv(columns, warps * width), top, threads=threads) as (
            block,
            rank,
        ):
            tx = T.get_thread_binding()
            warp, lane = tx // 32, tx % 32
            expert = ids[0, rank]
            accum = T.alloc_local((width,), "float32")
            for col in T.unroll(width):
                accum[col] = 0
            for tile in T.serial(T.ceildiv(depth, 32 * unroll)):
                for step in T.unroll(unroll):
                    channel = (tile * unroll + step) * 32 + lane
                    activation = T.alloc_var("float32")
                    activation = 0
                    if channel < depth:
                        if routed:
                            activation = T.cast(
                                T.cast(x[0, rank, channel], "float16"), "float32"
                            )
                        else:
                            activation = T.cast(
                                T.cast(x[0, channel], "float16"), "float32"
                            )
                    for col in T.unroll(width):
                        column = (block * warps + warp) * width + col
                        if column < columns and channel < depth:
                            coefficient = T.cast(
                                T.cast(
                                    read((w,), expert * columns + column, channel),
                                    "float16",
                                ),
                                "float32",
                            )
                            accum[col] += activation * coefficient
            for col in T.unroll(width):
                for step in T.unroll(5):
                    accum[col] += T.shfl_down(accum[col], 16 >> step)
                column = (block * warps + warp) * width + col
                if lane == 0 and column < columns:
                    out[0, rank, column] = T.if_then_else(
                        counts[expert] > 0, accum[col], 0
                    )
        return out

    return compute


def decode_moe_combine(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, top, c = (p["r"], p["top"], p["c"])

    @kernel
    def compute(
        experts: Tensor((r, top, c), "float32"),
        weights: Tensor((r, top), "float32"),
        shared: Tensor((r, c), "float32"),
        gate: Tensor((r, 1), "float32"),
    ) -> Tensor((r, c), "float32"):
        out = output((r, c), "float32")
        with T.Kernel(r, T.ceildiv(c, 256), threads=256) as (b, tile):
            for j in T.Parallel(256):
                col = tile * 256 + j
                if col < c:
                    total = T.alloc_var("float32")
                    total = 0
                    for rank in T.serial(top):
                        total += (
                            T.cast(T.cast(experts[b, rank, col], "bfloat16"), "float32")
                            * weights[b, rank]
                        )
                    shared_value = T.cast(T.cast(shared[b, col], "bfloat16"), "float32")
                    shared_gate = T.cast(T.cast(gate[b, 0], "bfloat16"), "float32")
                    out[b, col] = total + shared_value / (1 + T.exp(-shared_gate))
        return out

    return compute


def fp8_inline(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.cuda import _mma

    p = {**spec.parameters, **schedule}
    return _mma(p, grouped=True, prequantized=False)


def fp8_prequantized(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.cuda import _mma

    p = {**spec.parameters, **schedule}
    return _mma(p, grouped=True, prequantized=True)


def decode_fp8_experts(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, o, top, e = (p[n] for n in ("r", "k", "o", "top", "experts"))
    columns, threads = (p.get("columns", 4), p.get("threads", 128))
    routed = p.get("routed_input", False)

    @kernel
    def compute(
        x: Tensor((r, top, k) if routed else (r, k), "bfloat16"),
        w: Tensor((e, o, k), "uint8"),
        scales: Tensor((e, T.ceildiv(o, 128), k // 128), "bfloat16"),
        experts: Tensor((r * top,), "int32"),
        routes: Tensor((r * top, r), "int32"),
    ) -> Tensor((r, top, o), "float32"):
        out = output((r, top, o), "float32")
        with T.Kernel(o // columns, r * top, threads=threads) as (bx, group):
            expert = experts[group]
            if expert >= 0:
                accum = T.alloc_fragment((r, columns, 32), "float32")
                total = T.alloc_fragment((r, columns), "float32")
                coefficient = T.alloc_fragment((columns, 32), "float32")
                activation = T.alloc_shared((r, 128), "float32")
                absolute = T.alloc_fragment((r, 128), "float32")
                maximum = T.alloc_fragment((r,), "float32")
                T.clear(accum)
                for tile in T.serial(k // 128):
                    for b, j in T.Parallel(r, 128):
                        if routes[group, b] >= 0:
                            if routed:
                                absolute[b, j] = T.abs(
                                    T.cast(
                                        x[b, routes[group, b], tile * 128 + j],
                                        "float32",
                                    )
                                )
                            else:
                                absolute[b, j] = T.abs(
                                    T.cast(x[b, tile * 128 + j], "float32")
                                )
                        else:
                            absolute[b, j] = 0
                    T.reduce_max(absolute, maximum, dim=1)
                    for b, j in T.Parallel(r, 128):
                        if routes[group, b] >= 0:
                            scale = T.max(maximum[b], T.float32(1e-12)) / T.float32(448)
                            if routed:
                                value = T.cast(
                                    x[b, routes[group, b], tile * 128 + j], "float32"
                                )
                            else:
                                value = T.cast(x[b, tile * 128 + j], "float32")
                            encoded = T.call_extern(
                                "uint32", "tensor_encode_e4m3", value / scale
                            )
                            activation[b, j] = (
                                T.call_extern("float32", "tensor_decode_e4m3", encoded)
                                * scale
                            )
                        else:
                            activation[b, j] = 0
                    for part in T.unroll(4):
                        for n, lane in T.Parallel(columns, 32):
                            coefficient[n, lane] = T.call_extern(
                                "float32",
                                "tensor_decode_e4m3",
                                T.cast(
                                    w[
                                        expert,
                                        bx * columns + n,
                                        tile * 128 + part * 32 + lane,
                                    ],
                                    "uint32",
                                ),
                            )
                            coefficient[n, lane] *= T.cast(
                                scales[expert, (bx * columns + n) // 128, tile],
                                "float32",
                            )
                        for b, n, lane in T.Parallel(r, columns, 32):
                            accum[b, n, lane] = T.ieee_fmaf(
                                activation[b, part * 32 + lane],
                                coefficient[n, lane],
                                accum[b, n, lane],
                            )
                T.reduce_sum(accum, total, dim=2)
                for b, n in T.Parallel(r, columns):
                    if routes[group, b] >= 0:
                        out[b, routes[group, b], bx * columns + n] = total[b, n]
        return out

    return compute


def prefill_fp8_experts(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.moe.cuda import _bucketed as expert_kernel

    p = {**spec.parameters, **schedule}
    return expert_kernel({**p, "rows": p["r"]})


def _bucketed(p):
    import tilelang.language as T

    rows, k, o = (p["rows"], p["k"], p["o"])
    n, top, experts = (p.get("columns", 64), 8, 256)
    m, threads = (p.get("block_m", 16), p.get("threads", 128))
    routed = p.get("routed_input", False)
    compact = p.get("compact", False)
    max_tiles = T.ceildiv(rows * top, m) + experts - 1

    @T.macro
    def body(x, activation_scales, w, scales, counts, routes, out, bx, expert, tokens):
        if tokens * m < counts[expert]:
            lhs = T.alloc_shared((m, 128), "float8_e4m3fn")
            rhs = T.alloc_shared((n, 128), "float8_e4m3fn")
            scale = T.alloc_shared((m,), "float32")
            selected = T.alloc_shared((m,), "int32")
            block = T.alloc_fragment((m, n), "float32")
            total = T.alloc_fragment((m, n), "float32")
            for b in T.Parallel(m):
                selected[b] = T.if_then_else(
                    tokens * m + b < counts[expert],
                    routes[expert, T.min(tokens * m + b, rows - 1)],
                    -1,
                )
            T.sync_threads()
            T.clear(total)
            for tile in T.serial(k // 128):
                for b in T.Parallel(m):
                    scale[b] = 1
                    if selected[b] >= 0:
                        if routed:
                            scale[b] = activation_scales[
                                selected[b] // top, selected[b] % top, tile
                            ]
                        else:
                            scale[b] = activation_scales[selected[b] // top, tile]
                T.sync_threads()
                for b, j in T.Parallel(m, 128):
                    bits = T.alloc_var("uint8")
                    bits = 0
                    if selected[b] >= 0:
                        if routed:
                            bits = x[
                                selected[b] // top, selected[b] % top, tile * 128 + j
                            ]
                        else:
                            bits = x[selected[b] // top, tile * 128 + j]
                    lhs[b, j] = T.reinterpret("float8_e4m3fn", bits)
                weights = T.view(w, dtype="float8_e4m3fn")
                T.copy(
                    weights[expert, bx * n : bx * n + n, tile * 128 : tile * 128 + 128],
                    rhs,
                )
                T.gemm(lhs, rhs, block, transpose_B=True, clear_accum=True)
                for b, j in T.Parallel(m, n):
                    if compact:
                        product = T.call_extern(
                            "float32", "__fmul_rn", block[b, j], scale[b]
                        )
                        total[b, j] = T.call_extern(
                            "float32",
                            "__fmaf_rn",
                            product,
                            T.cast(
                                scales[expert, (bx * n + j) // 128, tile], "float32"
                            ),
                            total[b, j],
                        )
                    else:
                        total[b, j] += (
                            block[b, j]
                            * scale[b]
                            * T.cast(
                                scales[expert, (bx * n + j) // 128, tile], "float32"
                            )
                        )
            for b, j in T.Parallel(m, n):
                if selected[b] >= 0:
                    out[selected[b] // top, selected[b] % top, bx * n + j] = total[b, j]

    if compact:

        @kernel
        def compute(
            x: Tensor((rows, top, k) if routed else (rows, k), "uint8"),
            activation_scales: Tensor(
                (rows, top, k // 128) if routed else (rows, k // 128), "float32"
            ),
            w: Tensor((experts, o, k), "uint8"),
            scales: Tensor((experts, T.ceildiv(o, 128), k // 128), "bfloat16"),
            counts: Tensor((experts,), "int32"),
            routes: Tensor((experts, rows), "int32"),
            tile_experts: Tensor((max_tiles,), "int32"),
            tile_offsets: Tensor((max_tiles,), "int32"),
        ) -> Tensor((rows, top, o), "float32"):
            out = output((rows, top, o), "float32", position=6)
            with T.Kernel(o // n, max_tiles, threads=threads) as (bx, tile):
                expert = tile_experts[tile]
                if expert >= 0:
                    body(
                        x,
                        activation_scales,
                        w,
                        scales,
                        counts,
                        routes,
                        out,
                        bx,
                        expert,
                        tile_offsets[tile],
                    )
            return out

        return compute
    if not compact:

        @kernel
        def compute(
            x: Tensor((rows, top, k) if routed else (rows, k), "uint8"),
            activation_scales: Tensor(
                (rows, top, k // 128) if routed else (rows, k // 128), "float32"
            ),
            w: Tensor((experts, o, k), "uint8"),
            scales: Tensor((experts, T.ceildiv(o, 128), k // 128), "bfloat16"),
            counts: Tensor((experts,), "int32"),
            routes: Tensor((experts, rows), "int32"),
        ) -> Tensor((rows, top, o), "float32"):
            out = output((rows, top, o), "float32")
            with T.Kernel(o // n, experts, T.ceildiv(rows, m), threads=threads) as (
                bx,
                expert,
                tokens,
            ):
                body(
                    x,
                    activation_scales,
                    w,
                    scales,
                    counts,
                    routes,
                    out,
                    bx,
                    expert,
                    tokens,
                )
            return out

        return compute


def bucketed(spec: KernelSpec, schedule: Mapping[str, object]):
    """Bucketed expert GEMM with the graph's logical row count."""
    p = {**spec.parameters, **schedule}
    return _bucketed({**p, "rows": p["r"]})


def partial(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    rows, k, o = (p["rows"], p["k"], p["o"])
    n, top, experts = (p.get("columns", 64), 8, 256)
    m, threads = (p.get("block_m", 16), p.get("threads", 128))
    parts = p.get("partitions", 4)
    stages = p.get("stages", 2)
    routed = p.get("routed_input", False)

    @kernel
    def compute(
        x: Tensor((rows, top, k) if routed else (rows, k), "uint8"),
        activation_scales: Tensor(
            (rows, top, k // 128) if routed else (rows, k // 128), "float32"
        ),
        w: Tensor((experts, o, k), "uint8"),
        scales: Tensor((experts, T.ceildiv(o, 128), k // 128), "bfloat16"),
        counts: Tensor((experts,), "int32"),
        routes: Tensor((experts, rows), "int32"),
    ) -> Tensor((rows, top, parts, o), "float32"):
        out = output((rows, top, parts, o), "float32")
        with T.Kernel(o // n, experts, T.ceildiv(rows, m) * parts, threads=threads) as (
            bx,
            expert,
            group,
        ):
            tokens = group // parts
            part = group % parts
            if tokens * m < counts[expert]:
                lhs = T.alloc_shared((m, 128), "float8_e4m3fn")
                rhs = T.alloc_shared((n, 128), "float8_e4m3fn")
                scale = T.alloc_shared((m,), "float32")
                selected = T.alloc_shared((m,), "int32")
                block = T.alloc_fragment((m, n), "float32")
                total = T.alloc_fragment((m, n), "float32")
                for b in T.Parallel(m):
                    selected[b] = T.if_then_else(
                        tokens * m + b < counts[expert],
                        routes[expert, T.min(tokens * m + b, rows - 1)],
                        -1,
                    )
                T.sync_threads()
                T.clear(total)
                for local in T.Pipelined(k // (128 * parts), num_stages=stages):
                    tile = part * (k // (128 * parts)) + local
                    for b in T.Parallel(m):
                        scale[b] = 1
                        if selected[b] >= 0:
                            if routed:
                                scale[b] = activation_scales[
                                    selected[b] // top, selected[b] % top, tile
                                ]
                            else:
                                scale[b] = activation_scales[selected[b] // top, tile]
                    T.sync_threads()
                    for b, j in T.Parallel(m, 128):
                        bits = T.alloc_var("uint8")
                        bits = 0
                        if selected[b] >= 0:
                            if routed:
                                bits = x[
                                    selected[b] // top,
                                    selected[b] % top,
                                    tile * 128 + j,
                                ]
                            else:
                                bits = x[selected[b] // top, tile * 128 + j]
                        lhs[b, j] = T.reinterpret("float8_e4m3fn", bits)
                    weights = T.view(w, dtype="float8_e4m3fn")
                    T.copy(
                        weights[
                            expert, bx * n : bx * n + n, tile * 128 : tile * 128 + 128
                        ],
                        rhs,
                    )
                    T.gemm(lhs, rhs, block, transpose_B=True, clear_accum=True)
                    for b, j in T.Parallel(m, n):
                        total[b, j] += (
                            block[b, j]
                            * scale[b]
                            * T.cast(
                                scales[expert, (bx * n + j) // 128, tile], "float32"
                            )
                        )
                for b, j in T.Parallel(m, n):
                    if selected[b] >= 0:
                        out[selected[b] // top, selected[b] % top, part, bx * n + j] = (
                            total[b, j]
                        )
        return out

    return compute


def split(spec: KernelSpec, schedule: Mapping[str, object]):
    from dataclasses import replace

    return partial(
        replace(spec, parameters={**spec.parameters, "rows": spec.parameters["r"]}),
        schedule,
    )


def hopper_experts_partial(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    rows, k, o = (p["rows"], p["k"], p["o"])
    n, top, experts = (p.get("columns", 64), 8, 256)
    m, threads = (p.get("block_m", 16), p.get("threads", 128))
    routed = p.get("routed_input", False)
    compact = p.get("compact", True)
    stages = p.get("stages", 1)
    packed_gather = p.get("packed_gather", False)
    bf16_mma = p.get("bf16_mma", False)
    parts = p.get("partitions", 1)
    persistent = p.get("persistent_tiles", 0)
    if type(persistent) is not int or persistent < 0:
        raise ValueError("invalid persistent expert grid")
    reduction = p.get("mma_reduction", 128)
    reorder = p.get("mma_reorder", False)
    asynchronous = p.get("async_mma", False)
    packed_widen = p.get("packed_widen", False)
    resident_weights = p.get("resident_weights", False)
    if resident_weights and (
        not bf16_mma or not reorder or asynchronous or packed_widen
    ):
        raise ValueError("resident operands require synchronous paired MMA")
    if packed_widen and (not bf16_mma or not reorder or asynchronous):
        raise ValueError("packed widening requires synchronous paired MMA")
    if asynchronous and (not (reorder and reduction == 32)):
        raise ValueError("async MMA requires paired K32")
    if reorder and reduction != 32:
        raise ValueError("paired operands require K32")
    if reduction not in (32, 128):
        raise ValueError("invalid Hopper MMA reduction")
    if k % (128 * parts):
        raise ValueError("invalid expert reduction partition")
    max_tiles = T.ceildiv(rows * top, m) + experts - 1

    @T.macro
    def body(
        x, activation_scales, w, scales, counts, routes, out, bx, expert, tokens, part
    ):
        if tokens * m < counts[expert]:
            if packed_widen:
                from akas.quantization.weights.fp8.cuda import CUDA_SOURCE

                T.import_source(CUDA_SOURCE)
            lhs = T.alloc_shared((m, 128), "float8_e4m3fn")
            rhs = T.alloc_shared((n, 128), "float8_e4m3fn")
            if bf16_mma:
                if asynchronous:
                    lhs_batch = T.alloc_shared((m, 128), "float16")
                    rhs_batch = T.alloc_shared((n, 128), "float16")
                    p1 = T.alloc_fragment((m, n), "float32")
                    p2 = T.alloc_fragment((m, n), "float32")
                    p3 = T.alloc_fragment((m, n), "float32")
                if reduction == 128:
                    lhs_wide = T.alloc_shared((m, 128), "bfloat16")
                    rhs_wide = T.alloc_shared((n, 128), "bfloat16")
                if reduction < 128:
                    lhs_sub = T.alloc_shared((m, reduction), "float16")
                    rhs_sub = T.alloc_shared((n, reduction), "float16")
                    partial = T.alloc_fragment((m, n), "float32")
            scale = T.alloc_shared((m,), "float32")
            selected = T.alloc_shared((m,), "int32")
            block = T.alloc_fragment((m, n), "float32")
            total = T.alloc_fragment((m, n), "float32")
            for b in T.Parallel(m):
                selected[b] = T.if_then_else(
                    tokens * m + b < counts[expert],
                    routes[expert, T.min(tokens * m + b, rows - 1)],
                    -1,
                )
            T.sync_threads()
            T.clear(total)
            for local in T.Pipelined(k // (128 * parts), num_stages=stages):
                tile = part * (k // (128 * parts)) + local
                for b in T.Parallel(m):
                    scale[b] = 1
                    if selected[b] >= 0:
                        if routed:
                            scale[b] = activation_scales[
                                selected[b] // top, selected[b] % top, tile
                            ]
                        else:
                            scale[b] = activation_scales[selected[b] // top, tile]
                T.sync_threads()
                if packed_gather:
                    words = T.view(
                        x,
                        shape=(rows, top, k // 4) if routed else (rows, k // 4),
                        dtype="uint32",
                    )
                    destination = T.view(lhs, shape=(m, 32), dtype="uint32")
                    for b, j in T.Parallel(m, 32):
                        bits = T.alloc_var("uint32")
                        bits = 0
                        if selected[b] >= 0:
                            if routed:
                                bits = words[
                                    selected[b] // top, selected[b] % top, tile * 32 + j
                                ]
                            else:
                                bits = words[selected[b] // top, tile * 32 + j]
                        if stages > 1:
                            for byte in T.unroll(4):
                                lhs[b, j * 4 + byte] = T.reinterpret(
                                    "float8_e4m3fn",
                                    T.cast(bits >> byte * 8 & 255, "uint8"),
                                )
                        else:
                            destination[b, j] = bits
                else:
                    for b, j in T.Parallel(m, 128):
                        bits = T.alloc_var("uint8")
                        bits = 0
                        if selected[b] >= 0:
                            if routed:
                                bits = x[
                                    selected[b] // top,
                                    selected[b] % top,
                                    tile * 128 + j,
                                ]
                            else:
                                bits = x[selected[b] // top, tile * 128 + j]
                        lhs[b, j] = T.reinterpret("float8_e4m3fn", bits)
                if not resident_weights:
                    weights = T.view(w, dtype="float8_e4m3fn")
                    T.copy(
                        weights[
                            expert, bx * n : bx * n + n, tile * 128 : tile * 128 + 128
                        ],
                        rhs,
                    )
                if packed_gather:
                    T.sync_threads()
                if bf16_mma:
                    if asynchronous:
                        for b, j in T.Parallel(m, 128):
                            lhs_batch[b, j] = lhs[
                                b,
                                j // 32 * 32
                                + j % 16 // 2 * 4
                                + j % 2
                                + j % 32 // 16 * 2,
                            ]
                        for i, j in T.Parallel(n, 128):
                            rhs_batch[i, j] = rhs[
                                i,
                                j // 32 * 32
                                + j % 16 // 2 * 4
                                + j % 2
                                + j % 32 // 16 * 2,
                            ]
                        T.sync_threads()
                        T.wgmma_gemm(
                            lhs_batch[:, 0:32],
                            rhs_batch[:, 0:32],
                            block,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.wgmma_gemm(
                            lhs_batch[:, 32:64],
                            rhs_batch[:, 32:64],
                            p1,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.wgmma_gemm(
                            lhs_batch[:, 64:96],
                            rhs_batch[:, 64:96],
                            p2,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.wgmma_gemm(
                            lhs_batch[:, 96:128],
                            rhs_batch[:, 96:128],
                            p3,
                            transpose_B=True,
                            clear_accum=True,
                        )
                        T.wait_wgmma(0)
                        for b, j in T.Parallel(m, n):
                            block[b, j] = T.call_extern(
                                "float32", "__fadd_rn", block[b, j], p1[b, j]
                            )
                            block[b, j] = T.call_extern(
                                "float32", "__fadd_rn", block[b, j], p2[b, j]
                            )
                            block[b, j] = T.call_extern(
                                "float32", "__fadd_rn", block[b, j], p3[b, j]
                            )
                    elif reduction == 128:
                        for b, j in T.Parallel(m, 128):
                            lhs_wide[b, j] = lhs[b, j]
                        for i, j in T.Parallel(n, 128):
                            rhs_wide[i, j] = rhs[i, j]
                        T.sync_threads()
                        T.gemm(
                            lhs_wide,
                            rhs_wide,
                            block,
                            transpose_B=True,
                            clear_accum=True,
                        )
                    else:
                        T.clear(block)
                        for sub in T.serial(128 // reduction):
                            if packed_widen:
                                lhs_pairs = T.view(lhs, shape=(m, 64), dtype="uint16")
                                rhs_pairs = T.view(rhs, shape=(n, 64), dtype="uint16")
                                a_pairs = T.view(lhs_sub, shape=(m, 16), dtype="uint32")
                                b_pairs = T.view(rhs_sub, shape=(n, 16), dtype="uint32")
                                for b, j in T.Parallel(m, 16):
                                    a_pairs[b, j] = T.call_extern(
                                        "uint32",
                                        "tensor_widen_e4m3x2",
                                        lhs_pairs[b, sub * 16 + j % 8 * 2 + j // 8],
                                    )
                                for i, j in T.Parallel(n, 16):
                                    b_pairs[i, j] = T.call_extern(
                                        "uint32",
                                        "tensor_widen_e4m3x2",
                                        rhs_pairs[i, sub * 16 + j % 8 * 2 + j // 8],
                                    )
                                T.sync_threads()
                            elif reorder:
                                for b, j in T.Parallel(m, reduction):
                                    lhs_sub[b, j] = lhs[
                                        b,
                                        sub * 32
                                        + j % 16 // 2 * 4
                                        + j % 2
                                        + j // 16 * 2,
                                    ]
                                if resident_weights:
                                    T.copy(
                                        w[
                                            expert,
                                            bx * n : bx * n + n,
                                            tile * 128 + sub * 32 : tile * 128
                                            + (sub + 1) * 32,
                                        ],
                                        rhs_sub,
                                    )
                                else:
                                    for i, j in T.Parallel(n, reduction):
                                        rhs_sub[i, j] = rhs[
                                            i,
                                            sub * 32
                                            + j % 16 // 2 * 4
                                            + j % 2
                                            + j // 16 * 2,
                                        ]
                            else:
                                T.copy(
                                    lhs[:, sub * reduction : (sub + 1) * reduction],
                                    lhs_sub,
                                )
                                T.copy(
                                    rhs[:, sub * reduction : (sub + 1) * reduction],
                                    rhs_sub,
                                )
                            T.gemm(
                                lhs_sub,
                                rhs_sub,
                                partial,
                                transpose_B=True,
                                clear_accum=True,
                            )
                            for b, j in T.Parallel(m, n):
                                block[b, j] = T.call_extern(
                                    "float32", "__fadd_rn", block[b, j], partial[b, j]
                                )
                else:
                    T.gemm(lhs, rhs, block, transpose_B=True, clear_accum=True)
                for b, j in T.Parallel(m, n):
                    if compact:
                        product = T.call_extern(
                            "float32", "__fmul_rn", block[b, j], scale[b]
                        )
                        total[b, j] = T.call_extern(
                            "float32",
                            "__fmaf_rn",
                            product,
                            T.cast(
                                scales[expert, (bx * n + j) // 128, tile], "float32"
                            ),
                            total[b, j],
                        )
                    else:
                        total[b, j] += (
                            block[b, j]
                            * scale[b]
                            * T.cast(
                                scales[expert, (bx * n + j) // 128, tile], "float32"
                            )
                        )
            for b, j in T.Parallel(m, n):
                if selected[b] >= 0:
                    if parts > 1:
                        out[selected[b] // top, selected[b] % top, part, bx * n + j] = (
                            total[b, j]
                        )
                    else:
                        out[selected[b] // top, selected[b] % top, bx * n + j] = total[
                            b, j
                        ]

    if compact:

        @kernel
        def compute(
            x: Tensor((rows, top, k) if routed else (rows, k), "uint8"),
            activation_scales: Tensor(
                (rows, top, k // 128) if routed else (rows, k // 128), "float32"
            ),
            w: Tensor((experts, o, k), "float16" if resident_weights else "uint8"),
            scales: Tensor((experts, T.ceildiv(o, 128), k // 128), "bfloat16"),
            counts: Tensor((experts,), "int32"),
            routes: Tensor((experts, rows), "int32"),
            tile_experts: Tensor((max_tiles,), "int32"),
            tile_offsets: Tensor((max_tiles,), "int32"),
        ) -> Tensor((rows, top, parts, o) if parts > 1 else (rows, top, o), "float32"):
            out = output(
                (rows, top, parts, o) if parts > 1 else (rows, top, o),
                "float32",
                position=6,
            )
            with T.Kernel(
                o // n, persistent if persistent else max_tiles, parts, threads=threads
            ) as (bx, block, part):
                if persistent:
                    for iteration in T.serial(T.ceildiv(max_tiles, persistent)):
                        tile = block + iteration * persistent
                        if tile < max_tiles:
                            expert = tile_experts[tile]
                            if expert >= 0:
                                body(
                                    x,
                                    activation_scales,
                                    w,
                                    scales,
                                    counts,
                                    routes,
                                    out,
                                    bx,
                                    expert,
                                    tile_offsets[tile],
                                    part,
                                )
                else:
                    expert = tile_experts[block]
                    if expert >= 0:
                        body(
                            x,
                            activation_scales,
                            w,
                            scales,
                            counts,
                            routes,
                            out,
                            bx,
                            expert,
                            tile_offsets[block],
                            part,
                        )
            return out

        return compute
    if not compact:

        @kernel
        def compute(
            x: Tensor((rows, top, k) if routed else (rows, k), "uint8"),
            activation_scales: Tensor(
                (rows, top, k // 128) if routed else (rows, k // 128), "float32"
            ),
            w: Tensor((experts, o, k), "float16" if resident_weights else "uint8"),
            scales: Tensor((experts, T.ceildiv(o, 128), k // 128), "bfloat16"),
            counts: Tensor((experts,), "int32"),
            routes: Tensor((experts, rows), "int32"),
        ) -> Tensor((rows, top, parts, o) if parts > 1 else (rows, top, o), "float32"):
            out = output(
                (rows, top, parts, o) if parts > 1 else (rows, top, o), "float32"
            )
            with T.Kernel(
                o // n, experts, T.ceildiv(rows, m) * parts, threads=threads
            ) as (bx, expert, group):
                body(
                    x,
                    activation_scales,
                    w,
                    scales,
                    counts,
                    routes,
                    out,
                    bx,
                    expert,
                    group // parts,
                    group % parts,
                )
            return out

        return compute


def decode_moe_groups(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, top = (p["r"], p["top"])

    @kernel
    def compute(
        ids: Tensor((r, top), "int32"),
    ) -> tuple[Tensor((r * top,), "int32"), Tensor((r * top, r), "int32")]:
        experts = output((r * top,), "int32")
        routes = output((r * top, r), "int32")
        with T.Kernel(r * top, threads=32) as group:
            expert = ids[group // top, group % top]
            owner = T.alloc_var("int32")
            owner = group
            for earlier in T.serial(r * top):
                if (earlier < group) & (ids[earlier // top, earlier % top] == expert):
                    owner = T.min(owner, earlier)
            for i in T.Parallel(1):
                experts[group] = T.if_then_else(owner == group, expert, -1)
            for b in T.Parallel(r):
                rank = T.alloc_var("int32")
                rank = -1
                for j in T.serial(top):
                    if ids[b, j] == expert:
                        rank = j
                routes[group, b] = T.if_then_else(owner == group, rank, -1)
        return (experts, routes)

    return compute


def decode_router(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, e, top = (p["r"], p["experts"], p["top"])

    @kernel
    def compute(
        logits: Tensor((r, e), "float32"),
    ) -> tuple[Tensor((r, top), "int32"), Tensor((r, top), "float32")]:
        ids = output((r, top), "int32")
        weights = output((r, top), "float32")
        with T.Kernel(r, threads=256) as b:
            values = T.alloc_fragment((e,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            indices = T.alloc_fragment((e,), "int32")
            first = T.alloc_fragment((1,), "int32")
            selected = T.alloc_shared((top,), "float32")
            for j in T.Parallel(e):
                values[j] = T.cast(T.cast(logits[b, j], "bfloat16"), "float32")
            for rank in T.serial(top):
                T.reduce_max(values, maximum, dim=0)
                for j in T.Parallel(e):
                    indices[j] = T.if_then_else(values[j] == maximum[0], j, e)
                T.reduce_min(indices, first, dim=0)
                for j in T.Parallel(1):
                    ids[b, rank] = first[0]
                    selected[rank] = maximum[0]
                for j in T.Parallel(e):
                    if j == first[0]:
                        values[j] = -T.infinity("float32")
            for j in T.Parallel(top):
                denominator = T.alloc_var("float32")
                denominator = 0
                for rank in T.serial(top):
                    denominator += T.exp(selected[rank] - selected[0])
                weights[b, j] = T.exp(selected[j] - selected[0]) / denominator
        return (ids, weights)

    return compute


def prefill_expert_routes(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])
    rows = slots * chunk
    experts, top = (256, 8)

    @kernel
    def compute(
        ids: Tensor((rows, top), "int32"),
        active: Tensor((rows,), "int32"),
        counts: Tensor((experts,), "int32"),
    ) -> tuple[Tensor((experts,), "int32"), Tensor((experts, rows), "int32")]:
        routes = output((experts, rows), "int32")
        with T.Kernel(experts, threads=256) as expert:
            tx = T.get_thread_binding()
            counter = T.alloc_shared((1,), "int32")
            if tx == 0:
                counter[0] = 0
            T.sync_threads()
            for tile in T.serial(T.ceildiv(rows, 256)):
                row = tile * 256 + tx
                if row < rows:
                    if active[row] != 0:
                        for rank in T.unroll(top):
                            if ids[row, rank] == expert:
                                index = T.atomic_add(counter[0], 1, return_prev=True)
                                routes[expert, index] = row * top + rank
            T.sync_threads()
            if tx == 0:
                counts[expert] = counter[0]
        return (counts, routes)

    return compute


def tile_map(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    rows, m = (p["rows"], p["block_m"])
    experts, top = (256, 8)
    max_tiles = T.ceildiv(rows * top, m) + experts - 1

    @kernel
    def compute(
        counts: Tensor((experts,), "int32"),
    ) -> tuple[Tensor((max_tiles,), "int32"), Tensor((max_tiles,), "int32")]:
        tile_experts = output((max_tiles,), "int32")
        tile_offsets = output((max_tiles,), "int32")
        with T.Kernel(1, threads=256):
            prefix = T.alloc_shared((experts + 1,), "int32")
            tx = T.get_thread_binding()
            if tx == 0:
                prefix[0] = 0
                for expert in T.serial(experts):
                    prefix[expert + 1] = prefix[expert] + T.ceildiv(counts[expert], m)
            for i in T.Parallel(max_tiles):
                tile_experts[i] = -1
                tile_offsets[i] = 0
            T.sync_threads()
            for expert in T.Parallel(experts):
                for tile in T.serial(T.ceildiv(rows, m)):
                    if tile < T.ceildiv(counts[expert], m):
                        index = prefix[expert] + tile
                        tile_experts[index] = expert
                        tile_offsets[index] = tile
        return (tile_experts, tile_offsets)

    return compute


def tiles(spec: KernelSpec, schedule: Mapping[str, object]):
    from dataclasses import replace

    return tile_map(
        replace(spec, parameters={**spec.parameters, "rows": spec.parameters["r"]}),
        schedule,
    )


from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "cuda"))
