"""Shared TileLang computations for moe."""


def expert_swiglu(spec, schedule, *, provider="cuda"):
    """Portable single-token gate/up/SwiGLU with independently bound codecs."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.kernels.reductions import tree_reduce
    from akas.quantization.weights import bind_parts

    p = {**spec.parameters, **schedule}
    depth, columns, experts, top = (p[n] for n in ("k", "o", "experts", "top"))
    if p["r"] != 1 or p.get("input_dtype", "float32") != "float32":
        raise ValueError("expert SwiGLU requires a single FP32 activation row")
    gate = bind_parts(p["weight_parts"], depth, provider=provider)
    up = bind_parts(p["up_weight_parts"], depth, provider=provider)
    if gate.storage.metadata or up.storage.metadata:
        raise ValueError("expert SwiGLU requires single embedded weight pieces")
    gw, uw = gate.storage.values, up.storage.values
    read_gate, read_up = gate.read, up.read
    threads = p.get("threads", 128)
    if threads not in (64, 128, 256):
        raise ValueError("unsupported portable expert SwiGLU thread count")
    reduce = tree_reduce(threads)
    rounded = p.get("round_inputs", False)

    @kernel
    def compute(
        x: Tensor((1, depth), "float32"),
        w: Tensor(gw.shape, gw.dtype),
        w2: Tensor(uw.shape, uw.dtype),
        ids: Tensor((1, top), "int32"),
        counts: Tensor((experts,), "int32"),
    ) -> Tensor((1, top, columns), "float32"):
        out = output((1, top, columns), "float32")
        with T.Kernel(columns, top, threads=threads) as (column, rank):
            tx = T.get_thread_binding()
            expert = ids[0, rank]
            g = T.alloc_var("float32")
            g = 0
            u = T.alloc_var("float32")
            u = 0
            gs = T.alloc_shared((threads,), "float32")
            us = T.alloc_shared((threads,), "float32")
            for tile in T.serial(T.ceildiv(depth, threads)):
                channel = tile * threads + tx
                if channel < depth:
                    a = T.alloc_var("float32")
                    b = T.alloc_var("float32")
                    value = T.alloc_var("float32")
                    a = read_gate((w,), expert * columns + column, channel)
                    b = read_up((w2,), expert * columns + column, channel)
                    value = x[0, channel]
                    if rounded:
                        a = T.cast(T.cast(a, "float16"), "float32")
                        b = T.cast(T.cast(b, "float16"), "float32")
                        value = T.cast(T.cast(value, "float16"), "float32")
                    g += value * a
                    u += value * b
            gs[tx] = g
            us[tx] = u
            T.sync_threads()
            reduce(gs, tx)
            reduce(us, tx)
            if tx == 0:
                out[0, rank, column] = T.if_then_else(
                    counts[expert] > 0, gs[0] / (1 + T.exp(-gs[0])) * us[0], 0
                )
        return out

    return compute


def expert_tile_map(spec, schedule):
    """Build compact expert-row tasks for every supported projection tile width."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    rows, top, experts = p["r"], p["top"], p["experts"]
    capacity = (rows * top + 3) // 4 + experts

    @kernel
    def compute(
        counts: Tensor((experts,), "int32"),
    ) -> tuple[Tensor((5, capacity), "int32"), Tensor((5, capacity), "int32")]:
        tile_experts = output((5, capacity), "int32")
        tile_rows = output((5, capacity), "int32")
        with T.Kernel(5, threads=32) as layout:
            if T.get_thread_binding() == 0:
                cursor = T.alloc_var("int32")
                cursor = 0
                for expert in T.serial(experts):
                    # Keep the numerator nonnegative for empty experts. This
                    # also avoids signed floor-division correction in WGSL.
                    size = T.truncdiv(counts[expert] + (4 << layout) - 1, 4 << layout)
                    for block in T.serial(size):
                        tile_experts[layout, cursor] = expert
                        tile_rows[layout, cursor] = block
                        cursor += 1
                for index in T.serial(capacity):
                    if index >= cursor:
                        tile_experts[layout, index] = -1
                        tile_rows[layout, index] = 0
        return tile_experts, tile_rows

    return compute


def quantized_routes(spec, schedule):
    """Deterministic compact token routes, shared independently of weight format."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    rows, experts, top = (p["r"], p["experts"], p["top"])

    @kernel
    def compute(
        ids: Tensor((rows, top), "int32"), active: Tensor((rows,), "int32")
    ) -> tuple[Tensor((experts,), "int32"), Tensor((experts, rows), "int32")]:
        counts = output((experts,), "int32")
        routes = output((experts, rows), "int32")
        with T.Kernel(experts, threads=32) as expert:
            for row in T.Parallel(rows):
                routes[expert, row] = -1
            T.sync_threads()
            for lane in T.Parallel(1):
                count = T.alloc_var("int32")
                count = 0
                for row in T.serial(rows):
                    if active[row] != 0:
                        for rank in T.serial(top):
                            if ids[row, rank] == expert:
                                routes[expert, count] = row * top + rank
                                count += 1
                counts[expert] = count
        return (counts, routes)

    return compute


def biased_router(spec, schedule):
    """Select with sigmoid scores plus bias; combine with unbiased probabilities."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    rows, experts, top = (p[name] for name in ("r", "experts", "top"))

    @kernel
    def compute(
        logits: Tensor((rows, experts), "float32"), bias: Tensor((experts,), "float32")
    ) -> tuple[Tensor((rows, top), "int32"), Tensor((rows, top), "float32")]:
        ids = output((rows, top), "int32")
        weights = output((rows, top), "float32")
        with T.Kernel(rows, threads=128) as row:
            scores = T.alloc_fragment((experts,), "float32")
            candidates = T.alloc_fragment((experts,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            first = T.alloc_fragment((1,), "float32")
            chosen = T.alloc_shared((top,), "float32")
            for i in T.Parallel(experts):
                scores[i] = 1 / (1 + T.exp(-logits[row, i])) + bias[i]
            for rank in T.serial(top):
                T.reduce_max(scores, maximum, dim=0)
                for i in T.Parallel(experts):
                    candidates[i] = T.if_then_else(
                        scores[i] == maximum[0],
                        -T.cast(i, "float32"),
                        -T.cast(experts, "float32"),
                    )
                T.reduce_max(candidates, first, dim=0)
                for i in T.Parallel(1):
                    ids[row, rank] = T.cast(-first[0], "int32")
                    chosen[rank] = 1 / (
                        1 + T.exp(-logits[row, T.cast(-first[0], "int32")])
                    )
                T.sync_threads()
                for i in T.Parallel(experts):
                    if i == T.cast(-first[0], "int32"):
                        scores[i] = -T.infinity("float32")
            for rank in T.Parallel(top):
                denominator = T.alloc_var("float32")
                denominator = 1e-6 if p.get("normalize", True) else 1
                if p.get("normalize", True):
                    for i in T.serial(top):
                        denominator += chosen[i]
                weights[row, rank] = chosen[rank] * p.get("scale", 1.0) / denominator
        return ids, weights

    return compute


def routed_combine(spec, schedule):
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    rows, top, width = (p[name] for name in ("r", "top", "c"))

    @kernel
    def compute(
        experts: Tensor((rows, top, width), "float32"),
        weights: Tensor((rows, top), "float32"),
        active: Tensor((rows,), "int32"),
    ) -> Tensor((rows, width), "float32"):
        out = output((rows, width), "float32")
        with T.Kernel(rows, T.ceildiv(width, 128), threads=128) as (row, tile):
            for lane in T.Parallel(128):
                column = tile * 128 + lane
                if column < width:
                    value = T.alloc_var("float32")
                    value = 0
                    if active[row] != 0:
                        for rank in T.serial(top):
                            value += experts[row, rank, column] * weights[row, rank]
                    out[row, column] = value
        return out

    return compute


def portable_experts(spec, schedule):
    """Fused encoded expert projection with bounded reduction tiles on WebGPU."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.quantization.weights import bind_parts

    p = {**spec.parameters, **schedule}
    rows, top, experts, columns, depth = (
        p[name] for name in ("r", "top", "experts", "o", "k")
    )
    binding = bind_parts(p["weight_parts"], depth, provider="webgpu")
    storage, read = binding.storage.values, binding.read
    bn, bk = p.get("columns", 4), p.get("block_k", 128)
    routed = p.get("routed_input", False)
    shape = (rows, top, depth) if routed else (rows, depth)

    @T.macro
    def projection(x, w, out, expert, route, bx):
        products = T.alloc_fragment((bn, bk), "float32")
        partial = T.alloc_fragment((bn,), "float32")
        accum = T.alloc_fragment((bn,), "float32")
        T.clear(accum)
        for tile in T.serial(T.ceildiv(depth, bk)):
            for i, j in T.Parallel(bn, bk):
                column, channel = bx * bn + i, tile * bk + j
                source = T.alloc_var("float32")
                if routed:
                    source = x[route // top, route % top, T.min(channel, depth - 1)]
                else:
                    source = x[route // top, T.min(channel, depth - 1)]
                products[i, j] = T.if_then_else(
                    column < columns and channel < depth,
                    source
                    * read(
                        (w,),
                        expert * columns + T.min(column, columns - 1),
                        T.min(channel, depth - 1),
                    ),
                    0,
                )
            T.reduce_sum(products, partial, dim=1)
            for i in T.Parallel(bn):
                accum[i] += partial[i]
        for i in T.Parallel(bn):
            if bx * bn + i < columns:
                out[route // top, route % top, bx * bn + i] = accum[i]

    if rows == 1:

        @kernel
        def compute(
            x: Tensor(shape, "float32"),
            w: Tensor(storage.shape, storage.dtype),
            ids: Tensor((rows, top), "int32"),
            counts: Tensor((experts,), "int32"),
        ) -> Tensor((rows, top, columns), "float32"):
            out = output((rows, top, columns), "float32")
            with T.Kernel(T.ceildiv(columns, bn), top, threads=128) as (bx, rank):
                if counts[ids[0, rank]] > 0:
                    projection(x, w, out, ids[0, rank], rank, bx)
                else:
                    for i in T.Parallel(bn):
                        if bx * bn + i < columns:
                            out[0, rank, bx * bn + i] = 0
            return out
    else:

        @kernel
        def compute(
            x: Tensor(shape, "float32"),
            w: Tensor(storage.shape, storage.dtype),
            counts: Tensor((experts,), "int32"),
            routes: Tensor((experts, rows), "int32"),
        ) -> Tensor((rows, top, columns), "float32"):
            out = output((rows, top, columns), "float32")
            with T.Kernel(T.ceildiv(columns, bn), rows, experts, threads=128) as (
                bx,
                index,
                expert,
            ):
                if index < counts[expert]:
                    projection(x, w, out, expert, routes[expert, index], bx)
            return out

    return compute
