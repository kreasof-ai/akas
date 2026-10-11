"""Operation kernels for moe.cuda.prefill."""


def make_kernel(kind, p):
    import tilelang.language as T

    slots, chunk = p["slots"], p["chunk"]
    rows = slots * chunk
    if kind == "expert_routes":
        experts, top = 256, 8

        @T.prim_func
        def kernel(
            ids: T.Tensor((rows, top), "int32"),
            active: T.Tensor((rows,), "int32"),
            counts: T.Tensor((experts,), "int32"),
            routes: T.Tensor((experts, rows), "int32"),
        ):
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
                                    index = T.atomic_add(
                                        counter[0], 1, return_prev=True
                                    )
                                    routes[expert, index] = row * top + rank
                T.sync_threads()
                if tx == 0:
                    counts[expert] = counter[0]

        return kernel
    raise ValueError("unsupported operation: " + kind)


def expert_kernel(p):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    rows, k, o = p["rows"], p["k"], p["o"]
    n, top, experts = p.get("columns", 64), 8, 256
    m, threads = p.get("block_m", 16), p.get("threads", 128)
    routed = p.get("routed_input", False)
    compact = p.get("compact", False)
    max_tiles = T.ceildiv(rows * top, m) + experts - 1
    arguments = [
        ("x", (rows, top, k) if routed else (rows, k), "uint8"),
        (
            "activation_scales",
            (rows, top, k // 128) if routed else (rows, k // 128),
            "float32",
        ),
        ("w", (experts, o, k), "uint8"),
        ("scales", (experts, T.ceildiv(o, 128), k // 128), "bfloat16"),
        ("counts", (experts,), "int32"),
        ("routes", (experts, rows), "int32"),
        ("out", (rows, top, o), "float32"),
    ]

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
                        # Preserve the control profile's rounded product and
                        # fused accumulation when indirection changes lowering.
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

    @T.macro
    def algorithm(x, activation_scales, w, scales, counts, routes, out):
        with T.Kernel(o // n, experts, T.ceildiv(rows, m), threads=threads) as (
            bx,
            expert,
            tokens,
        ):
            body(
                x, activation_scales, w, scales, counts, routes, out, bx, expert, tokens
            )

    @T.macro
    def compact_algorithm(
        x, activation_scales, w, scales, counts, routes, out, tile_experts, tile_offsets
    ):
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

    if compact:
        arguments += [
            ("tile_experts", (max_tiles,), "int32"),
            ("tile_offsets", (max_tiles,), "int32"),
        ]
    return primitive(arguments, compact_algorithm if compact else algorithm)
