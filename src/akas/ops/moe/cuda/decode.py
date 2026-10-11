"""Operation kernels for moe.cuda.decode."""


def make_kernel(kind, p):
    import tilelang.language as T

    if kind == "moe_groups":
        r, top = p["r"], p["top"]

        @T.prim_func
        def kernel(
            ids: T.Tensor((r, top), "int32"),
            experts: T.Tensor((r * top,), "int32"),
            routes: T.Tensor((r * top, r), "int32"),
        ):
            with T.Kernel(r * top, threads=32) as group:
                expert = ids[group // top, group % top]
                owner = T.alloc_var("int32")
                owner = group
                for earlier in T.serial(r * top):
                    if (earlier < group) & (
                        ids[earlier // top, earlier % top] == expert
                    ):
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

        return kernel
    import tilelang.language as T

    if kind == "fp8_experts":
        r, k, o, top, e = (p[n] for n in ("r", "k", "o", "top", "experts"))
        columns, threads = p.get("columns", 4), p.get("threads", 128)
        routed = p.get("routed_input", False)
        from tensor.compiler.frontend.source import primitive

        arguments = [
            ("x", (r, top, k) if routed else (r, k), "bfloat16"),
            ("w", (e, o, k), "uint8"),
            ("scales", (e, T.ceildiv(o, 128), k // 128), "bfloat16"),
            ("experts", (r * top,), "int32"),
            ("routes", (r * top, r), "int32"),
            ("out", (r, top, o), "float32"),
        ]

        @T.macro
        def algorithm(x, w, scales, experts, routes, out):
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
                                scale = T.max(maximum[b], T.float32(1e-12)) / T.float32(
                                    448
                                )
                                if routed:
                                    value = T.cast(
                                        x[b, routes[group, b], tile * 128 + j],
                                        "float32",
                                    )
                                else:
                                    value = T.cast(x[b, tile * 128 + j], "float32")
                                encoded = T.call_extern(
                                    "uint32", "tensor_encode_e4m3", value / scale
                                )
                                activation[b, j] = (
                                    T.call_extern(
                                        "float32", "tensor_decode_e4m3", encoded
                                    )
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

        return primitive(arguments, algorithm)
    import tilelang.language as T

    if kind == "router":
        r, e, top = p["r"], p["experts"], p["top"]

        @T.prim_func
        def kernel(
            logits: T.Tensor((r, e), "float32"),
            ids: T.Tensor((r, top), "int32"),
            weights: T.Tensor((r, top), "float32"),
        ):
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

        return kernel
    import tilelang.language as T

    if kind == "moe_combine":
        r, top, c = p["r"], p["top"], p["c"]

        @T.prim_func
        def kernel(
            experts: T.Tensor((r, top, c), "float32"),
            weights: T.Tensor((r, top), "float32"),
            shared: T.Tensor((r, c), "float32"),
            gate: T.Tensor((r, 1), "float32"),
            out: T.Tensor((r, c), "float32"),
        ):
            with T.Kernel(r, T.ceildiv(c, 256), threads=256) as (b, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        total = T.alloc_var("float32")
                        total = 0
                        for rank in T.serial(top):
                            total += (
                                T.cast(
                                    T.cast(experts[b, rank, col], "bfloat16"), "float32"
                                )
                                * weights[b, rank]
                            )
                        shared_value = T.cast(
                            T.cast(shared[b, col], "bfloat16"), "float32"
                        )
                        shared_gate = T.cast(T.cast(gate[b, 0], "bfloat16"), "float32")
                        out[b, col] = total + shared_value / (1 + T.exp(-shared_gate))

        return kernel
    raise ValueError("unsupported operation: " + kind)
