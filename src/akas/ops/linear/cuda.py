"""CUDA specializations and shared bindings for linear."""

from akas.kernels.kernel import Tensor, kernel, output
from collections.abc import Mapping
from akas.kernels.definition import KernelSpec


def _mma(p, *, grouped, prequantized):
    import tilelang.language as T

    r, k, o = (p["r"], p["k"], p["o"])
    n = p.get("columns", 64)
    m = p.get("block_m", 16)
    threads = p.get("threads", 128)
    top, e = (p.get("top", 8), p.get("experts", 256))
    routed = p.get("routed_input", False)
    parts = p.get("partitions", 1)
    stages = p.get("stages", 1)
    packed_copy = p.get("packed_copy", False)
    if (
        k % (128 * parts)
        or o % n
        or m not in (m, 32, 64, 128)
        or (grouped and m != 16)
        or (n not in (32, 64, 128))
        or (threads not in (128, 256))
    ):
        raise ValueError("invalid FP8 tensor-core projection schedule")
    output_shape = (
        ((r, top, parts, o) if grouped else (r, parts, o))
        if parts > 1
        else (r, top, o)
        if grouped
        else (r, o)
    )

    @T.macro
    def body(
        x,
        w,
        scales,
        input_scales,
        experts,
        routes,
        out,
        bx,
        group,
        expert,
        rowbase,
        part,
    ):
        lhs = T.alloc_shared((m, 128), "float8_e4m3fn")
        rhs = T.alloc_shared((n, 128), "float8_e4m3fn")
        absolute = T.alloc_fragment((m, 128), "float32")
        maxima = T.alloc_fragment((m,), "float32")
        activation_scales = T.alloc_shared((m,), "float32")
        block = T.alloc_fragment((m, n), "float32")
        total = T.alloc_fragment((m, n), "float32")
        T.clear(total)
        for local_tile in T.Pipelined(k // (128 * parts), num_stages=stages):
            tile = part * (k // (128 * parts)) + local_tile
            if prequantized:
                for b in T.Parallel(m):
                    activation_scales[b] = 1
                    if rowbase + b < r:
                        if grouped:
                            if routes[group, rowbase + b] >= 0:
                                if routed:
                                    activation_scales[b] = input_scales[
                                        rowbase + b, routes[group, rowbase + b], tile
                                    ]
                                else:
                                    activation_scales[b] = input_scales[
                                        rowbase + b, tile
                                    ]
                        else:
                            activation_scales[b] = input_scales[rowbase + b, tile]
                T.sync_threads()
                for b, j in T.Parallel(m, 128):
                    bits = T.alloc_var("uint8")
                    bits = 0
                    if rowbase + b < r:
                        if grouped:
                            if routes[group, rowbase + b] >= 0:
                                if routed:
                                    bits = x[
                                        rowbase + b,
                                        routes[group, rowbase + b],
                                        tile * 128 + j,
                                    ]
                                else:
                                    bits = x[rowbase + b, tile * 128 + j]
                        else:
                            bits = x[rowbase + b, tile * 128 + j]
                    lhs[b, j] = T.reinterpret("float8_e4m3fn", bits)
            else:
                for b, j in T.Parallel(m, 128):
                    if rowbase + b < r:
                        if grouped:
                            if routes[group, rowbase + b] >= 0:
                                if routed:
                                    absolute[b, j] = T.abs(
                                        T.cast(
                                            x[
                                                rowbase + b,
                                                routes[group, rowbase + b],
                                                tile * 128 + j,
                                            ],
                                            "float32",
                                        )
                                    )
                                else:
                                    absolute[b, j] = T.abs(
                                        T.cast(
                                            x[rowbase + b, tile * 128 + j], "float32"
                                        )
                                    )
                            else:
                                absolute[b, j] = 0
                        else:
                            absolute[b, j] = T.abs(
                                T.cast(x[rowbase + b, tile * 128 + j], "float32")
                            )
                    else:
                        absolute[b, j] = 0
                T.reduce_max(absolute, maxima, dim=1)
                for b in T.Parallel(m):
                    activation_scales[b] = T.max(
                        maxima[b], T.float32(1e-12)
                    ) / T.float32(448)
                T.sync_threads()
                for b, j in T.Parallel(m, 128):
                    value = T.alloc_var("float32")
                    value = 0
                    if rowbase + b < r:
                        if grouped:
                            if routes[group, rowbase + b] >= 0:
                                if routed:
                                    value = T.cast(
                                        x[
                                            rowbase + b,
                                            routes[group, rowbase + b],
                                            tile * 128 + j,
                                        ],
                                        "float32",
                                    )
                                else:
                                    value = T.cast(
                                        x[rowbase + b, tile * 128 + j], "float32"
                                    )
                        else:
                            value = T.cast(x[rowbase + b, tile * 128 + j], "float32")
                    bits = T.cast(
                        T.call_extern(
                            "uint32", "tensor_encode_e4m3", value / activation_scales[b]
                        ),
                        "uint8",
                    )
                    lhs[b, j] = T.reinterpret("float8_e4m3fn", bits)
            if packed_copy:
                weights = T.view(w, dtype="float8_e4m3fn")
                if grouped:
                    T.copy(
                        weights[
                            expert, bx * n : bx * n + n, tile * 128 : tile * 128 + 128
                        ],
                        rhs,
                    )
                else:
                    T.copy(
                        weights[bx * n : bx * n + n, tile * 128 : tile * 128 + 128], rhs
                    )
            else:
                for i, j in T.Parallel(n, 128):
                    if grouped:
                        rhs[i, j] = T.reinterpret(
                            "float8_e4m3fn", w[expert, bx * n + i, tile * 128 + j]
                        )
                    else:
                        rhs[i, j] = T.reinterpret(
                            "float8_e4m3fn", w[bx * n + i, tile * 128 + j]
                        )
            T.gemm(lhs, rhs, block, transpose_B=True, clear_accum=True)
            for b, j in T.Parallel(m, n):
                if grouped:
                    total[b, j] += (
                        block[b, j]
                        * activation_scales[b]
                        * T.cast(scales[expert, (bx * n + j) // 128, tile], "float32")
                    )
                else:
                    total[b, j] += (
                        block[b, j]
                        * activation_scales[b]
                        * T.cast(scales[(bx * n + j) // 128, tile], "float32")
                    )
        for b, j in T.Parallel(m, n):
            if rowbase + b < r:
                if grouped:
                    if routes[group, rowbase + b] >= 0:
                        if parts > 1:
                            out[
                                rowbase + b,
                                routes[group, rowbase + b],
                                part,
                                bx * n + j,
                            ] = total[b, j]
                        else:
                            out[rowbase + b, routes[group, rowbase + b], bx * n + j] = (
                                total[b, j]
                            )
                elif parts > 1:
                    out[rowbase + b, part, bx * n + j] = total[b, j]
                else:
                    out[rowbase + b, bx * n + j] = total[b, j]

    @T.macro
    def experts(x, w, scales, expert_ids, routes, out):
        with T.Kernel(o // n, r * top * T.ceildiv(r, m), parts, threads=threads) as (
            bx,
            group_tile,
            part,
        ):
            group = group_tile // T.ceildiv(r, m)
            rowbase = group_tile % T.ceildiv(r, m) * m
            expert = expert_ids[group]
            if expert >= 0:
                body(
                    x,
                    w,
                    scales,
                    None,
                    expert_ids,
                    routes,
                    out,
                    bx,
                    group,
                    expert,
                    rowbase,
                    part,
                )

    if grouped and prequantized:

        @kernel
        def compute(
            x: Tensor((r, top, k) if routed else (r, k), "uint8"),
            w: Tensor((e, o, k), "uint8"),
            scales: Tensor((e, T.ceildiv(o, 128), k // 128), "bfloat16"),
            activation_scales: Tensor(
                (r, top, k // 128) if routed else (r, k // 128), "float32"
            ),
            experts: Tensor((r * top,), "int32"),
            routes: Tensor((r * top, r), "int32"),
        ) -> Tensor(output_shape, "float32"):
            out = output(output_shape, "float32")
            with T.Kernel(
                o // n, r * top * T.ceildiv(r, m), parts, threads=threads
            ) as (bx, group_tile, part):
                group = group_tile // T.ceildiv(r, m)
                rowbase = group_tile % T.ceildiv(r, m) * m
                expert = experts[group]
                if expert >= 0:
                    body(
                        x,
                        w,
                        scales,
                        activation_scales,
                        experts,
                        routes,
                        out,
                        bx,
                        group,
                        expert,
                        rowbase,
                        part,
                    )
            return out

        return compute
    if grouped and (not prequantized):

        @kernel
        def compute(
            x: Tensor((r, top, k) if routed else (r, k), "bfloat16"),
            w: Tensor((e, o, k), "uint8"),
            scales: Tensor((e, T.ceildiv(o, 128), k // 128), "bfloat16"),
            experts: Tensor((r * top,), "int32"),
            routes: Tensor((r * top, r), "int32"),
        ) -> Tensor(output_shape, "float32"):
            out = output(output_shape, "float32")
            with T.Kernel(
                o // n, r * top * T.ceildiv(r, m), parts, threads=threads
            ) as (bx, group_tile, part):
                group = group_tile // T.ceildiv(r, m)
                rowbase = group_tile % T.ceildiv(r, m) * m
                expert = experts[group]
                if expert >= 0:
                    body(
                        x,
                        w,
                        scales,
                        None,
                        experts,
                        routes,
                        out,
                        bx,
                        group,
                        expert,
                        rowbase,
                        part,
                    )
            return out

        return compute
    if not grouped and prequantized:

        @kernel
        def compute(
            x: Tensor((r, k), "uint8"),
            w: Tensor((o, k), "uint8"),
            scales: Tensor((T.ceildiv(o, 128), k // 128), "bfloat16"),
            activation_scales: Tensor((r, k // 128), "float32"),
        ) -> Tensor(output_shape, "float32"):
            out = output(output_shape, "float32")
            with T.Kernel(o // n, T.ceildiv(r, m), parts, threads=threads) as (
                bx,
                by,
                part,
            ):
                body(
                    x,
                    w,
                    scales,
                    activation_scales,
                    None,
                    None,
                    out,
                    bx,
                    0,
                    0,
                    by * m,
                    part,
                )
            return out

        return compute
    if not grouped and (not prequantized):

        @kernel
        def compute(
            x: Tensor((r, k), "bfloat16"),
            w: Tensor((o, k), "uint8"),
            scales: Tensor((T.ceildiv(o, 128), k // 128), "bfloat16"),
        ) -> Tensor(output_shape, "float32"):
            out = output(output_shape, "float32")
            with T.Kernel(o // n, T.ceildiv(r, m), parts, threads=threads) as (
                bx,
                by,
                part,
            ):
                body(x, w, scales, None, None, None, out, bx, 0, 0, by * m, part)
            return out

        return compute


def linear_bf16(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    k, o = (p["k"], p["o"])
    m, n, bk = (p.get("m", 16), p.get("n", 64), p.get("bk", 64))

    @kernel
    def compute(x: Tensor((r, k), "bfloat16"), w: Tensor((o, k), "bfloat16")) -> Tensor(
        (r, o), "float32"
    ):
        out = output((r, o), "float32")
        with T.Kernel(T.ceildiv(o, n), T.ceildiv(r, m), threads=128) as (bx, by):
            a = T.alloc_shared((m, bk), "bfloat16")
            b = T.alloc_shared((n, bk), "bfloat16")
            acc = T.alloc_fragment((m, n), "float32")
            T.clear(acc)
            for tile in T.Pipelined(T.ceildiv(k, bk), num_stages=p.get("stages", 2)):
                for i, j in T.Parallel(m, bk):
                    a[i, j] = T.if_then_else(
                        (by * m + i < r) & (tile * bk + j < k),
                        x[T.min(by * m + i, r - 1), T.min(tile * bk + j, k - 1)],
                        0,
                    )
                for i, j in T.Parallel(n, bk):
                    b[i, j] = T.if_then_else(
                        (bx * n + i < o) & (tile * bk + j < k),
                        w[T.min(bx * n + i, o - 1), T.min(tile * bk + j, k - 1)],
                        0,
                    )
                T.gemm(a, b, acc, transpose_B=True)
            for i, j in T.Parallel(m, n):
                if (by * m + i < r) & (bx * n + j < o):
                    out[by * m + i, bx * n + j] = acc[i, j]
        return out

    return compute


def _packed(p, paired=False, add_residual=False):
    """FP16 tensor-core operands, FP32 accumulation, and paired FFN tiles."""
    import tilelang.language as T
    from akas.ops.linear.kernel import cuda_abi as projection_abi
    from akas.quantization.weights.ggml.cuda import pair_decoder
    from akas.quantization.weights.ggml.kernel import weight_decoder

    r, k, o, q = (p[n] for n in ("r", "k", "o", "type"))
    bm, bn, bk = (p.get("block_m", 32), p.get("block_n", 64), p.get("block_k", 32))
    stages, threads = (p.get("stages", 2), p.get("threads", 128))
    packed = p.get("packed_pairs", False)
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
        elif bx * bn + i < o:
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
                    elif add_residual:
                        out[(by * bm + i) * o + bx * bn + j] = (
                            residual[(by * bm + i) * o + bx * bn + j] + accum[i, j]
                        )
                    else:
                        out[(by * bm + i) * o + bx * bn + j] = accum[i, j]

    return projection_abi(p, matmul, paired=paired, add_residual=add_residual)


def _floating_block_m(p):
    """Keep two row tiles at the transition from GEMV to tensor cores."""
    ceiling = p.get("maximum_gemv_rows", MAXIMUM_GEMV_ROWS)
    return p.get("block_m", 16 if ceiling < p["r"] <= 32 else 32)


def native_bf16(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    return _native_bf16(p)


def fp8_inline(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.cuda import _mma

    p = {**spec.parameters, **schedule}
    return _mma(p, grouped=False, prequantized=False)


def fp8_prequantized(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.cuda import _mma

    p = {**spec.parameters, **schedule}
    return _mma(p, grouped=False, prequantized=True)


def _native_bf16(p):
    import tilelang.language as T

    r, k, o = (p["r"], p["k"], p["o"])
    m, n, depth = (16, 32, 64)

    @kernel
    def compute(x: Tensor((r, k), "bfloat16"), w: Tensor((o, k), "bfloat16")) -> Tensor(
        (r, o), "float32"
    ):
        out = output((r, o), "float32")
        with T.Kernel(T.ceildiv(o, n), T.ceildiv(r, m), threads=128) as (bx, by):
            lhs = T.alloc_shared((m, depth), "bfloat16")
            rhs = T.alloc_shared((n, depth), "bfloat16")
            accum = T.alloc_fragment((m, n), "float32")
            T.clear(accum)
            for tile in T.Pipelined(k // depth, num_stages=2):
                for i, j in T.Parallel(m, depth):
                    lhs[i, j] = T.if_then_else(
                        by * m + i < r, x[T.min(by * m + i, r - 1), tile * depth + j], 0
                    )
                for i, j in T.Parallel(n, depth):
                    rhs[i, j] = T.if_then_else(
                        bx * n + i < o, w[T.min(bx * n + i, o - 1), tile * depth + j], 0
                    )
                T.gemm(lhs, rhs, accum, transpose_B=True)
            for i, j in T.Parallel(m, n):
                if (by * m + i < r) & (bx * n + j < o):
                    out[by * m + i, bx * n + j] = accum[i, j]
        return out

    return compute


def gemv_linear_bf16(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    import tilelang.language as T

    k, o, cols = (p["k"], p["o"], p.get("columns", 16))

    @kernel
    def compute(x: Tensor((r, k), "bfloat16"), w: Tensor((o, k), "bfloat16")) -> Tensor(
        (r, o), "float32"
    ):
        out = output((r, o), "float32")
        with T.Kernel(T.ceildiv(o, cols), r, threads=256) as (block, row):
            sums = T.alloc_fragment((cols, 32), "float32")
            total = T.alloc_fragment((cols,), "float32")
            T.clear(sums)
            for tile in T.serial(T.ceildiv(k, 32)):
                for i, j in T.Parallel(cols, 32):
                    if (block * cols + i < o) & (tile * 32 + j < k):
                        sums[i, j] = T.ieee_fmaf(
                            T.cast(x[row, tile * 32 + j], "float32"),
                            T.cast(w[block * cols + i, tile * 32 + j], "float32"),
                            sums[i, j],
                        )
            T.reduce_sum(sums, total, dim=1)
            for i in T.Parallel(cols):
                if block * cols + i < o:
                    out[row, block * cols + i] = total[i]
        return out

    return compute


def gemv_native_bf16(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, o = (p["r"], p["k"], p["o"])

    @kernel
    def compute(x: Tensor((r, k), "bfloat16"), w: Tensor((o, k), "bfloat16")) -> Tensor(
        (r, o), "float32"
    ):
        out = output((r, o), "float32")
        with T.Kernel(T.ceildiv(o, 4), threads=128) as bx:
            accum = T.alloc_fragment((r, 4, 32), "float32")
            total = T.alloc_fragment((r, 4), "float32")
            T.clear(accum)
            for tile in T.serial(k // 32):
                for b, n, lane in T.Parallel(r, 4, 32):
                    if bx * 4 + n < o:
                        accum[b, n, lane] = T.ieee_fmaf(
                            T.cast(x[b, tile * 32 + lane], "float32"),
                            T.cast(w[bx * 4 + n, tile * 32 + lane], "float32"),
                            accum[b, n, lane],
                        )
            T.reduce_sum(accum, total, dim=2)
            for b, n in T.Parallel(r, 4):
                if bx * 4 + n < o:
                    out[b, bx * 4 + n] = total[b, n]
        return out

    return compute


def native_fp8(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, o = (p["r"], p["k"], p["o"])
    columns, threads = (p.get("columns", 4), p.get("threads", 128))
    if (
        any((type(v) is not int or v <= 0 for v in (r, k, o, columns, threads)))
        or k % 128
        or o % columns
        or threads % 32
        or (threads > 1024)
        or (r > 8)
        or columns * 32 % threads
    ):
        raise ValueError("invalid native FP8 GEMV schedule")

    @kernel
    def compute(
        x: Tensor((r, k), "bfloat16"),
        w: Tensor((o, k), "uint8"),
        scales: Tensor((T.ceildiv(o, 128), k // 128), "bfloat16"),
    ) -> Tensor((r, o), "float32"):
        out = output((r, o), "float32")
        with T.Kernel(o // columns, threads=threads) as bx:
            accum = T.alloc_fragment((r, columns, 32), "float32")
            total = T.alloc_fragment((r, columns), "float32")
            coefficient = T.alloc_fragment((columns, 32), "float32")
            activation = T.alloc_shared((r, 128), "float32")
            absolute = T.alloc_fragment((r, 128), "float32")
            maximum = T.alloc_fragment((r,), "float32")
            T.clear(accum)
            for tile in T.serial(k // 128):
                for b, j in T.Parallel(r, 128):
                    absolute[b, j] = T.abs(T.cast(x[b, tile * 128 + j], "float32"))
                T.reduce_max(absolute, maximum, dim=1)
                for b, j in T.Parallel(r, 128):
                    scale = T.max(maximum[b], T.float32(1e-12)) / T.float32(448)
                    encoded = T.call_extern(
                        "uint32",
                        "tensor_encode_e4m3",
                        T.cast(x[b, tile * 128 + j], "float32") / scale,
                    )
                    activation[b, j] = (
                        T.call_extern("float32", "tensor_decode_e4m3", encoded) * scale
                    )
                for part in T.unroll(4):
                    for n, lane in T.Parallel(columns, 32):
                        coefficient[n, lane] = T.call_extern(
                            "float32",
                            "tensor_decode_e4m3",
                            T.cast(
                                w[bx * columns + n, tile * 128 + part * 32 + lane],
                                "uint32",
                            ),
                        )
                        coefficient[n, lane] *= T.cast(
                            scales[(bx * columns + n) // 128, tile], "float32"
                        )
                    for b, n, lane in T.Parallel(r, columns, 32):
                        accum[b, n, lane] = T.ieee_fmaf(
                            T.cast(activation[b, part * 32 + lane], "float32"),
                            coefficient[n, lane],
                            accum[b, n, lane],
                        )
            T.reduce_sum(accum, total, dim=2)
            for b, n in T.Parallel(r, columns):
                out[b, bx * columns + n] = total[b, n]
        return out

    return compute


def gemv_packed(p, paired=False, add_residual=False):
    import tilelang.language as T
    from akas.quantization.weights.ggml.definition import ENCODINGS as TYPES
    from akas.ops.linear.kernel import cuda_abi as projection_abi
    from akas.ops.linear.cuda import _fused as fused_kernel
    from akas.quantization.weights.ggml.cuda import packed_dot

    k, o, q = (p["k"], p["o"], p["type"])
    if q not in (0, 1, 2, 12, 14) or k % 256:
        return fused_kernel(p, paired=paired, add_residual=add_residual)
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
                elif add_residual:
                    out[row] = residual[row] + total
                else:
                    out[row] = total

    return projection_abi(p, gemv, paired=paired, add_residual=add_residual)


def _fused(p, paired=False, add_residual=False):
    import tilelang.language as T
    from akas.kernels.definition import KernelSpec
    from akas.ops.linear.kernel import cuda_abi as projection_abi
    from akas.ops.linear.cuda import _packed as prefill_kernel
    from akas.ops.linear.kernel import packed_linear as _factory_packed_linear
    from akas.quantization.weights.ggml.kernel import weight_decoder

    if not (paired or add_residual):
        return _factory_packed_linear(KernelSpec("linear", p), {})
    if p["r"] > 1:
        return prefill_kernel(
            {
                **p,
                "block_m": 32,
                "block_n": 64,
                "block_k": 32,
                "stages": 2,
                "threads": 128,
                "packed_pairs": False,
            },
            paired=paired,
            add_residual=add_residual,
        )
    k, o, q = (p["k"], p["o"], p["type"])
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

    return projection_abi(p, gemv, paired=paired, add_residual=add_residual)


def full_depth_bf16(spec: KernelSpec, schedule: Mapping[str, object]):
    """Parallel full-depth reductions for the small unquantized projections."""
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, o = (p["r"], p["k"], p["o"])

    @kernel
    def compute(x: Tensor((r, k), "bfloat16"), w: Tensor((o, k), "bfloat16")) -> Tensor(
        (r, o), "float32"
    ):
        out = output((r, o), "float32")
        with T.Kernel(r, o, threads=256) as (row, column):
            products = T.alloc_fragment((k,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(k):
                products[j] = T.cast(x[row, j], "float32") * T.cast(
                    w[column, j], "float32"
                )
            T.reduce_sum(products, total, dim=0)
            for j in T.Parallel(1):
                out[row, column] = total[0]
        return out

    return compute


def hopper_gemm_partial(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, o = (p["r"], p["k"], p["o"])
    m = p.get("block_m", 64)
    n = p.get("columns", 128)
    parts = p.get("partitions", 1)
    threads = p.get("threads", 256)
    stages = p.get("stages", 1)
    packed = p.get("packed_gather", True)
    reduction = p.get("mma_reduction", 128)
    reorder = p.get("mma_reorder", False)
    asynchronous = p.get("async_mma", False)
    packed_widen = p.get("packed_widen", False)
    if packed_widen and (not reorder or asynchronous):
        raise ValueError("packed widening requires synchronous paired MMA")
    if asynchronous and (not (reorder and reduction == 32)):
        raise ValueError("async MMA requires paired K32")
    if reorder and reduction != 32:
        raise ValueError("paired operands require K32")
    if reduction not in (32, 128):
        raise ValueError("invalid Hopper MMA reduction")
    if k % (128 * parts) or o % n:
        raise ValueError("invalid Hopper dense geometry")
    shape = (r, parts, o) if parts > 1 else (r, o)

    @kernel
    def compute(
        x: Tensor((r, k), "uint8"),
        w: Tensor((o, k), "uint8"),
        scales: Tensor((T.ceildiv(o, 128), k // 128), "bfloat16"),
        activation_scales: Tensor((r, k // 128), "float32"),
    ) -> Tensor(shape, "float32"):
        out = output(shape, "float32")
        with T.Kernel(o // n, T.ceildiv(r, m), parts, threads=threads) as (
            bx,
            by,
            part,
        ):
            if packed_widen:
                from akas.quantization.weights.fp8.cuda import CUDA_SOURCE

                T.import_source(CUDA_SOURCE)
            lhs = T.alloc_shared((m, 128), "float8_e4m3fn")
            rhs = T.alloc_shared((n, 128), "float8_e4m3fn")
            if asynchronous:
                a_batch = T.alloc_shared((m, 128), "float16")
                b_batch = T.alloc_shared((n, 128), "float16")
                p1 = T.alloc_fragment((m, n), "float32")
                p2 = T.alloc_fragment((m, n), "float32")
                p3 = T.alloc_fragment((m, n), "float32")
            if reduction == 128:
                a = T.alloc_shared((m, 128), "bfloat16")
                b = T.alloc_shared((n, 128), "bfloat16")
            if reduction < 128:
                a_sub = T.alloc_shared((m, reduction), "float16")
                b_sub = T.alloc_shared((n, reduction), "float16")
                partial = T.alloc_fragment((m, n), "float32")
            row_scale = T.alloc_shared((m,), "float32")
            block = T.alloc_fragment((m, n), "float32")
            total = T.alloc_fragment((m, n), "float32")
            T.clear(total)
            words = T.view(x, shape=(r, k // 4), dtype="uint32")
            destination = T.view(lhs, shape=(m, 32), dtype="uint32")
            weights = T.view(w, dtype="float8_e4m3fn")
            for local in T.Pipelined(k // (128 * parts), num_stages=stages):
                tile = part * (k // (128 * parts)) + local
                if packed:
                    for i, j in T.Parallel(m, 32):
                        bits = T.if_then_else(
                            by * m + i < r,
                            words[T.min(by * m + i, r - 1), tile * 32 + j],
                            0,
                        )
                        if stages > 1:
                            for byte in T.unroll(4):
                                lhs[i, j * 4 + byte] = T.reinterpret(
                                    "float8_e4m3fn",
                                    T.cast(bits >> byte * 8 & 255, "uint8"),
                                )
                        else:
                            destination[i, j] = bits
                else:
                    activations = T.view(x, dtype="float8_e4m3fn")
                    T.copy(
                        activations[by * m : by * m + m, tile * 128 : tile * 128 + 128],
                        lhs,
                    )
                for i in T.Parallel(m):
                    row_scale[i] = T.if_then_else(
                        by * m + i < r,
                        activation_scales[T.min(by * m + i, r - 1), tile],
                        1,
                    )
                T.copy(weights[bx * n : bx * n + n, tile * 128 : tile * 128 + 128], rhs)
                T.sync_threads()
                if asynchronous:
                    for i, j in T.Parallel(m, 128):
                        a_batch[i, j] = lhs[
                            i, j // 32 * 32 + j % 16 // 2 * 4 + j % 2 + j % 32 // 16 * 2
                        ]
                    for i, j in T.Parallel(n, 128):
                        b_batch[i, j] = rhs[
                            i, j // 32 * 32 + j % 16 // 2 * 4 + j % 2 + j % 32 // 16 * 2
                        ]
                    T.sync_threads()
                    T.wgmma_gemm(
                        a_batch[:, 0:32],
                        b_batch[:, 0:32],
                        block,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.wgmma_gemm(
                        a_batch[:, 32:64],
                        b_batch[:, 32:64],
                        p1,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.wgmma_gemm(
                        a_batch[:, 64:96],
                        b_batch[:, 64:96],
                        p2,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.wgmma_gemm(
                        a_batch[:, 96:128],
                        b_batch[:, 96:128],
                        p3,
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.wait_wgmma(0)
                    for i, j in T.Parallel(m, n):
                        block[i, j] = T.call_extern(
                            "float32", "__fadd_rn", block[i, j], p1[i, j]
                        )
                        block[i, j] = T.call_extern(
                            "float32", "__fadd_rn", block[i, j], p2[i, j]
                        )
                        block[i, j] = T.call_extern(
                            "float32", "__fadd_rn", block[i, j], p3[i, j]
                        )
                elif reduction == 128:
                    for i, j in T.Parallel(m, 128):
                        a[i, j] = lhs[i, j]
                    for i, j in T.Parallel(n, 128):
                        b[i, j] = rhs[i, j]
                    T.sync_threads()
                    T.gemm(a, b, block, transpose_B=True, clear_accum=True)
                else:
                    T.clear(block)
                    for sub in T.serial(128 // reduction):
                        if packed_widen:
                            lhs_pairs = T.view(lhs, shape=(m, 64), dtype="uint16")
                            rhs_pairs = T.view(rhs, shape=(n, 64), dtype="uint16")
                            a_pairs = T.view(a_sub, shape=(m, 16), dtype="uint32")
                            b_pairs = T.view(b_sub, shape=(n, 16), dtype="uint32")
                            for i, j in T.Parallel(m, 16):
                                a_pairs[i, j] = T.call_extern(
                                    "uint32",
                                    "tensor_widen_e4m3x2",
                                    lhs_pairs[i, sub * 16 + j % 8 * 2 + j // 8],
                                )
                            for i, j in T.Parallel(n, 16):
                                b_pairs[i, j] = T.call_extern(
                                    "uint32",
                                    "tensor_widen_e4m3x2",
                                    rhs_pairs[i, sub * 16 + j % 8 * 2 + j // 8],
                                )
                            T.sync_threads()
                        elif reorder:
                            for i, j in T.Parallel(m, reduction):
                                a_sub[i, j] = lhs[
                                    i, sub * 32 + j % 16 // 2 * 4 + j % 2 + j // 16 * 2
                                ]
                            for i, j in T.Parallel(n, reduction):
                                b_sub[i, j] = rhs[
                                    i, sub * 32 + j % 16 // 2 * 4 + j % 2 + j // 16 * 2
                                ]
                        else:
                            T.copy(
                                lhs[:, sub * reduction : (sub + 1) * reduction], a_sub
                            )
                            T.copy(
                                rhs[:, sub * reduction : (sub + 1) * reduction], b_sub
                            )
                        T.gemm(
                            a_sub, b_sub, partial, transpose_B=True, clear_accum=True
                        )
                        for i, j in T.Parallel(m, n):
                            block[i, j] = T.call_extern(
                                "float32", "__fadd_rn", block[i, j], partial[i, j]
                            )
                for i, j in T.Parallel(m, n):
                    product = T.call_extern(
                        "float32", "__fmul_rn", block[i, j], row_scale[i]
                    )
                    total[i, j] = T.call_extern(
                        "float32",
                        "__fmaf_rn",
                        product,
                        T.cast(scales[(bx * n + j) // 128, tile], "float32"),
                        total[i, j],
                    )
            for i, j in T.Parallel(m, n):
                if by * m + i < r:
                    if parts > 1:
                        out[by * m + i, part, bx * n + j] = total[i, j]
                    else:
                        out[by * m + i, bx * n + j] = total[i, j]
        return out

    return compute


def ffn(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.cuda import _packed as prefill_kernel
    from akas.ops.linear.cuda import _fused as fused_kernel
    from akas.ops.linear.cuda import gemv_packed as gemv_kernel

    p = {**spec.parameters, **schedule}
    if p["r"] == 1:
        return gemv_kernel(p, paired=True, add_residual=False)
    return (
        prefill_kernel(p, paired=True, add_residual=False)
        if "block_m" in p
        else fused_kernel(p, paired=True, add_residual=False)
    )


def linear(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.cuda import _packed as prefill_kernel
    from akas.ops.linear.cuda import _fused as fused_kernel
    from akas.ops.linear.cuda import gemv_packed as gemv_kernel

    p = {**spec.parameters, **schedule}
    if p["r"] == 1:
        return gemv_kernel(p, paired=False, add_residual=False)
    return (
        prefill_kernel(p, paired=False, add_residual=False)
        if "block_m" in p
        else fused_kernel(p, paired=False, add_residual=False)
    )


def linear_add(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.cuda import _packed as prefill_kernel
    from akas.ops.linear.cuda import _fused as fused_kernel
    from akas.ops.linear.cuda import gemv_packed as gemv_kernel

    p = {**spec.parameters, **schedule}
    if p["r"] == 1:
        return gemv_kernel(p, paired=False, add_residual=True)
    return (
        prefill_kernel(p, paired=False, add_residual=True)
        if "block_m" in p
        else fused_kernel(p, paired=False, add_residual=True)
    )


def floating_ffn(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.cuda import _floating_block_m as mma_block_m

    p = {**spec.parameters, **schedule}
    from akas.compilation.transforms.projection import (
        batch_rows,
        column_tiles,
        output_view,
        projection_algorithm,
    )
    from akas.ops.linear.cuda import _packed as prefill_kernel
    from akas.ops.linear.cuda import gemv_packed as gemv_kernel

    parameters = {**p, "type": 1 if p.get("weight_dtype") == "float16" else 0}
    if (
        projection_algorithm(
            {**p, "maximum_gemv_rows": p.get("maximum_gemv_rows", MAXIMUM_GEMV_ROWS)},
            minimum_groups=p.get("minimum_mma_groups", MINIMUM_MMA_GROUPS),
            default_tile=(32, 64, 32),
        )
        == "gemv"
    ):
        kernel = gemv_kernel({**parameters, "r": 1}, paired=True, add_residual=False)
        if p["r"] > 1:
            kernel = batch_rows(kernel, p["r"], p["k"], p["o"])
    else:
        if p["k"] % p.get("block_k", 32):
            raise ValueError("projection reduction width must be divisible by block_k")
        kernel = column_tiles(
            prefill_kernel(
                {**parameters, "block_m": mma_block_m(p)},
                paired=True,
                add_residual=False,
            )
        )
    return output_view(kernel, p)


def floating_linear(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.cuda import _floating_block_m as mma_block_m

    p = {**spec.parameters, **schedule}
    from akas.compilation.transforms.projection import (
        batch_rows,
        column_tiles,
        output_view,
        projection_algorithm,
    )
    from akas.ops.linear.cuda import _packed as prefill_kernel
    from akas.ops.linear.cuda import gemv_packed as gemv_kernel

    parameters = {**p, "type": 1 if p.get("weight_dtype") == "float16" else 0}
    if (
        projection_algorithm(
            {**p, "maximum_gemv_rows": p.get("maximum_gemv_rows", MAXIMUM_GEMV_ROWS)},
            minimum_groups=p.get("minimum_mma_groups", MINIMUM_MMA_GROUPS),
            default_tile=(32, 64, 32),
        )
        == "gemv"
    ):
        kernel = gemv_kernel({**parameters, "r": 1}, paired=False, add_residual=False)
        if p["r"] > 1:
            kernel = batch_rows(kernel, p["r"], p["k"], p["o"])
    else:
        if p["k"] % p.get("block_k", 32):
            raise ValueError("projection reduction width must be divisible by block_k")
        kernel = column_tiles(
            prefill_kernel(
                {**parameters, "block_m": mma_block_m(p)},
                paired=False,
                add_residual=False,
            )
        )
    return output_view(kernel, p)


def split(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, o, parts = (p[n] for n in ("r", "k", "o", "parts"))

    @kernel
    def compute(x: Tensor((r, parts, o), "float32")) -> Tensor((r, o), "float32"):
        out = output((r, o), "float32")
        with T.Kernel(r, T.ceildiv(o, 256), threads=256) as (row, tile):
            for j in T.Parallel(256):
                col = tile * 256 + j
                if col < o:
                    value = T.alloc_var("float32")
                    value = 0
                    for part in T.serial(parts):
                        value += x[row, part, col]
                    out[row, col] = value
        return out

    return compute


def fp8(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, o, parts = (p["r"], p["o"], p["partitions"])
    top = p.get("top")
    rows = r * (top or 1)

    @kernel
    def compute(
        partial: Tensor((r, top, parts, o) if top else (r, parts, o), "float32"),
    ) -> Tensor((r, top, o) if top else (r, o), "float32"):
        out = output((r, top, o) if top else (r, o), "float32")
        with T.Kernel(rows, T.ceildiv(o, 256), threads=256) as (b, tile):
            for j in T.Parallel(256):
                col = tile * 256 + j
                if col < o:
                    total = T.alloc_var("float32")
                    total = 0
                    for part in T.serial(parts):
                        if top:
                            total += partial[b // top, b % top, part, col]
                        else:
                            total += partial[b, part, col]
                    if top:
                        out[b // top, b % top, col] = total
                    else:
                        out[b, col] = total
        return out

    return compute


def partial(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, o, parts = (p[n] for n in ("r", "k", "o", "parts"))
    m, n, bk = (p["m"], p["n"], p["bk"])
    tiles = T.ceildiv(T.ceildiv(k, bk), parts)

    @kernel
    def compute(x: Tensor((r, k), "bfloat16"), w: Tensor((o, k), "bfloat16")) -> Tensor(
        (r, parts, o), "float32"
    ):
        out = output((r, parts, o), "float32")
        with T.Kernel(
            T.ceildiv(r, m), T.ceildiv(o, n), parts, threads=p.get("threads", 128)
        ) as (by, bx, part):
            a = T.alloc_shared((m, bk), "bfloat16")
            b = T.alloc_shared((n, bk), "bfloat16")
            acc = T.alloc_fragment((m, n), "float32")
            T.clear(acc)
            for local in T.Pipelined(tiles, num_stages=2):
                tile = part * tiles + local
                for i, j in T.Parallel(m, bk):
                    a[i, j] = T.if_then_else(
                        (by * m + i < r) & (tile * bk + j < k),
                        x[T.min(by * m + i, r - 1), T.min(tile * bk + j, k - 1)],
                        0,
                    )
                for i, j in T.Parallel(n, bk):
                    b[i, j] = T.if_then_else(
                        (bx * n + i < o) & (tile * bk + j < k),
                        w[T.min(bx * n + i, o - 1), T.min(tile * bk + j, k - 1)],
                        0,
                    )
                T.gemm(a, b, acc, transpose_B=True)
            for i, j in T.Parallel(m, n):
                if (by * m + i < r) & (bx * n + j < o):
                    out[by * m + i, part, bx * n + j] = acc[i, j]
        return out

    return compute


MINIMUM_MMA_GROUPS = 32
MAXIMUM_GEMV_ROWS = 31
MINIMUM_MMA_GROUPS = 32
MAXIMUM_GEMV_ROWS = 31
from .kernel import linear as shared_linear, packed_linear
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "cuda"))


def quantized(spec, schedule):
    from .kernel import quantized_linear

    return quantized_linear(spec, schedule, provider="cuda")


def projection(spec, schedule):
    from .kernel import projection

    return projection(spec, schedule, provider="cuda")
