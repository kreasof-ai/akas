"""WebGPU specializations and shared bindings for linear."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def linear(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, o = (p["r"], p["k"], p["o"])
    threads = p.get("threads", 64)
    if threads not in (32, 64, 128, 256):
        raise ValueError("reference projection threads must be 32/64/128/256")
    stride, offset = (p.get("output_stride", o), p.get("output_offset", 0))
    matrix = p.get("matrix_output", False)
    shape = (r, stride) if matrix else (r * stride,)

    @kernel
    def compute(
        x: Tensor((r * k,), "float32"), w: Tensor((o * k,), "float32")
    ) -> Tensor(shape, "float32"):
        out = output(shape, "float32")
        with T.Kernel(T.ceildiv(o, threads), r, threads=threads) as (block, row):
            tx = T.get_thread_binding()
            col = block * threads + tx
            if col < o:
                accum = T.alloc_var("float32")
                accum = 0
                for channel in T.serial(k):
                    accum += x[row * k + channel] * w[col * k + channel]
                if matrix:
                    out[row, offset + col] = accum
                else:
                    out[row * stride + offset + col] = accum
        return out

    return compute


def _packed(p, paired=False, add_residual=False):
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.outer_product import outer_product_matmul
    from tensor.compiler.webgpu.schedules.partitioned import partitioned_matmul
    from tensor.compiler.webgpu.schedules.register_matmul import register_matmul
    from akas.ops.linear.kernel import _epilogue as projection_epilogue
    from akas.ops.linear.kernel import webgpu_abi as projection_abi
    from akas.quantization._shared.rounding import round_half, round_half_branch
    from akas.quantization.weights.ggml.kernel import weight_decoder

    r, k, o, q = (p[n] for n in ("r", "k", "o", "type"))
    parts = 2 if paired else 1
    tm, tn, bk = p.get("tile", (16, 32, 64))
    rounded = round_half_branch() if p.get("branch_staging", False) else round_half()
    if p.get("weight_layout", "raw") == "planar":
        if q != 2:
            raise ValueError("planar dense projection requires Q4_0")
        from akas.quantization.weights.ggml.kernel import planar_reader
        planar = planar_reader((o, k))

        @T.macro
        def decoder(w, row, column):
            return planar((w,), row, column)
    else:
        decoder = weight_decoder(q, k, packed_words=True)
    epilogue = projection_epilogue(paired=paired, add_residual=add_residual)

    @T.macro
    def read_rhs(w, col, channel):
        if q == 1:
            return decoder(w, col, channel)
        else:
            return rounded(decoder(w, col, channel))

    if p.get("schedule") == "outer":
        if p.get("half_accum") and not p["outer"].get("half_accum"):
            raise ValueError("short-half contract requires a half-accumulating outer schedule")
        if q not in (1, 2, 14):
            raise ValueError("outer-product prefill requires F16, Q4_0 or Q6_K weights")
        dtype = p.get("outer_shared_dtype", "float16")
        algorithm = outer_product_matmul(
            r, k, o, rounded, read_rhs, epilogue, parts=parts, dtype=dtype, **p["outer"]
        )
    elif p.get("schedule") == "partitioned":
        if q != 1:
            raise ValueError("searched partitioned profile requires F16 weights")
        config = {
            name: p[name]
            for name in (
                "threads",
                "partitions",
                "unroll",
                "dot_width",
                "owner_axis",
                "k_layout",
            )
        }
        algorithm = partitioned_matmul(
            r,
            k,
            o,
            rounded,
            read_rhs,
            epilogue,
            parts=parts,
            tile_m=tm,
            tile_n=tn,
            explicit_unroll=p.get("explicit_unroll", False),
            **config,
        )
    else:
        if p.get("half_accum"):
            raise ValueError("short-half contract requires an outer schedule")
        algorithm = register_matmul(
            r,
            k,
            o,
            rounded,
            read_rhs,
            epilogue,
            parts=parts,
            tile_m=tm,
            tile_n=tn,
            tile_k=bk,
            threads=p.get("threads", 128),
            pad=p.get("pad", 0),
            lhs_pad=p.get("lhs_pad", 0),
            lhs_transpose=p.get("lhs_transpose", False),
            dot_width=p.get("dot_width", 4 if q == 1 or k >= 2048 else 1),
            unroll=p.get("unroll", False),
            explicit_unroll=p.get("explicit_unroll", False),
        )
    return projection_abi(p, algorithm, paired=paired, add_residual=add_residual)


def _integer(p, paired=False, add_residual=False):
    from tensor.compiler.webgpu.schedules.packed_integer import packed_integer_matmul
    from akas.ops.linear.kernel import _epilogue as projection_epilogue

    r, k, o = (p.get("r", 1), p["k"], p["o"])
    parts = 2 if paired else 1
    from akas.quantization.weights.ggml.webgpu import integer_rhs

    prepacked = p.get("q4_prepacked", False)
    storage, read_word, read_scale = integer_rhs((o, k), prepacked=prepacked)
    count = storage.values.shape[0]

    config = dict(p.get("integer", {}))
    if prepacked:
        config["signed_rhs"] = True
    algorithm = packed_integer_matmul(
        r,
        k,
        o,
        read_word,
        read_scale,
        projection_epilogue(paired=paired, add_residual=add_residual),
        parts=parts,
        **config,
    )
    if parts == 2:

        @kernel
        def compute(
            packed: Tensor(r * k // 2, "uint32"),
            scales: Tensor(r * k // 16, "float32"),
            sums: Tensor(r * k // 16, "int32"),
            w: Tensor(count, "uint32"),
            w2: Tensor(count, "uint32"),
        ) -> Tensor(r * o, "float32"):
            out = output(r * o, "float32")
            algorithm(packed, scales, sums, w, out, w2)
            return out

        return compute

    @kernel
    def compute(
        packed: Tensor(r * k // 2, "uint32"),
        scales: Tensor(r * k // 16, "float32"),
        sums: Tensor(r * k // 16, "int32"),
        w: Tensor(count, "uint32"),
    ) -> Tensor(r * o, "float32"):
        out = output(r * o, "float32")
        algorithm(packed, scales, sums, w, out)
        return out

    return compute


def gemv_packed(p, paired=False, add_residual=False):
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.gemv import streamed_gemv
    from tensor.compiler.webgpu.schedules.registers import registers, shared_buffers
    from akas.ops.linear.kernel import _epilogue as projection_epilogue
    from akas.ops.linear.kernel import webgpu_abi as projection_abi

    k, o, q = (p["k"], p["o"], p["type"])
    parts = 2 if paired else 1
    epilogue = projection_epilogue(paired=paired, add_residual=add_residual)
    if p.get("decode_schedule") == "streamed":
        if q != 1:
            raise ValueError("streamed GEMV requires native F16 weights")
        config = {
            name: p[name]
            for name in (
                "lanes",
                "threads",
                "micro_rows",
                "dot_width",
                "unroll",
                "accumulators",
                "k_layout",
                "shared_input",
            )
        }
        return projection_abi(
            p,
            streamed_gemv(k, o, epilogue, parts=parts, **config),
            paired=paired,
            add_residual=add_residual,
        )
    wide = q == 2 and p.get("sg") and (k % 256 == 0) and (min(k, o) >= 2048)
    lanes, threads, chains = (16 if q == 2 else 32, 128, 1)
    fast_q4 = q == 2 and k % 128 == 0
    if fast_q4:
        lanes, threads, chains = (
            p.get("gemv_lanes", 32 if wide else 16),
            p.get("gemv_threads", 128),
            p.get("gemv_accumulators", 1),
        )
        if (
            lanes not in (8, 16, 32, 64)
            or threads not in (64, 128, 256, 512)
            or threads % lanes
            or (chains not in (1, 4))
            or k % (lanes * 8)
        ):
            raise ValueError("unsupported packed Q4 decode schedule")
    if q == 14:
        threads = p.get("gemv_threads", 128)
        if threads not in (64, 128, 256, 512):
            raise ValueError("invalid Q6 decode workgroup")
    row_count = threads // lanes
    tiles = (
        k // (lanes * 8)
        if fast_q4
        else k // 256
        if q == 14
        else k // 128
        if q in (0, 1) and k % 128 == 0
        else k // 32
    )
    unroll, independent = (p.get("gemv_unroll", 1), p.get("gemv_chains", 1))
    dot_q4, dot_q6 = (
        fast_q4 and p.get("gemv_dot", wide),
        q == 14 and p.get("gemv_q6_dot"),
    )
    from akas.quantization.weights.ggml.webgpu import gemv_terms

    compute = gemv_terms(
        p,
        fast_q4=fast_q4,
        lanes=lanes,
        row_count=row_count,
        dot_q4=dot_q4,
        dot_q6=dot_q6,
    )
    if dot_q4 and chains != 1:
        raise ValueError("packed dot schedule uses one accumulator")
    if unroll != 1 or independent != 1:
        if (
            q not in (2, 14)
            or type(unroll) is not int
            or unroll not in (1, 2, 4, 8, 16)
            or (type(independent) is not int)
            or (independent not in (1, 2, 4, 8))
            or (independent > unroll)
            or (chains != 1)
            or tiles % unroll
        ):
            raise ValueError("invalid packed decode unroll or accumulator chains")
        chains = independent

    def compute_slots(acc, x, weights, chunk, lane, row, bx):
        for slot in range(unroll):
            for part in range(parts):
                compute(
                    {j: acc[part, j] for j in range(chains)},
                    x,
                    weights[part],
                    chunk * unroll + slot,
                    lane,
                    row,
                    bx,
                    slot % independent,
                )

    def combine(acc, totals):
        for part in range(parts):
            if chains == 4:
                combine4(totals[part,][0], *(acc[part, j][0] for j in range(4)))
            else:
                for j in range(chains):
                    add(totals[part,][0], acc[part, j][0])

    def shuffles(totals):
        for stride in (lanes >> i for i in range(1, lanes.bit_length())):
            for part in range(parts):
                shuffle(totals[part,][0], stride)

    def spills(totals, scratch, tx):
        for part in range(parts):
            spill(scratch[part], totals[part,][0], tx)

    def steps(scratch, tx, stride):
        for part in range(parts):
            shared_step(scratch[part], tx, stride)

    def reduction(scratch, tx, lane):
        for stride in (lanes >> i for i in range(1, lanes.bit_length())):
            reduction_step(scratch, tx, lane, stride)

    @T.macro
    def combine4(out: T.Ref, a, b, c, d):
        out = a + b + (c + d)

    @T.macro
    def add(out: T.Ref, value):
        out = out + value

    @T.macro
    def shuffle(acc: T.Ref, stride):
        acc = acc + T.call_extern(
            "float32", "subgroupShuffleXor", acc, T.uint32(stride)
        )

    @T.macro
    def spill(scratch, value, tx):
        scratch[tx] = value

    @T.macro
    def shared_step(scratch, tx, stride):
        scratch[tx] = scratch[tx] + scratch[tx + stride]

    @T.macro
    def reduction_step(scratch, tx, lane, stride):
        if lane < stride:
            steps(scratch, tx, stride)
        T.sync_threads()

    @T.macro
    def gemv(x, w, out, w2=None, residual=None):
        with T.Kernel(T.ceildiv(o, row_count), threads=threads) as bx:
            tx = T.get_thread_binding()
            row = tx // lanes
            lane = tx % lanes
            accum = registers((parts, chains))
            totals = registers((parts,))
            scratch = shared_buffers((threads,), "float32", parts)
            if bx * row_count + row < o:
                for chunk in T.serial(tiles // unroll):
                    compute_slots(accum, x, (w, w2), chunk, lane, row, bx)
            combine(accum, totals)
            if p.get("sg"):
                if T.call_extern("uint32", "tensor_subgroup_size") >= lanes:
                    shuffles(totals)
                    spills(totals, scratch, tx)
                else:
                    spills(totals, scratch, tx)
                    T.sync_threads()
                    reduction(scratch, tx, lane)
            else:
                spills(totals, scratch, tx)
                T.sync_threads()
                reduction(scratch, tx, lane)
            if (lane == 0) & (bx * row_count + row < o):
                epilogue(
                    out,
                    residual,
                    bx * row_count + row,
                    scratch[0][tx],
                    scratch[1][tx] if parts == 2 else 0,
                )

    return projection_abi(p, gemv, paired=paired, add_residual=add_residual)


def vector_decode(spec: KernelSpec, schedule: Mapping[str, object]):
    """Coupled vector and independent-accumulator decode candidates."""
    p = {**spec.parameters, **schedule}
    layouts = (
        (128, 32, 2, 2, True),
        (256, 32, 2, 2, True),
        (128, 16, 4, 4, True),
        (256, 16, 4, 4, True),
        (128, 32, 1, 1, False),
        (128, 16, 1, 1, False),
        (512, 32, 2, 2, True),
        (128, 32, 4, 4, True),
    )
    threads, lanes, unroll, chains, dot = layouts[p.get("decode_layout", 0)]
    tiles = p["k"] // (lanes * 8) if p["type"] == 2 else p["k"] // 256
    while tiles % unroll:
        unroll //= 2
    chains = min(chains, unroll)
    return gemv_packed(
        {**p, "sg": True, "gemv_threads": threads, "gemv_lanes": lanes,
         "gemv_unroll": unroll, "gemv_chains": chains,
         "gemv_dot": dot, "gemv_q6_dot": True,
         "gemv_accumulators": 1 if dot or p["type"] != 2 else 4},
        paired=spec.kind == "ffn", add_residual=spec.kind == "linear_add",
    )


def archived_decode(spec: KernelSpec, schedule: Mapping[str, object]):
    """Replay the RX 6700 XT packed F32 schedules retained at Tensor d5ee508."""
    p = {**spec.parameters, **schedule}
    selected = {
        (2048, 128000, 14): dict(gemv_q6_dot=True, gemv_threads=256,
                               gemv_unroll=4, gemv_chains=1),
        (2048, 6144, 2): dict(gemv_unroll=8, gemv_chains=8),
        (10752, 2048, 2): dict(gemv_lanes=16, gemv_threads=64,
                             gemv_dot=True, gemv_unroll=4),
        (2048, 2048, 2): dict(gemv_unroll=8, gemv_chains=8),
        (2048, 512, 2): dict(gemv_lanes=32, gemv_threads=64,
                            gemv_dot=True, gemv_unroll=4),
    }
    # Paired gate/up retained the subgroup control in the archive.
    config = {} if spec.kind == "ffn" else selected.get((p["k"], p["o"], p["type"]), {})
    return gemv_packed({**p, "sg": True, **config}, paired=spec.kind == "ffn",
                       add_residual=spec.kind == "linear_add")


def q8(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.weights.ggml.webgpu import q8_dot

    p = {**spec.parameters, **schedule}

    k, o = p["k"], p["o"]
    product = q8_dot(k, o)

    @kernel
    def compute(
        x: Tensor((k // 4,), "uint32"),
        scales: Tensor((k // 32,), "float32"),
        sums: Tensor((k // 32,), "int32"),
        w: Tensor((k * o // 32 * 18 // 4,), "uint32"),
    ) -> Tensor((o,), "float32"):
        out = output((o,), "float32")
        product(x, scales, sums, w, out)
        return out

    return compute


def outer_prefill(spec: KernelSpec, schedule: Mapping[str, object]):
    """Staged packed projections with a scalar, resumable search axis."""
    p = {**spec.parameters, **schedule}
    rows = p.get("outer_rows", 32)
    return _packed(
        {**p, "schedule": "outer", "outer": dict(
            tile_m=rows, tile_n=64, tile_k=16, micro_m=rows // 16,
            micro_n=4, threads=256, lhs_layout="km", unroll=4,
            dot_width=2, packed_pairs=True)},
        paired=spec.kind == "ffn",
    )


def blocked_prefill(spec: KernelSpec, schedule: Mapping[str, object]):
    """Explore larger reuse tiles while retaining FP32 accumulation."""
    p = {**spec.parameters, **schedule}
    # M, N, K, private M, private N, threads. Scalar selectors keep
    # resumable search keys hashable while the ownership moves together.
    layouts = (
        (64, 64, 16, 4, 4, 256),
        (64, 64, 32, 4, 4, 256),
        (64, 128, 32, 4, 8, 256),
        (64, 128, 32, 8, 4, 256),
        (64, 64, 32, 4, 8, 128),
        (128, 64, 32, 8, 4, 256),
        (128, 128, 32, 8, 8, 256),
        (32, 128, 32, 4, 4, 256),
        (32, 64, 32, 4, 4, 128),
    )
    m, n, k, mm, mn, threads = layouts[p.get("outer_layout", 0)]
    return _packed(
        {**p, "schedule": "outer", "outer": dict(
            tile_m=m, tile_n=n, tile_k=k, micro_m=mm, micro_n=mn,
            threads=threads, lhs_layout="km", unroll=4,
            dot_width=2, packed_pairs=True)},
        paired=spec.kind == "ffn", add_residual=spec.kind == "linear_add",
    )


def branched_prefill(spec: KernelSpec, schedule: Mapping[str, object]):
    """Bypass subnormal rounding work for normal operands, retaining exact RNE."""
    return blocked_prefill(spec, {**schedule, "branch_staging": True})


def padded_prefill(spec: KernelSpec, schedule: Mapping[str, object]):
    """Replay archived shared padding and grid order with F32 totals."""
    p = {**spec.parameters, **schedule}
    # A packed word of padding was the archived FFN layout winner. Keep a
    # matched unpadded control and separate padding from grid placement.
    pad, order = ((0, "column"), (1, "column"), (1, "row"), (2, "row"))[
        p.get("shared_layout", 0)
    ]
    wide = p["o"] == 6144
    return _packed(
        {**p, "schedule": "outer", "outer": dict(
            tile_m=64, tile_n=128 if wide else 64, tile_k=16,
            micro_m=4, micro_n=8 if wide else 4, threads=256,
            lhs_layout="km", unroll=8, explicit_unroll=True,
            dot_width=2, packed_pairs=True, lhs_pad=pad, rhs_pad=pad,
            group_order=order)},
        paired=spec.kind == "ffn", add_residual=spec.kind == "linear_add",
    )


def half_prefill(spec: KernelSpec, schedule: Mapping[str, object]):
    """Eight adjacent even/odd F16 FMAs followed by a complete-K F32 total."""
    p = {**spec.parameters, **schedule}
    layouts = (
        (32, 64, 4, 4, 128),
        (64, 64, 4, 4, 256),
        (64, 128, 4, 8, 256),
        (32, 128, 4, 8, 128),
        (64, 128, 8, 8, 128),
        (128, 128, 8, 8, 256),
    )
    m, n, mm, mn, threads = layouts[p.get("half_layout", 0)]
    if p["r"] <= 8:
        m, n, mm, mn, threads = 8, 64, 1, 4, 128
    pad = p.get("half_pad", 1)
    if p.get("input_dtype") == "float16" and not p.get("half_native", True):
        raise ValueError("staged F16 input requires the native half-pair kernel")
    if not p.get("half_native", True) and p.get("half_k", 16) != 16:
        raise ValueError("generic short-half staging uses a fixed depth of 16")
    if p.get("half_streamed") and not p.get("half_native", True):
        raise ValueError("streamed half staging requires the native half-pair kernel")
    if p.get("half_native", True):
        return native_half_pairs({**p, "half_tile": (m, n, mm, mn, threads), "half_pad": pad},
                                 paired=spec.kind == "ffn", add_residual=spec.kind == "linear_add")
    primitive = _packed(
        {**p, "schedule": "outer", "outer": dict(
            tile_m=m, tile_n=n, tile_k=16, micro_m=mm, micro_n=mn,
            threads=threads, lhs_layout="km", unroll=8, explicit_unroll=True,
            dot_width=2, packed_pairs=True, half_accum=True,
            lhs_pad=pad, rhs_pad=pad, group_order="row")},
        paired=spec.kind == "ffn", add_residual=spec.kind == "linear_add",
    )
    return primitive


def half_input(spec: KernelSpec, schedule: Mapping[str, object]):
    """Round once before tiled reuse, including ties and half subnormals."""
    import tilelang.language as T
    from akas.quantization._shared.rounding import round_half

    count = spec.parameters["r"] * spec.parameters["k"]
    rounded = round_half()

    @kernel
    def compute(x: Tensor(count, "float32")) -> Tensor(count, "float16"):
        out = output(count, "float16")
        with T.Kernel(T.ceildiv(count, 256), threads=256) as block:
            index = block * 256 + T.get_thread_binding()
            if index < count:
                out[index] = T.cast(rounded(x[index]), "float16")
        return out

    return compute


def native_half_pairs(p, paired=False, add_residual=False):
    """Typed half-pair shared tiles avoid unpack/narrow conversions per FMA."""
    import tilelang.language as T
    from itertools import product
    from tensor.compiler.webgpu.schedules.registers import registers, shared_buffers
    from akas.ops.linear.kernel import _epilogue, webgpu_abi
    from akas.quantization._shared.rounding import round_half
    from akas.quantization.weights.ggml.kernel import weight_decoder

    r, k, o, q = (p[n] for n in ("r", "k", "o", "type"))
    m, n, mm, mn, threads = p["half_tile"]
    pad, parts = p["half_pad"], 2 if paired else 1
    bk = p.get("half_k", 16)
    if bk not in (16, 32, 64) or k % bk:
        raise ValueError("short-half staging depth must divide K and be 16/32/64")
    pairs = bk // 2
    if pairs * 4 * (m + pad + parts * (n + pad)) > 32768:
        raise ValueError("short-half staging exceeds 32 KiB")
    rm, rn = m // mm, n // mn
    rounded, decode = round_half(), weight_decoder(q, k, packed_words=True)
    epilogue = _epilogue(paired=paired, add_residual=add_residual)

    @T.macro
    def input_value(x, index):
        if p.get("input_dtype") == "float16":
            return T.cast(x[index], "float32")
        else:
            return rounded(x[index])

    @T.macro
    def weight(w, row, channel):
        if q == 1:
            return decode(w, row, channel)
        else:
            return rounded(decode(w, row, channel))

    @T.macro
    def stage_a(x, lhs, by, tile, tx):
        for load in T.serial(T.ceildiv(m * pairs, threads)):
            flat = load * threads + tx
            row, pair = flat // pairs, flat % pairs
            if row < m:
                a, b = T.alloc_var("float32"), T.alloc_var("float32")
                a, b = 0, 0
                if (by * m + row < r) & (tile * bk + pair * 2 < k):
                    a = input_value(x, (by * m + row) * k + tile * bk + pair * 2)
                if (by * m + row < r) & (tile * bk + pair * 2 + 1 < k):
                    b = input_value(x, (by * m + row) * k + tile * bk + pair * 2 + 1)
                lhs[pair, row, 0] = T.cast(a, "float16")
                lhs[pair, row, 1] = T.cast(b, "float16")

    @T.macro
    def stage_b(w, rhs, bx, tile, tx):
        for load in T.serial(T.ceildiv(n * pairs, threads)):
            flat = load * threads + tx
            row, pair = flat // pairs, flat % pairs
            if row < n:
                a, b = T.alloc_var("float32"), T.alloc_var("float32")
                a, b = 0, 0
                if (bx * n + row < o) & (tile * bk + pair * 2 < k):
                    a = weight(w, bx * n + row, tile * bk + pair * 2)
                if (bx * n + row < o) & (tile * bk + pair * 2 + 1 < k):
                    b = weight(w, bx * n + row, tile * bk + pair * 2 + 1)
                rhs[pair, row, 0] = T.cast(a, "float16")
                rhs[pair, row, 1] = T.cast(b, "float16")

    def stage_rhs(w, rhs, w2, bx, tile, tx):
        stage_b(w, rhs[0], bx, tile, tx)
        if paired:
            stage_b(w2, rhs[1], bx, tile, tx)

    @T.macro
    def clear(value: T.Ref):
        value = T.call_extern("float16x2", "vec2<f16>", T.cast(0, "float16"))

    @T.macro
    def fma(value: T.Ref, left0, left1, right0, right1):
        value = T.call_extern("float16x2", "fma",
            T.call_extern("float16x2", "vec2<f16>", left0, left1),
            T.call_extern("float16x2", "vec2<f16>", right0, right1), value)

    def multiply(partial, lhs, rhs, kk, mr, nr):
        for part, i, j in product(range(parts), range(mm), range(mn)):
            fma(partial[part, i, j][0], lhs[kk, mr + i * rm, 0], lhs[kk, mr + i * rm, 1],
                rhs[part][kk, nr + j * rn, 0], rhs[part][kk, nr + j * rn, 1])

    @T.macro
    def f32_pair(value: T.Ref, left0, left1, right0, right1):
        value += T.call_extern("float32", "dot",
            T.call_extern("float32x2", "vec2<f32>", T.cast(left0, "float32"), T.cast(left1, "float32")),
            T.call_extern("float32x2", "vec2<f32>", T.cast(right0, "float32"), T.cast(right1, "float32")))

    def multiply_f32(total, lhs, rhs, kk, mr, nr):
        for part, i, j in product(range(parts), range(mm), range(mn)):
            f32_pair(total[part, i, j][0], lhs[kk, mr + i * rm, 0], lhs[kk, mr + i * rm, 1],
                     rhs[part][kk, nr + j * rn, 0], rhs[part][kk, nr + j * rn, 1])

    @T.macro
    def accumulate(total: T.Ref, value):
        total += T.call_extern("float32", "tensor_sum2_f16_vec", value)

    def add_partials(total, partial):
        for key in total:
            accumulate(total[key][0], partial[key][0])

    def clear_partials(partial):
        for value in partial.values():
            clear(value[0])

    @T.macro
    def store(out, residual, gate, up, row, col):
        if (row < r) & (col < o):
            epilogue(out, residual, row * o + col, gate, up)

    def store_all(out, residual, total, by, bx, mr, nr):
        for i, j in product(range(mm), range(mn)):
            store(out, residual, total[0, i, j][0], total[1, i, j][0] if paired else 0,
                  by * m + mr + i * rm, bx * n + nr + j * rn)

    @T.macro
    def matmul(x, w, out, w2=None, residual=None):
        T.func_attr({"tensor.webgpu.loop_unroll": "explicit"})
        with T.Kernel(T.ceildiv(r, m), T.ceildiv(o, n), threads=threads) as (by, bx):
            tx = T.get_thread_binding()
            mr, nr = tx // rn, tx % rn
            lhs = T.alloc_shared((pairs, m + pad, 2), "float16")
            rhs = shared_buffers((pairs, n + pad, 2), "float16", parts)
            total = registers((parts, mm, mn))
            partial = registers((parts, mm, mn), "float16x2", initialize=False)
            if p.get("half_streamed", False):
                for pair in T.serial(k // 2):
                    if pair % pairs == 0:
                        stage_a(x, lhs, by, pair // pairs, tx)
                        stage_rhs(w, rhs, w2, bx, pair // pairs, tx)
                        T.sync_threads()
                    if pair % 8 == 0:
                        clear_partials(partial)
                    multiply(partial, lhs, rhs, pair % pairs, mr, nr)
                    if pair % 8 == 7:
                        add_partials(total, partial)
                    if pair % pairs == pairs - 1:
                        T.sync_threads()
            else:
                for tile in T.serial(T.ceildiv(k, bk)):
                    stage_a(x, lhs, by, tile, tx)
                    stage_rhs(w, rhs, w2, bx, tile, tx)
                    T.sync_threads()
                    if p.get("native_f32", False):
                        for kk in T.unroll(pairs):
                            multiply_f32(total, lhs, rhs, kk, mr, nr)
                    else:
                        for segment in T.unroll(bk // 16):
                            clear_partials(partial)
                            for kk in T.unroll(8):
                                multiply(partial, lhs, rhs, segment * 8 + kk, mr, nr)
                            add_partials(total, partial)
                    T.sync_threads()
            store_all(out, residual, total, by, bx, mr, nr)

    return webgpu_abi(p, matmul, paired=paired, add_residual=add_residual)


def integer_prefill(spec: KernelSpec, schedule: Mapping[str, object]):
    """Coupled reuse/ownership search for the declared two-component ABI."""
    p = {**spec.parameters, **schedule}
    layouts = (
        (16, 32, 2, 2, 128),
        (32, 64, 2, 4, 256),
        (64, 128, 4, 8, 256),
        (32, 128, 4, 8, 128),
        (64, 64, 4, 4, 256),
    )
    m, n, mm, mn, threads = layouts[p.get("integer_layout", 0)]
    return _integer(
        {**p, "integer": {**p.get("integer", {}), "tile_m": m, "tile_n": n,
            "micro_m": mm, "micro_n": mn, "threads": threads,
            "owner_axis": "column", "group_order": "row", "tile_k": p.get("integer_k", 32)}},
        paired=spec.kind == "ffn", add_residual=spec.kind == "linear_add",
    )


def ffn(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.webgpu import _integer as integer_prefill_kernel
    from akas.ops.linear.webgpu import _packed as prefill_kernel
    from akas.ops.linear.webgpu import gemv_packed as decode_kernel

    p = {**spec.parameters, **schedule}
    if p.get("q16"):
        return integer_prefill_kernel(p, paired=True, add_residual=False)
    r = p.get("r", 1)
    return (
        decode_kernel(p, paired=True, add_residual=False)
        if r == 1
        else prefill_kernel(p, paired=True, add_residual=False)
    )


def ffn_q16(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.webgpu import _integer as integer_prefill_kernel

    p = {**spec.parameters, **schedule}
    return integer_prefill_kernel(p, paired=True, add_residual=False)


def kernel_linear(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.webgpu import _integer as integer_prefill_kernel
    from akas.ops.linear.webgpu import _packed as prefill_kernel
    from akas.ops.linear.webgpu import gemv_packed as decode_kernel

    p = {**spec.parameters, **schedule}
    if p.get("q16"):
        return integer_prefill_kernel(p, paired=False, add_residual=False)
    r = p.get("r", 1)
    return (
        decode_kernel(p, paired=False, add_residual=False)
        if r == 1
        else prefill_kernel(p, paired=False, add_residual=False)
    )


def linear_add(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.webgpu import _integer as integer_prefill_kernel
    from akas.ops.linear.webgpu import _packed as prefill_kernel
    from akas.ops.linear.webgpu import gemv_packed as decode_kernel

    p = {**spec.parameters, **schedule}
    r = p.get("r", 1)
    if r != 1:
        raise ValueError("residual projection fusion requires decode rows")
    return (
        decode_kernel(p, paired=False, add_residual=True)
        if r == 1
        else prefill_kernel(p, paired=False, add_residual=True)
    )


def linear_q16(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.linear.webgpu import _integer as integer_prefill_kernel

    p = {**spec.parameters, **schedule}
    return integer_prefill_kernel(p, paired=False, add_residual=False)


def linear_q8(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.kernels.definition import KernelSpec
    from akas.ops.linear.webgpu import q8 as q8_kernel

    p = {**spec.parameters, **schedule}
    r = p.get("r", 1)
    if r != 1:
        raise ValueError(
            "packed Q8 dot requires one row; use quantized_linear for other shapes"
        )
    return q8_kernel(KernelSpec("q8", p), {})


def floating_ffn(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    from akas.compilation.transforms.projection import (
        batch_rows,
        column_tiles,
        output_view,
        projection_algorithm,
    )
    from akas.ops.linear.webgpu import _packed as prefill_kernel
    from akas.ops.linear.webgpu import gemv_packed as decode_kernel

    parameters = {
        **p,
        "type": 1 if p.get("weight_dtype") == "float16" else 0,
        "sg": True,
    }
    if parameters.get("schedule") == "outer":
        parameters.setdefault("outer", {})
    if (
        projection_algorithm(
            p, minimum_groups=p.get("minimum_mma_groups", MINIMUM_MMA_GROUPS)
        )
        == "gemv"
    ):
        kernel = decode_kernel({**parameters, "r": 1}, paired=True, add_residual=False)
        if p["r"] > 1:
            kernel = batch_rows(kernel, p["r"], p["k"], p["o"])
    else:
        kernel = prefill_kernel(parameters, paired=True, add_residual=False)
        if parameters.get("schedule") != "outer":
            kernel = column_tiles(kernel)
    return output_view(kernel, p)


def floating_linear(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    from akas.compilation.transforms.projection import (
        batch_rows,
        column_tiles,
        output_view,
        projection_algorithm,
    )
    from akas.ops.linear.webgpu import _packed as prefill_kernel
    from akas.ops.linear.webgpu import gemv_packed as decode_kernel

    parameters = {
        **p,
        "type": 1 if p.get("weight_dtype") == "float16" else 0,
        "sg": True,
    }
    if parameters.get("schedule") == "outer":
        parameters.setdefault("outer", {})
    if (
        projection_algorithm(
            p, minimum_groups=p.get("minimum_mma_groups", MINIMUM_MMA_GROUPS)
        )
        == "gemv"
    ):
        kernel = decode_kernel({**parameters, "r": 1}, paired=False, add_residual=False)
        if p["r"] > 1:
            kernel = batch_rows(kernel, p["r"], p["k"], p["o"])
    else:
        kernel = prefill_kernel(parameters, paired=False, add_residual=False)
        if parameters.get("schedule") != "outer":
            kernel = column_tiles(kernel)
    return output_view(kernel, p)


MINIMUM_MMA_GROUPS = 512
from .kernel import linear as shared_linear, packed_linear
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))


def quantized(spec, schedule):
    from .kernel import quantized_linear

    return quantized_linear(spec, schedule, provider="webgpu")


def projection(spec, schedule):
    from .kernel import projection

    return projection(spec, schedule, provider="webgpu")


from .kernel import column_projection as column_projection
