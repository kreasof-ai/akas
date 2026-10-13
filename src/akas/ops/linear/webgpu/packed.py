"""TileLang DSL for packed portable and standard-subgroup LFM2 schedules."""

from akas.quantization.weights.ggml import weight_decoder
from akas.quantization.weights.ggml import weight_storage
from akas.ops.quantization.webgpu.storage import round_half
from akas.ops.quantization.webgpu.storage import packed_loads
from akas.ops.reductions.webgpu import segmented_reduce


def projection_epilogue(kind):
    import tilelang.language as T

    @T.macro
    def store(out, residual, index, gate, up):
        if kind in ("ffn", "ffn_q16"):
            out[index] = gate / (1 + T.exp(-gate)) * up
        elif kind == "linear_add":
            out[index] = residual[index] + gate
        else:
            out[index] = gate

    return store


def projection_abi(kind, p, algorithm):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    r, k, o, q = p.get("r", 1), p["k"], p["o"], p["type"]
    count, dtype = weight_storage(q, k * o, packed_words=True)
    args = [("x", r * k, "float32"), ("w", count, dtype)]
    if kind == "ffn":

        @T.macro
        def call(x, w, w2, out):
            algorithm(x, w, out, w2)

        return primitive([*args, ("w2", count, dtype), ("out", r * o, "float32")], call)
    if kind == "linear_add":

        @T.macro
        def call(x, w, residual, out):
            algorithm(x, w, out, None, residual)

        return primitive(
            [*args, ("residual", r * o, "float32"), ("out", r * o, "float32")], call
        )
    return primitive([*args, ("out", r * o, "float32")], algorithm)


def prefill_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.register_matmul import register_matmul
    from tensor.compiler.webgpu.schedules.partitioned import partitioned_matmul
    from tensor.compiler.webgpu.schedules.outer_product import outer_product_matmul

    r, k, o, q = (p[n] for n in ("r", "k", "o", "type"))
    parts = 2 if kind == "ffn" else 1
    tm, tn, bk = p.get("tile", (16, 32, 64))
    rounded, decoder = round_half(), weight_decoder(q, k, packed_words=True)
    epilogue = projection_epilogue(kind)

    @T.macro
    def read_rhs(w, col, channel):
        if q == 1:
            return decoder(w, col, channel)
        else:
            return rounded(decoder(w, col, channel))

    if p.get("schedule") == "outer":
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
    return projection_abi(kind, p, algorithm)


def decode_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import registers
    from tensor.compiler.webgpu.schedules.registers import shared_buffers
    from tensor.compiler.webgpu.schedules.gemv import streamed_gemv

    k, o, q = p["k"], p["o"], p["type"]
    parts = 2 if kind == "ffn" else 1
    epilogue = projection_epilogue(kind)
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
            kind, p, streamed_gemv(k, o, epilogue, parts=parts, **config)
        )
    byte, half_value, word_value = packed_loads()
    decoder = weight_decoder(q, k, packed_words=True)
    wide = q == 2 and p.get("sg") and k % 256 == 0 and min(k, o) >= 2048
    lanes, threads, chains = (16 if q == 2 else 32), 128, 1
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
            or chains not in (1, 4)
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
        else (
            k // 256
            if q == 14
            else k // 128
            if q in (0, 1) and k % 128 == 0
            else k // 32
        )
    )
    unroll, independent = p.get("gemv_unroll", 1), p.get("gemv_chains", 1)
    dot_q4, dot_q6 = (
        fast_q4 and p.get("gemv_dot", wide),
        q == 14 and p.get("gemv_q6_dot"),
    )
    if dot_q4 and chains != 1:
        raise ValueError("packed dot schedule uses one accumulator")
    if unroll != 1 or independent != 1:
        if (
            q not in (2, 14)
            or type(unroll) is not int
            or unroll not in (1, 2, 4, 8, 16)
            or type(independent) is not int
            or independent not in (1, 2, 4, 8)
            or independent > unroll
            or chains != 1
            or tiles % unroll
        ):
            raise ValueError("invalid packed decode unroll or accumulator chains")
        chains = independent

    def q4_terms(acc, x, index, scale, packed, slot):
        if dot_q4:
            for component in range(2):
                vector_term(acc[slot][0], x, index, scale, packed, component)
        else:
            for component in range(4):
                chain = component if p.get("gemv_accumulators", 1) == 4 else slot
                scalar_term(
                    acc[chain][0], x, index + component, scale, packed, component * 8, 0
                )
                scalar_term(
                    acc[chain][0],
                    x,
                    index + component,
                    scale,
                    packed,
                    component * 8 + 4,
                    16,
                )

    def q6_terms(acc, x, w, base, tile, lane, slot, scale):
        if dot_q6:
            for half in range(2):
                q6_vector(acc[slot][0], x, w, base, tile, lane, half, scale)
        else:
            for group in range(8):
                q6_scalar(acc[slot][0], x, w, base, tile, lane, group, scale)

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
    def scalar_term(acc: T.Ref, x, index, scale, packed, shift, offset):
        acc = acc + x[index + offset] * (
            scale * (T.cast((packed >> shift) & 15, "float32") - 8)
        )

    @T.macro
    def vector_term(acc: T.Ref, x, index, scale, packed, component):
        left = T.call_extern(
            "float32x4",
            "vec4<f32>",
            x[index + component * 16],
            x[index + component * 16 + 1],
            x[index + component * 16 + 2],
            x[index + component * 16 + 3],
        )
        right = T.call_extern(
            "float32x4",
            "vec4<f32>",
            T.cast((packed >> (component * 4)) & 15, "float32") - 8,
            T.cast((packed >> (8 + component * 4)) & 15, "float32") - 8,
            T.cast((packed >> (16 + component * 4)) & 15, "float32") - 8,
            T.cast((packed >> (24 + component * 4)) & 15, "float32") - 8,
        )
        acc = acc + scale * T.call_extern("float32", "dot", left, right)

    @T.macro
    def q6_scalar(acc: T.Ref, x, w, base, tile, lane, group, scale):
        low = byte(w, base + (group // 4 * 2 + group % 2) * 32 + lane)
        high = byte(w, base + 128 + group // 4 * 32 + lane)
        bits = byte(w, base + 192 + group * 2 + lane // 16)
        signed_scale = T.cast(
            T.cast(bits, "int32")
            - T.if_then_else(T.cast(bits, "int32") >= 128, 256, 0),
            "float32",
        )
        value = ((low >> (4 * (group % 4 // 2))) & 15) | (
            ((high >> (2 * (group % 4))) & 3) << 4
        )
        acc = acc + x[tile * 256 + group * 32 + lane] * (
            scale * signed_scale * (T.cast(value, "float32") - 32)
        )

    @T.macro
    def q6_vector(acc: T.Ref, x, w, base, tile, lane, half, scale):
        low = word_value(w, base + half * 64 + (lane // 8 % 2) * 32 + lane % 8 * 4)
        high = word_value(w, base + 128 + half * 32 + lane % 8 * 4)
        j = tile * 256 + half * 128 + lane * 4
        left = T.call_extern(
            "float32x4", "vec4<f32>", x[j], x[j + 1], x[j + 2], x[j + 3]
        )
        right = T.call_extern(
            "float32x4",
            "vec4<f32>",
            T.cast(
                ((low >> (lane // 16 * 4)) & 15)
                | (((high >> (lane // 8 * 2)) & 3) << 4),
                "float32",
            )
            - 32,
            T.cast(
                ((low >> (8 + lane // 16 * 4)) & 15)
                | (((high >> (8 + lane // 8 * 2)) & 3) << 4),
                "float32",
            )
            - 32,
            T.cast(
                ((low >> (16 + lane // 16 * 4)) & 15)
                | (((high >> (16 + lane // 8 * 2)) & 3) << 4),
                "float32",
            )
            - 32,
            T.cast(
                ((low >> (24 + lane // 16 * 4)) & 15)
                | (((high >> (24 + lane // 8 * 2)) & 3) << 4),
                "float32",
            )
            - 32,
        )
        bits = byte(w, base + 192 + half * 8 + lane // 4)
        signed_scale = T.cast(
            T.cast(bits, "int32")
            - T.if_then_else(T.cast(bits, "int32") >= 128, 256, 0),
            "float32",
        )
        acc = acc + scale * signed_scale * T.call_extern("float32", "dot", left, right)

    @T.macro
    def update_native(acc: T.Ref, x, w, tile, lane, row, bx):
        index = tile * 128 + lane * 4
        left = T.call_extern(
            "float32x4", "vec4<f32>", x[index], x[index + 1], x[index + 2], x[index + 3]
        )
        base = (bx * 4 + row) * k + index
        right = T.call_extern(
            "float32x4",
            "vec4<f32>",
            T.cast(w[base], "float32"),
            T.cast(w[base + 1], "float32"),
            T.cast(w[base + 2], "float32"),
            T.cast(w[base + 3], "float32"),
        )
        acc = acc + T.call_extern("float32", "dot", left, right)

    @T.macro
    def update_scalar(acc: T.Ref, x, w, tile, lane, row, bx):
        acc = acc + x[tile * 32 + lane] * decoder(
            w, bx * row_count + row, tile * 32 + lane
        )

    @T.macro
    def compute(acc, x, w, tile, lane, row, bx, slot):
        if fast_q4:
            base = (
                (bx * row_count + row) * (k // 32) + tile * (lanes // 4) + lane // 4
            ) * 18
            scale = half_value(w, base)
            packed = word_value(w, base + 2 + lane % 4 * 4)
            index = tile * (lanes * 8) + lane // 4 * 32 + lane % 4 * 4
            q4_terms(acc, x, index, scale, packed, slot)
        elif q == 14:
            base = ((bx * row_count + row) * (k // 256) + tile) * 210
            scale = half_value(w, base + 208)
            q6_terms(acc, x, w, base, tile, lane, slot, scale)
        elif q in (0, 1) and k % 128 == 0:
            update_native(acc[slot][0], x, w, tile, lane, row, bx)
        elif q == 2:
            base = ((bx * row_count + row) * (k // 32) + tile) * 18
            scale = half_value(w, base)
            packed = byte(w, base + 2 + lane)
            scalar_term(acc[slot][0], x, tile * 32 + lane, scale, packed, 0, 0)
            scalar_term(acc[slot][0], x, tile * 32 + lane, scale, packed, 4, 16)
        else:
            update_scalar(acc[slot][0], x, w, tile, lane, row, bx)

    @T.macro
    def combine4(out: T.Ref, a, b, c, d):
        out = (a + b) + (c + d)

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

    return projection_abi(kind, p, gemv)


def integer_prefill_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive
    from tensor.compiler.webgpu.schedules.packed_integer import packed_integer_matmul

    r, k, o = p.get("r", 1), p["k"], p["o"]
    parts = 2 if kind == "ffn_q16" else 1
    byte, half_value, word_value = packed_loads()
    prepacked = p.get("q4_prepacked", False)
    count = k * o // 32 * (9 if prepacked else 18) // (1 if prepacked else 4)

    @T.macro
    def read_word(w, col, block, word):
        if prepacked:
            return w[(col * (k // 32) + block) * 9 + word]
        else:
            return word_value(w, (col * (k // 32) + block) * 18 + 2 + word * 4)

    @T.macro
    def read_scale(w, col, block):
        if prepacked:
            return T.reinterpret("float32", w[(col * (k // 32) + block) * 9 + 8])
        else:
            return half_value(w, (col * (k // 32) + block) * 18)

    config = dict(p.get("integer", {}))
    if prepacked:
        config["signed_rhs"] = True
    algorithm = packed_integer_matmul(
        r, k, o, read_word, read_scale, projection_epilogue(kind), parts=parts, **config
    )
    args = [
        ("packed", r * k // 2, "uint32"),
        ("scales", r * k // 16, "float32"),
        ("sums", r * k // 16, "int32"),
        ("w", count, "uint32"),
    ]
    if parts == 2:

        @T.macro
        def call(packed, scales, sums, w, w2, out):
            algorithm(packed, scales, sums, w, out, w2)

        return primitive(
            [*args, ("w2", count, "uint32"), ("out", r * o, "float32")], call
        )
    return primitive([*args, ("out", r * o, "float32")], algorithm)


def q8_kernel(p):
    import tilelang.language as T

    k, o, lanes = p["k"], p["o"], 16
    segmented = segmented_reduce(lanes)
    byte, half_value, word_value = packed_loads()

    @T.prim_func
    def kernel(
        x: T.Tensor((k // 4,), "uint32"),
        scales: T.Tensor((k // 32,), "float32"),
        sums: T.Tensor((k // 32,), "int32"),
        w: T.Tensor((k * o // 32 * 18 // 4,), "uint32"),
        out: T.Tensor((o,), "float32"),
    ):
        with T.Kernel(T.ceildiv(o, 8), threads=128) as bx:
            tx = T.get_thread_binding()
            row = tx // lanes
            lane = tx % lanes
            accum = T.alloc_var("float32")
            scratch = T.alloc_shared((128,), "float32")
            accum = 0
            if bx * 8 + row < o:
                for tile in T.serial(k // 128):
                    block = tile * 4 + lane // 4
                    base = ((bx * 8 + row) * (k // 32) + block) * 18
                    packed = word_value(w, base + 2 + lane % 4 * 4)
                    low = T.call_extern(
                        "int32",
                        "dot4I8Packed",
                        packed & T.uint32(252645135),
                        x[block * 8 + lane % 4],
                    )
                    high = T.call_extern(
                        "int32",
                        "dot4I8Packed",
                        (packed >> 4) & T.uint32(252645135),
                        x[block * 8 + lane % 4 + 4],
                    )
                    accum = accum + T.cast(
                        low + high - 2 * sums[block], "float32"
                    ) * scales[block] * half_value(w, base)
            segmented(accum, scratch, tx, lane)
            if (lane == 0) & (bx * 8 + row < o):
                out[bx * 8 + row] = scratch[tx]

    return kernel


def make_kernel(kind, p):
    if kind in ("linear", "ffn") and p.get("q16"):
        return integer_prefill_kernel(kind + "_q16", p)
    if kind in ("linear_q16", "ffn_q16"):
        return integer_prefill_kernel(kind, p)
    r = p.get("r", 1)
    if kind in ("linear", "ffn", "linear_add"):
        if kind == "linear_add" and r != 1:
            raise ValueError("residual projection fusion requires decode rows")
        return decode_kernel(kind, p) if r == 1 else prefill_kernel(kind, p)
    if kind == "linear_q8":
        return q8_kernel(p)
    raise ValueError("unsupported operation: " + kind)
