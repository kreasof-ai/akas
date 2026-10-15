"""WebGPU specializations and shared bindings for ggml."""

from .definition import FORMAT, IMPLEMENTATIONS as _CHOICES


def decoder(encoding, depth):
    from akas.quantization.weights.ggml.kernel import weight_decoder

    return weight_decoder(encoding, depth, packed_words=True)


def bind(shape, *, encoding):
    from akas.quantization.formats import Codec
    from akas.quantization._shared.access import matrix_reader, matrix_tile
    from .definition import FORMAT

    storage = FORMAT.storage(shape, provider="webgpu", encoding=encoding)
    read = matrix_reader(decoder(encoding, shape[-1]))

    return Codec(storage, read, matrix_tile(read, shape))


def gemv_terms(p, *, fast_q4, lanes, row_count, dot_q4, dot_q6):
    """Format-specific lane products; the operation owns launch and reduction."""
    import tilelang.language as T
    from akas.quantization._shared.packing import packed_loads
    from .kernel import weight_decoder

    k, q = p["k"], p["type"]
    byte, half_value, word_value = packed_loads()
    decoder = weight_decoder(q, k, packed_words=True)

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

    @T.macro
    def scalar_term(acc: T.Ref, x, index, scale, packed, shift, offset):
        acc = acc + x[index + offset] * (
            scale * (T.cast(packed >> shift & 15, "float32") - 8)
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
            T.cast(packed >> component * 4 & 15, "float32") - 8,
            T.cast(packed >> 8 + component * 4 & 15, "float32") - 8,
            T.cast(packed >> 16 + component * 4 & 15, "float32") - 8,
            T.cast(packed >> 24 + component * 4 & 15, "float32") - 8,
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
        value = low >> 4 * (group % 4 // 2) & 15 | (high >> 2 * (group % 4) & 3) << 4
        acc = acc + x[tile * 256 + group * 32 + lane] * (
            scale * signed_scale * (T.cast(value, "float32") - 32)
        )

    @T.macro
    def q6_vector(acc: T.Ref, x, w, base, tile, lane, half, scale):
        low = word_value(w, base + half * 64 + lane // 8 % 2 * 32 + lane % 8 * 4)
        high = word_value(w, base + 128 + half * 32 + lane % 8 * 4)
        j = tile * 256 + half * 128 + lane * 4
        left = T.call_extern(
            "float32x4", "vec4<f32>", x[j], x[j + 1], x[j + 2], x[j + 3]
        )
        right = T.call_extern(
            "float32x4",
            "vec4<f32>",
            T.cast(
                low >> lane // 16 * 4 & 15 | (high >> lane // 8 * 2 & 3) << 4, "float32"
            )
            - 32,
            T.cast(
                low >> 8 + lane // 16 * 4 & 15 | (high >> 8 + lane // 8 * 2 & 3) << 4,
                "float32",
            )
            - 32,
            T.cast(
                low >> 16 + lane // 16 * 4 & 15 | (high >> 16 + lane // 8 * 2 & 3) << 4,
                "float32",
            )
            - 32,
            T.cast(
                low >> 24 + lane // 16 * 4 & 15 | (high >> 24 + lane // 8 * 2 & 3) << 4,
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

    return compute


def integer_rhs(shape, *, prepacked=False):
    """Q4_0 logical words/scales for an integer matrix consumer."""
    from .definition import integer_layout

    k = shape[-1]
    storage = integer_layout(shape, prepacked=prepacked)
    import tilelang.language as T
    from akas.quantization._shared.packing import packed_loads

    byte, half_value, word_value = packed_loads()

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

    return storage, read_word, read_scale


def q8_dot(k, o):
    """Packed Q8 activation × Q4_0 weight dot with inline correction sums."""
    if k <= 0 or k % 128:
        raise ValueError(
            "packed Q8 dot requires depth divisible by 128; use quantized_linear for other shapes"
        )
    import tilelang.language as T
    from akas.kernels.webgpu import segmented_reduce
    from akas.quantization._shared.packing import packed_loads

    lanes = 16
    segmented = segmented_reduce(lanes)
    byte, half_value, word_value = packed_loads()

    @T.macro
    def dot(x, scales, sums, w, out):
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
                        packed >> 4 & T.uint32(252645135),
                        x[block * 8 + lane % 4 + 4],
                    )
                    accum = accum + T.cast(
                        low + high - 2 * sums[block], "float32"
                    ) * scales[block] * half_value(w, base)
            segmented(accum, scratch, tx, lane)
            if (lane == 0) & (bx * 8 + row < o):
                out[bx * 8 + row] = scratch[tx]

    return dot


def dequantize(spec, schedule):
    from .kernel import dequantize

    return dequantize(spec, schedule, provider="webgpu")


AVAILABLE = "webgpu" in dict(FORMAT.providers)
CAPABILITIES = ("read", "read_tile", "dequantize")
IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "webgpu")
