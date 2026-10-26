"""CUDA specializations and shared bindings for ggml."""

from .definition import FORMAT, IMPLEMENTATIONS as _CHOICES


def hardware_loads():
    import tilelang.language as T

    @T.macro
    def u16(w, offset):
        return T.call_extern("uint32", "tensor_load_u16", T.address_of(w[offset]))

    @T.macro
    def word(w, offset):
        return u16(w, offset) | u16(w, offset + 2) << 16

    @T.macro
    def half(w, offset):
        return T.cast(
            T.reinterpret("float16", T.cast(u16(w, offset), "uint16")), "float32"
        )

    return (u16, word, half)


def packed_dot(q, k, row_bytes, width=4):
    """One lane's packed weight dot product, inlined at each GEMV use."""
    if q == 8:
        return q8_packed_dot(k, row_bytes)
    import tilelang.language as T

    u16, word, half_value = hardware_loads()

    @T.macro
    def dot_product(x, w, tile, lane, row):
        if q in (0, 1):
            j = tile * (32 * width) + lane * width
            activation = T.alloc_local((width,), "float32")
            coefficient = T.alloc_local((width,), "float16" if q == 1 else "float32")
            for component in T.vectorized(width):
                activation[component] = x[j + component]
            for component in T.vectorized(width):
                coefficient[component] = w[row * k + j + component]
            dot = T.alloc_var("float32")
            dot = 0
            for component in T.unroll(width):
                dot = T.ieee_fmaf(
                    activation[component],
                    T.cast(coefficient[component], "float32"),
                    dot,
                )
            return dot
        elif q == 2:
            base = row * row_bytes + (tile * 8 + lane // 4) * 18
            scale = half_value(w, base)
            packed = word(w, base + 2 + lane % 4 * 4)
            j = tile * 256 + lane // 4 * 32 + lane % 4 * 4
            activation = T.alloc_local((4,), "float32")
            activation_hi = T.alloc_local((4,), "float32")
            for component in T.vectorized(4):
                activation[component] = x[j + component]
            for component in T.vectorized(4):
                activation_hi[component] = x[j + 16 + component]
            dot = T.alloc_var("float32")
            dot = 0
            for component in T.unroll(4):
                dot = T.ieee_fmaf(
                    activation[component],
                    T.cast(packed >> component * 8 & 15, "float32") - 8,
                    dot,
                )
            for component in T.unroll(4):
                dot = T.ieee_fmaf(
                    activation_hi[component],
                    T.cast(packed >> component * 8 + 4 & 15, "float32") - 8,
                    dot,
                )
            dot = scale * dot
            return dot
        elif q == 12:
            base = row * row_bytes + tile * 144
            group = lane // 8 * 2
            packed = word(w, base + 16 + lane // 8 * 32 + lane % 8 * 4)
            j = tile * 256 + lane // 8 * 64 + lane % 8 * 4
            activation = T.alloc_local((4,), "float32")
            activation_hi = T.alloc_local((4,), "float32")
            for component in T.vectorized(4):
                activation[component] = x[j + component]
            for component in T.vectorized(4):
                activation_hi[component] = x[j + 32 + component]
            lo = T.alloc_var("float32")
            hi = T.alloc_var("float32")
            lo = 0
            hi = 0
            for component in T.unroll(4):
                lo = T.ieee_fmaf(
                    activation[component],
                    T.cast(packed >> component * 8 & 15, "float32"),
                    lo,
                )
                hi = T.ieee_fmaf(
                    activation_hi[component],
                    T.cast(packed >> component * 8 + 4 & 15, "float32"),
                    hi,
                )
            s0 = T.if_then_else(
                group < 4,
                w[base + 4 + group] & 63,
                w[base + 8 + group] & 15 | w[base + group] >> 6 << 4,
            )
            s1 = T.if_then_else(
                group < 4,
                w[base + 5 + group] & 63,
                w[base + 9 + group] & 15 | w[base + 1 + group] >> 6 << 4,
            )
            m0 = T.if_then_else(
                group < 4,
                w[base + 8 + group] & 63,
                w[base + 8 + group] >> 4 | w[base + 4 + group] >> 6 << 4,
            )
            m1 = T.if_then_else(
                group < 4,
                w[base + 9 + group] & 63,
                w[base + 9 + group] >> 4 | w[base + 5 + group] >> 6 << 4,
            )
            sum0 = activation[0] + activation[1] + (activation[2] + activation[3])
            sum1 = (
                activation_hi[0]
                + activation_hi[1]
                + (activation_hi[2] + activation_hi[3])
            )
            dot = half_value(w, base) * (
                T.cast(s0, "float32") * lo + T.cast(s1, "float32") * hi
            ) - half_value(w, base + 2) * (
                T.cast(m0, "float32") * sum0 + T.cast(m1, "float32") * sum1
            )
            return dot
        elif q == 14:
            base = row * row_bytes + tile * 210
            dot = T.alloc_var("float32")
            dot = 0
            for half in T.unroll(2):
                low = word(w, base + half * 64 + lane // 8 % 2 * 32 + lane % 8 * 4)
                high = word(w, base + 128 + half * 32 + lane % 8 * 4)
                activation = T.alloc_local((4,), "float32")
                for component in T.vectorized(4):
                    activation[component] = x[
                        tile * 256 + half * 128 + lane * 4 + component
                    ]
                value = T.alloc_var("float32")
                value = 0
                for component in T.unroll(4):
                    quant = T.cast(
                        low >> component * 8 + lane // 16 * 4 & 15, "int32"
                    ) | T.cast(
                        (high >> component * 8 + lane // 8 * 2 & 3) << 4, "int32"
                    )
                    value = T.ieee_fmaf(
                        activation[component], T.cast(quant - 32, "float32"), value
                    )
                scale = T.cast(
                    T.reinterpret("int8", w[base + 192 + half * 8 + lane // 4]),
                    "float32",
                )
                dot = T.ieee_fmaf(scale, value, dot)
            dot = half_value(w, base + 208) * dot
            return dot
        else:
            raise ValueError("unsupported packed dot encoding")

    return dot_product


def pair_decoder(q, k):
    import tilelang.language as T
    from akas.quantization.weights.ggml.definition import ENCODINGS as TYPES
    from akas.quantization.weights.ggml.cuda import hardware_loads

    u16, word, half_value = hardware_loads()
    _, block, size = TYPES[q]
    row_bytes = k * 2 if q == 1 else k // block * size

    @T.macro
    def pair_load(w, row, col):
        if q == 1:
            return T.ldg32(w[row * k + col])
        elif q == 2:
            base = row * row_bytes + col // 32 * 18
            scale = half_value(w, base)
            packed = u16(w, base + 2 + col % 16)
            shift = col % 32 // 16 * 4
            lo = scale * (T.cast(packed >> shift & 15, "float32") - 8)
            hi = scale * (T.cast(packed >> shift + 8 & 15, "float32") - 8)
            return T.call_extern("uint32", "tensor_pack_f16x2", lo, hi)
        elif q == 12:
            base = row * row_bytes + col // 256 * 144
            weight_col = col % 256
            group = weight_col // 32
            scale = T.if_then_else(
                group < 4,
                w[base + 4 + group] & 63,
                w[base + 8 + group] & 15 | w[base + group] >> 6 << 4,
            )
            minimum = T.if_then_else(
                group < 4,
                w[base + 8 + group] & 63,
                w[base + 8 + group] >> 4 | w[base + 4 + group] >> 6 << 4,
            )
            ds = half_value(w, base) * T.cast(scale, "float32")
            dm = half_value(w, base + 2) * T.cast(minimum, "float32")
            packed = u16(w, base + 16 + weight_col // 64 * 32 + weight_col % 32)
            shift = group % 2 * 4
            lo = ds * T.cast(packed >> shift & 15, "float32") - dm
            hi = ds * T.cast(packed >> shift + 8 & 15, "float32") - dm
            return T.call_extern("uint32", "tensor_pack_f16x2", lo, hi)
        elif q == 14:
            base = row * row_bytes + col // 256 * 210
            weight_col = col % 256
            group = weight_col % 128 // 32
            low = u16(
                w, base + weight_col // 128 * 64 + group % 2 * 32 + weight_col % 32
            )
            high = u16(w, base + 128 + weight_col // 128 * 32 + weight_col % 32)
            a = T.cast(low >> group // 2 * 4 & 15, "int32") | T.cast(
                (high >> group * 2 & 3) << 4, "int32"
            )
            b = T.cast(low >> group // 2 * 4 + 8 & 15, "int32") | T.cast(
                (high >> group * 2 + 8 & 3) << 4, "int32"
            )
            scale = half_value(w, base + 208) * T.cast(
                T.reinterpret("int8", w[base + 192 + weight_col // 16]), "float32"
            )
            return T.call_extern(
                "uint32",
                "tensor_pack_f16x2",
                scale * T.cast(a - 32, "float32"),
                scale * T.cast(b - 32, "float32"),
            )
        else:
            raise ValueError("unsupported packed pair encoding")

    return pair_load


def decoder(encoding, depth):
    from akas.quantization.weights.ggml.kernel import weight_decoder

    return weight_decoder(encoding, depth)


_Q4_HALF2_SOURCE = r"""
__device__ __forceinline__ unsigned int akas_q4_adjacent_pair(unsigned int bits, unsigned int scale) {
    unsigned int pair = (bits & 15u) | ((bits & 0x0f00u) << 8u) | 0x64006400u;
    unsigned int integers, result;
    asm("sub.f16x2 %0, %1, %2;" : "=r"(integers) : "r"(pair), "r"(0x64086408u));
    asm("mul.f16x2 %0, %1, %2;" : "=r"(result) : "r"(integers), "r"(scale | (scale << 16u)));
    return result;
}
"""


def packed_tile_half(shape, read):
    """Decode aligned Q4 words into FP16 tiles; retain other staging precisions.

    Two whole blocks per row keep word views within the original allocation.
    Each lane decodes four adjacent pairs with the same coefficient rounding
    as scalar F32 decoding followed by an FP16 store.
    """
    from math import prod
    from akas.quantization._shared.access import matrix_tile
    import tilelang.language as T

    rows, depth = prod(shape[:-1]), shape[-1]
    fallback = matrix_tile(read, shape)
    row_bytes = depth // 32 * 18
    words_count = rows * row_bytes // 4

    @T.macro
    def load(encoded, destination, row, column):
        if (
            destination.dtype == "float16"
            and destination.shape[1] % 32 == 0
            and depth % 64 == 0
        ):
            if column % 32 == 0:
                T.import_source(_Q4_HALF2_SOURCE)
                # Views are compile-time aliases. Naming the view as a device
                # variable renames its input pointer and collides when a fused
                # kernel binds two weight matrices through this same macro.
                words = T.meta_var(T.view(encoded[0], (words_count,), dtype="uint32"))
                for i, j in T.Parallel(destination.shape[0], destination.shape[1] // 8):
                    local = j // 4 * 32 + j % 4 * 4
                    channel = column + local
                    if row + i < rows and channel + 19 < depth:
                        base = (row + i) * row_bytes + channel // 32 * 18
                        scale = T.cast(words[base // 4] >> (base % 4 * 8), "uint16")
                        address = base + 2 + channel % 16
                        packed = T.call_extern(
                            "uint32",
                            "__funnelshift_r",
                            words[address // 4],
                            words[(address + 3) // 4],
                            T.cast(address % 4 * 8, "uint32"),
                        )
                        lo0 = T.call_extern(
                            "uint32",
                            "akas_q4_adjacent_pair",
                            packed,
                            T.cast(scale, "uint32"),
                        )
                        lo1 = T.call_extern(
                            "uint32",
                            "akas_q4_adjacent_pair",
                            packed >> 16,
                            T.cast(scale, "uint32"),
                        )
                        hi0 = T.call_extern(
                            "uint32",
                            "akas_q4_adjacent_pair",
                            packed >> 4,
                            T.cast(scale, "uint32"),
                        )
                        hi1 = T.call_extern(
                            "uint32",
                            "akas_q4_adjacent_pair",
                            packed >> 20,
                            T.cast(scale, "uint32"),
                        )
                        destination[i, local] = T.reinterpret(
                            "float16", T.cast(lo0, "uint16")
                        )
                        destination[i, local + 1] = T.reinterpret(
                            "float16", T.cast(lo0 >> 16, "uint16")
                        )
                        destination[i, local + 2] = T.reinterpret(
                            "float16", T.cast(lo1, "uint16")
                        )
                        destination[i, local + 3] = T.reinterpret(
                            "float16", T.cast(lo1 >> 16, "uint16")
                        )
                        destination[i, local + 16] = T.reinterpret(
                            "float16", T.cast(hi0, "uint16")
                        )
                        destination[i, local + 17] = T.reinterpret(
                            "float16", T.cast(hi0 >> 16, "uint16")
                        )
                        destination[i, local + 18] = T.reinterpret(
                            "float16", T.cast(hi1, "uint16")
                        )
                        destination[i, local + 19] = T.reinterpret(
                            "float16", T.cast(hi1 >> 16, "uint16")
                        )
                    else:
                        for offset in T.unroll(4):
                            destination[i, local + offset] = 0
                            destination[i, local + offset + 16] = 0
            else:
                fallback(encoded, destination, row, column)
        else:
            fallback(encoded, destination, row, column)

    return load


def packed_dot_half(shape):
    """A warp dots 256 values, rounding both operands to FP16 before F32 FMA.

    Keep the original GGUF bytes. A word view provides aligned loads and a
    funnel shift joins the alternating halfword-aligned Q4 blocks. No derived
    cache or fully dequantized matrix is allocated. ``offset`` selects one
    routed activation row from the caller's flat activation view. The final
    ``previous`` argument carries the lane accumulator across tiles, preserving
    continuous F32 FMA order instead of summing separately rounded tile dots.
    """
    import tilelang.language as T
    from math import prod

    depth = shape[-1]
    words_count = prod(shape) // 32 * 18 // 4
    row_bytes = depth // 32 * 18

    @T.macro
    def dot(encoded, x, offset, tile, lane, row, previous):
        T.import_source(_Q4_HALF2_SOURCE)
        words = T.meta_var(T.view(encoded[0], (words_count,), dtype="uint32"))
        base = row * row_bytes + (tile * 8 + lane // 4) * 18
        scale = T.cast(words[base // 4] >> (base % 4 * 8), "uint16")
        address = base + 2 + lane % 4 * 4
        packed = T.call_extern(
            "uint32",
            "__funnelshift_r",
            words[address // 4],
            words[(address + 3) // 4],
            T.cast(address % 4 * 8, "uint32"),
        )
        coefficients = T.alloc_local((8,), "float32")
        for half in T.unroll(2):
            lo_bits = T.call_extern(
                "uint32",
                "akas_q4_adjacent_pair",
                packed >> (half * 16),
                T.cast(scale, "uint32"),
            )
            hi_bits = T.call_extern(
                "uint32",
                "akas_q4_adjacent_pair",
                packed >> (half * 16 + 4),
                T.cast(scale, "uint32"),
            )
            coefficients[half * 2] = T.cast(
                T.reinterpret("float16", T.cast(lo_bits, "uint16")), "float32"
            )
            coefficients[half * 2 + 1] = T.cast(
                T.reinterpret("float16", T.cast(lo_bits >> 16, "uint16")), "float32"
            )
            coefficients[4 + half * 2] = T.cast(
                T.reinterpret("float16", T.cast(hi_bits, "uint16")), "float32"
            )
            coefficients[5 + half * 2] = T.cast(
                T.reinterpret("float16", T.cast(hi_bits >> 16, "uint16")), "float32"
            )
        j = offset + tile * 256 + lane // 4 * 32 + lane % 4 * 4
        lo = T.alloc_local((4,), "float32")
        hi = T.alloc_local((4,), "float32")
        for component in T.vectorized(4):
            lo[component] = T.cast(T.cast(x[j + component], "float16"), "float32")
            hi[component] = T.cast(T.cast(x[j + 16 + component], "float16"), "float32")
        result = T.alloc_var("float32")
        result = previous
        for component in T.unroll(4):
            result += lo[component] * coefficients[component]
        for component in T.unroll(4):
            result += hi[component] * coefficients[component + 4]
        return result

    return dot


def packed_dot_float(shape):
    """Packed Q4 lane tile with F32 operands, coefficients and accumulation."""
    import tilelang.language as T
    from math import prod

    depth = shape[-1]
    row_bytes = depth // 32 * 18
    words_count = prod(shape) // 32 * 18 // 4

    @T.macro
    def dot(encoded, x, offset, tile, lane, row, previous):
        words = T.meta_var(T.view(encoded[0], (words_count,), dtype="uint32"))
        base = row * row_bytes + (tile * 8 + lane // 4) * 18
        scale_bits = T.cast(words[base // 4] >> (base % 4 * 8), "uint16")
        scale = T.cast(T.reinterpret("float16", scale_bits), "float32")
        address = base + 2 + lane % 4 * 4
        packed = T.call_extern(
            "uint32",
            "__funnelshift_r",
            words[address // 4],
            words[(address + 3) // 4],
            T.cast(address % 4 * 8, "uint32"),
        )
        channel = offset + tile * 256 + lane // 4 * 32 + lane % 4 * 4
        partial = T.alloc_var("float32")
        partial = 0
        for component in T.unroll(4):
            coefficient = (
                T.cast((packed >> (component * 8)) & 15, "float32") - 8
            ) * scale
            partial = T.ieee_fmaf(x[channel + component], coefficient, partial)
        for component in T.unroll(4):
            coefficient = (
                T.cast((packed >> (component * 8 + 4)) & 15, "float32") - 8
            ) * scale
            partial = T.ieee_fmaf(x[channel + 16 + component], coefficient, partial)
        return previous + partial

    return dot


def supports(capability, shape, *, encoding, layout="raw"):
    """Compiler-free specialization inventory for packed consumers."""
    from .definition import validate_layout

    validate_layout(shape, encoding=encoding, layout=layout)
    if layout == "planar":
        return capability in ("read", "read_tile", "dot_half", "dot_float")
    if capability in ("dot_half", "dot_float"):
        return encoding == 2 and shape[-1] % 256 == 0
    if capability == "read_pair":
        return encoding in (1, 2, 12, 14)
    if capability == "dot":
        return encoding in (0, 1, 2, 8, 12, 14) and shape[-1] % 256 == 0
    if capability == "stage_tile":
        from .definition import ENCODINGS

        _, block, size = ENCODINGS[encoding]
        return (shape[-1] // block * size) % 2 == 0
    return capability in CAPABILITIES


def bind(shape, *, encoding, layout="raw", vector_tiles=False):
    if layout == "planar":
        return planar_binding(shape, encoding=encoding)
    import tilelang.language as T
    from akas.quantization.formats import Codec
    from akas.quantization._shared.access import matrix_reader, matrix_tile
    from .definition import FORMAT, ENCODINGS

    storage = FORMAT.storage(shape, provider="cuda", encoding=encoding)
    read = matrix_reader(decoder(encoding, shape[-1]))
    pair = pair_decoder(encoding, shape[-1]) if encoding in (1, 2, 12, 14) else None
    _, block, size = ENCODINGS[encoding]
    product = (
        packed_dot(encoding, shape[-1], shape[-1] // block * size)
        if encoding in (0, 1, 2, 8, 12, 14)
        else None
    )

    @T.macro
    def read_pair(encoded, row, column):
        return pair(encoded[0], row, column)

    @T.macro
    def dot(encoded, x, tile, lane, row):
        return product(x, encoded[0], tile, lane, row)

    tile = packed_tile_half(shape, read) if encoding == 2 else matrix_tile(read, shape)
    if vector_tiles and encoding in (8, 12, 14):
        tile = vector_tile(shape, encoding, tile)
    return Codec(
        storage,
        read,
        tile,
        read_pair=read_pair if pair is not None else None,
        dot=dot if product is not None else None,
        dot_half=packed_dot_half(shape)
        if supports("dot_half", shape, encoding=encoding)
        else None,
        dot_float=packed_dot_float(shape)
        if supports("dot_float", shape, encoding=encoding)
        else None,
        stage_tile=staged_tile(shape, encoding=encoding)
        if supports("stage_tile", shape, encoding=encoding)
        else None,
    )


def dequantize(spec, schedule):
    from .kernel import dequantize

    return dequantize(spec, schedule, provider="cuda")


AVAILABLE = "cuda" in dict(FORMAT.providers)
CAPABILITIES = (
    "read",
    "read_tile",
    "read_pair",
    "dot",
    "dot_half",
    "dot_float",
    "dequantize",
    "stage_tile",
)
IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "cuda")


def staged_tile(shape, *, encoding):
    """Stage original embedded GGML blocks, then reuse their local tile reader.

    A tile has the same encoding as a small independent matrix. There is no
    persistent dequantized weight copy. Consumers select geometry and pipeline
    stages; this capability supplies byte layout and coefficient arithmetic.
    """
    from math import prod
    import tilelang.language as T
    from akas.memory.state import BufferSpec
    from akas.quantization.formats import StagedTile
    from .definition import ENCODINGS

    _, block, size = ENCODINGS[encoding]
    rows, depth = prod(shape[:-1]), shape[-1]
    stride = depth // block * size

    def tile(tile_shape, dtype):
        height, width = tile_shape
        if (height < 1 or width < 1 or width % block or depth % width
                or height > rows or dtype not in ("float16", "float32")):
            raise ValueError("staged GGML tiles need complete aligned encoding blocks")
        tile_bytes = width // block * size
        unit = next((value for value in (8, 4, 2) if stride % value == 0
                     and tile_bytes % value == 0), None)
        if unit is None:
            raise ValueError("staged GGML tile has no aligned word representation")
        word_dtype = f"uint{unit * 8}"
        physical = BufferSpec((height, tile_bytes // unit), word_dtype)
        local = bind((height, width), encoding=encoding)
        read = local.require("read_tile")

        @T.macro
        def load(encoded, staged, row, column):
            words = T.meta_var(T.view(encoded[0], (rows, stride // unit),
                                     dtype=word_dtype))
            start = T.truncdiv(column * size, block * unit)
            T.copy(words[row:row + height, start:start + tile_bytes // unit], staged)

        @T.macro
        def decode(staged, destination):
            encoded = T.meta_var(T.view(staged, local.storage.values.shape,
                                       dtype=local.storage.values.dtype))
            read((encoded,), destination, 0, 0)

        return StagedTile(physical, load, decode)

    return tile


def planar_dot_half(shape):
    """Aligned Q4 payloads with unchanged F16 scales and continuous F32 FMA.

    Coefficients and activations retain the raw layout's FP16 rounding and
    lane addition order. Only byte addresses change; no wider weight cache or
    numerical requantization is introduced.
    """
    import tilelang.language as T
    from math import prod

    depth = shape[-1]
    words_count = prod(shape) // 32 * 18 // 4
    scales_offset = prod(shape) // 2

    @T.macro
    def dot(encoded, x, offset, tile, lane, row, previous):
        T.import_source(_Q4_HALF2_SOURCE)
        words = T.meta_var(T.view(encoded[0], (words_count,), dtype="uint32"))
        halves = T.meta_var(T.view(encoded[0], (words_count * 2,), dtype="uint16"))
        scale = halves[scales_offset // 2 + row * (depth // 32) + tile * 8 + lane // 4]
        packed = words[row * (depth // 8) + tile * 32 + lane]
        coefficients = T.alloc_local((8,), "float32")
        for half in T.unroll(2):
            lo_bits = T.call_extern(
                "uint32",
                "akas_q4_adjacent_pair",
                packed >> (half * 16),
                T.cast(scale, "uint32"),
            )
            hi_bits = T.call_extern(
                "uint32",
                "akas_q4_adjacent_pair",
                packed >> (half * 16 + 4),
                T.cast(scale, "uint32"),
            )
            coefficients[half * 2] = T.cast(
                T.reinterpret("float16", T.cast(lo_bits, "uint16")), "float32"
            )
            coefficients[half * 2 + 1] = T.cast(
                T.reinterpret("float16", T.cast(lo_bits >> 16, "uint16")), "float32"
            )
            coefficients[4 + half * 2] = T.cast(
                T.reinterpret("float16", T.cast(hi_bits, "uint16")), "float32"
            )
            coefficients[5 + half * 2] = T.cast(
                T.reinterpret("float16", T.cast(hi_bits >> 16, "uint16")), "float32"
            )
        j = offset + tile * 256 + lane // 4 * 32 + lane % 4 * 4
        lo = T.alloc_local((4,), "float32")
        hi = T.alloc_local((4,), "float32")
        for component in T.vectorized(4):
            lo[component] = T.cast(T.cast(x[j + component], "float16"), "float32")
            hi[component] = T.cast(T.cast(x[j + 16 + component], "float16"), "float32")
        result = T.alloc_var("float32")
        result = previous
        for component in T.unroll(4):
            result += lo[component] * coefficients[component]
        for component in T.unroll(4):
            result += hi[component] * coefficients[component + 4]
        return result

    return dot


def planar_tile_half(shape, read):
    from math import prod
    from akas.quantization._shared.access import matrix_tile

    rows, depth = prod(shape[:-1]), shape[-1]
    fallback = matrix_tile(read, shape)
    import tilelang.language as T

    words_count = rows * depth // 32 * 18 // 4

    @T.macro
    def load(encoded, destination, row, column):
        if destination.dtype == "float16" and destination.shape[1] % 32 == 0:
            T.import_source(_Q4_HALF2_SOURCE)
            words = T.meta_var(T.view(encoded[0], (words_count,), dtype="uint32"))
            halves = T.meta_var(T.view(encoded[0], (words_count * 2,), dtype="uint16"))
            for i, j in T.Parallel(destination.shape[0], destination.shape[1] // 8):
                local = j // 4 * 32 + j % 4 * 4
                channel = column + local
                if row + i < rows and channel + 19 < depth:
                    block = (row + i) * (depth // 32) + channel // 32
                    scale = halves[rows * depth // 4 + block]
                    packed = words[block * 4 + channel % 16 // 4]
                    lo0 = T.call_extern(
                        "uint32",
                        "akas_q4_adjacent_pair",
                        packed,
                        T.cast(scale, "uint32"),
                    )
                    lo1 = T.call_extern(
                        "uint32",
                        "akas_q4_adjacent_pair",
                        packed >> 16,
                        T.cast(scale, "uint32"),
                    )
                    hi0 = T.call_extern(
                        "uint32",
                        "akas_q4_adjacent_pair",
                        packed >> 4,
                        T.cast(scale, "uint32"),
                    )
                    hi1 = T.call_extern(
                        "uint32",
                        "akas_q4_adjacent_pair",
                        packed >> 20,
                        T.cast(scale, "uint32"),
                    )
                    destination[i, local] = T.reinterpret(
                        "float16", T.cast(lo0, "uint16")
                    )
                    destination[i, local + 1] = T.reinterpret(
                        "float16", T.cast(lo0 >> 16, "uint16")
                    )
                    destination[i, local + 2] = T.reinterpret(
                        "float16", T.cast(lo1, "uint16")
                    )
                    destination[i, local + 3] = T.reinterpret(
                        "float16", T.cast(lo1 >> 16, "uint16")
                    )
                    destination[i, local + 16] = T.reinterpret(
                        "float16", T.cast(hi0, "uint16")
                    )
                    destination[i, local + 17] = T.reinterpret(
                        "float16", T.cast(hi0 >> 16, "uint16")
                    )
                    destination[i, local + 18] = T.reinterpret(
                        "float16", T.cast(hi1, "uint16")
                    )
                    destination[i, local + 19] = T.reinterpret(
                        "float16", T.cast(hi1 >> 16, "uint16")
                    )
                else:
                    for offset in T.unroll(4):
                        destination[i, local + offset] = 0
                        destination[i, local + offset + 16] = 0
        else:
            fallback(encoded, destination, row, column)

    return load


def planar_dot_float(shape):
    """Packed Q4 lane tile with F32 operands, coefficients and accumulation."""
    import tilelang.language as T
    from math import prod

    depth = shape[-1]
    words_count = prod(shape) // 32 * 18 // 4
    scales_offset = prod(shape) // 2

    @T.macro
    def dot(encoded, x, offset, tile, lane, row, previous):
        words = T.meta_var(T.view(encoded[0], (words_count,), dtype="uint32"))
        block = row * (depth // 32) + tile * 8 + lane // 4
        scale_address = scales_offset + block * 2
        bits = words[scale_address // 4] >> (scale_address % 4 * 8)
        scale = T.cast(T.reinterpret("float16", T.cast(bits, "uint16")), "float32")
        packed = words[block * 4 + lane % 4]
        channel = offset + tile * 256 + lane // 4 * 32 + lane % 4 * 4
        partial = T.alloc_var("float32")
        partial = 0
        for component in T.unroll(4):
            coefficient = (
                T.cast((packed >> (component * 8)) & 15, "float32") - 8
            ) * scale
            partial = T.ieee_fmaf(x[channel + component], coefficient, partial)
        for component in T.unroll(4):
            coefficient = (
                T.cast((packed >> (component * 8 + 4)) & 15, "float32") - 8
            ) * scale
            partial = T.ieee_fmaf(x[channel + 16 + component], coefficient, partial)
        return previous + partial

    return dot


def planar_binding(shape, *, encoding):
    from akas.quantization.formats import Codec
    from .definition import FORMAT
    from .kernel import planar_reader

    storage = FORMAT.storage(shape, provider="cuda", encoding=encoding, layout="planar")
    read = planar_reader(shape)
    return Codec(
        storage,
        read,
        planar_tile_half(shape, read),
        dot_half=planar_dot_half(shape),
        dot_float=planar_dot_float(shape),
    )


def q8_packed_dot(depth, row_bytes):
    """Coalesced eight-value Q8_0 lane reads with the original F16 block scale."""
    import tilelang.language as T

    u16, word, half = hardware_loads()

    @T.macro
    def product(x, w, tile, lane, row):
        base = row * row_bytes + (tile * 8 + lane // 4) * 34
        scale = half(w, base)
        low = word(w, base + 2 + (lane % 4) * 8)
        high = word(w, base + 6 + (lane % 4) * 8)
        channel = tile * 256 + lane // 4 * 32 + lane % 4 * 8
        activation = T.alloc_local((8,), "float32")
        for j in T.vectorized(8):
            activation[j] = x[channel + j]
        result = T.alloc_var("float32")
        result = 0
        for j in T.unroll(4):
            quant = T.cast(
                T.reinterpret("int8", T.cast((low >> (j * 8)) & 255, "uint8")),
                "float32",
            )
            result = T.ieee_fmaf(activation[j], scale * quant, result)
        for j in T.unroll(4):
            quant = T.cast(
                T.reinterpret("int8", T.cast((high >> (j * 8)) & 255, "uint8")),
                "float32",
            )
            result = T.ieee_fmaf(activation[j + 4], scale * quant, result)
        return result

    return product


def vector_tile(shape, encoding, fallback):
    """Decode several coefficients per lane, sharing embedded block metadata.

    Original bytes remain the only global weight representation. F32 scale
    arithmetic and one final F16 rounding match the scalar tile reader.
    """
    import tilelang.language as T
    from math import prod
    from .definition import ENCODINGS

    rows, depth = prod(shape[:-1]), shape[-1]
    _, block, size = ENCODINGS[encoding]
    row_bytes = depth // block * size

    @T.macro
    def word(w, offset):
        halves = T.meta_var(T.view(w, (rows * row_bytes // 2,), dtype="uint16"))
        return (
            T.cast(halves[offset // 2], "uint32")
            | T.cast(halves[offset // 2 + 1], "uint32") << 16
        )

    @T.macro
    def half(w, offset):
        halves = T.meta_var(T.view(w, (rows * row_bytes // 2,), dtype="uint16"))
        return T.cast(T.reinterpret("float16", halves[offset // 2]), "float32")

    @T.macro
    def load(encoded, destination, row, column):
        if (
            encoding == 12
            and destination.dtype == "float16"
            and destination.shape[1] % 64 == 0
        ):
            for i, j in T.Parallel(destination.shape[0], destination.shape[1] // 8):
                local = j // 8 * 64 + j % 8 * 4
                channel = column + local
                if row + i < rows and channel + 35 < depth:
                    base = (row + i) * row_bytes + channel // 256 * 144
                    group = channel % 256 // 64 * 2
                    words = T.meta_var(
                        T.view(encoded[0], (rows * row_bytes // 4,), dtype="uint32")
                    )
                    packed = words[
                        (base + 16 + channel % 256 // 64 * 32 + channel % 32) // 4
                    ]
                    scales = words[base // 4]
                    ds = T.cast(
                        T.reinterpret("float16", T.cast(scales & 65535, "uint16")),
                        "float32",
                    )
                    dm = T.cast(
                        T.reinterpret("float16", T.cast(scales >> 16, "uint16")),
                        "float32",
                    )
                    sa, sb, sc = (
                        words[base // 4 + 1],
                        words[base // 4 + 2],
                        words[base // 4 + 3],
                    )
                    for part in T.unroll(2):
                        g = group + part
                        a = sa >> (8 * (g % 4)) & 255
                        b = sb >> (8 * (g % 4)) & 255
                        cmeta = sc >> (8 * (g % 4)) & 255
                        scale = T.if_then_else(
                            g < 4,
                            a & 63,
                            cmeta & 15 | a >> 6 << 4,
                        )
                        minimum = T.if_then_else(
                            g < 4,
                            b & 63,
                            cmeta >> 4 | b >> 6 << 4,
                        )
                        factor = ds * T.cast(scale, "float32")
                        offset = dm * T.cast(minimum, "float32")
                        for c in T.unroll(4):
                            quant = T.cast(packed >> (8 * c + 4 * part) & 15, "float32")
                            destination[i, local + part * 32 + c] = (
                                factor * quant - offset
                            )
                else:
                    for part in T.unroll(2):
                        for c in T.unroll(4):
                            destination[i, local + part * 32 + c] = 0
        elif (
            encoding == 14
            and destination.dtype == "float16"
            and destination.shape[1] % 128 == 0
        ):
            for i, j in T.Parallel(destination.shape[0], destination.shape[1] // 16):
                local = j // 8 * 128 + j % 8 * 4
                channel = column + local
                if row + i < rows and channel + 99 < depth:
                    base = (row + i) * row_bytes + channel // 256 * 210
                    half_block = channel % 256 // 128
                    lo0 = word(encoded[0], base + half_block * 64 + channel % 32)
                    lo1 = word(encoded[0], base + half_block * 64 + 32 + channel % 32)
                    high = word(encoded[0], base + 128 + half_block * 32 + channel % 32)
                    ds = half(encoded[0], base + 208)
                    for group in T.unroll(4):
                        low = T.if_then_else(group % 2 == 0, lo0, lo1)
                        signed_scale = T.cast(
                            T.reinterpret(
                                "int8",
                                encoded[0][
                                    base
                                    + 192
                                    + half_block * 8
                                    + group * 2
                                    + channel % 32 // 16
                                ],
                            ),
                            "float32",
                        )
                        factor = ds * signed_scale
                        for c in T.unroll(4):
                            quant = T.cast(
                                low >> (8 * c + group // 2 * 4) & 15, "int32"
                            ) | T.cast((high >> (8 * c + group * 2) & 3) << 4, "int32")
                            destination[i, local + group * 32 + c] = factor * T.cast(
                                quant - 32, "float32"
                            )
                else:
                    for group in T.unroll(4):
                        for c in T.unroll(4):
                            destination[i, local + group * 32 + c] = 0
        elif (
            encoding == 8
            and destination.dtype == "float16"
            and destination.shape[1] % 32 == 0
        ):
            for i, j in T.Parallel(destination.shape[0], destination.shape[1] // 8):
                local, channel = j * 8, column + j * 8
                if row + i < rows and channel + 7 < depth:
                    base = (row + i) * row_bytes + channel // 32 * 34
                    scale = half(encoded[0], base)
                    low = word(encoded[0], base + 2 + channel % 32)
                    high = word(encoded[0], base + 6 + channel % 32)
                    for part in T.unroll(2):
                        packed = T.if_then_else(part == 0, low, high)
                        for c in T.unroll(4):
                            quant = T.cast(
                                T.reinterpret(
                                    "int8", T.cast(packed >> (8 * c) & 255, "uint8")
                                ),
                                "float32",
                            )
                            destination[i, local + part * 4 + c] = scale * quant
                else:
                    for c in T.unroll(8):
                        destination[i, local + c] = 0
        else:
            fallback(encoded, destination, row, column)

    return load
