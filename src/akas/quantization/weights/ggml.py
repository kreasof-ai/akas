"""GGML weight storage, scalar/packed decoding and tuning samples."""

import numpy as np
from akas.checkpoints.gguf import TYPES


def weight_storage(kind, elements, *, packed_words=False):
    _, block, size = TYPES[kind]
    if kind in (0, 1):
        return elements, "float16" if kind == 1 else "float32"
    count = elements // block * size
    if packed_words:
        if count % 4:
            raise ValueError("WebGPU packed matrices must align to four bytes")
        return count // 4, "uint32"
    return count, "uint8"


def weight_decoder(kind, depth, *, packed_words=False):
    """Inline the exact GGML scalar decoder into a caller's TIRx function."""
    import tilelang.language as T

    _, block, size = TYPES[kind]

    @T.macro
    def byte(w, base, offset):
        if packed_words:
            return (w[(base + offset) // 4] >> ((base + offset) % 4 * 8)) & T.uint32(
                255
            )
        else:
            return T.cast(w[base + offset], "uint32")

    @T.macro
    def half(w, base, offset):
        bits = byte(w, base, offset) | (byte(w, base, offset + 1) << 8)
        if packed_words:
            return T.call_extern("float32", "tensor_unpack_f16", bits)
        else:
            return T.cast(T.reinterpret("float16", T.cast(bits, "uint16")), "float32")

    @T.macro
    def signed(value):
        if packed_words:
            return T.cast(
                T.cast(value, "int32")
                - T.if_then_else(T.cast(value, "int32") >= 128, 256, 0),
                "float32",
            )
        else:
            return T.cast(T.reinterpret("int8", T.cast(value, "uint8")), "float32")

    @T.macro
    def read_weight(w, row, col):
        if kind in (0, 1):
            return T.cast(w[row * depth + col], "float32")
        else:
            base = (row * depth + col) // block * size
            j = col % block
            if kind == 2:
                return half(w, base, 0) * (
                    T.cast(
                        (byte(w, base, 2 + j % 16) >> (4 * (j // 16))) & 15, "float32"
                    )
                    - 8
                )
            elif kind == 8:
                return half(w, base, 0) * signed(byte(w, base, 2 + j))
            elif kind == 12:
                group = j // 32
                scale = T.if_then_else(
                    group < 4,
                    byte(w, base, 4 + group) & 63,
                    (byte(w, base, 8 + group) & 15)
                    | ((byte(w, base, group) >> 6) << 4),
                )
                minimum = T.if_then_else(
                    group < 4,
                    byte(w, base, 8 + group) & 63,
                    (byte(w, base, 8 + group) >> 4)
                    | ((byte(w, base, 4 + group) >> 6) << 4),
                )
                quant = (
                    byte(w, base, 16 + j // 64 * 32 + j % 32) >> (4 * (group % 2))
                ) & 15
                return half(w, base, 0) * T.cast(scale, "float32") * T.cast(
                    quant, "float32"
                ) - half(w, base, 2) * T.cast(minimum, "float32")
            elif kind == 14:
                group = j % 128 // 32
                low = byte(w, base, j // 128 * 64 + group % 2 * 32 + j % 32)
                high = byte(w, base, 128 + j // 128 * 32 + j % 32)
                quant = ((low >> (4 * (group // 2))) & 15) | (
                    ((high >> (2 * group)) & 3) << 4
                )
                scale = byte(w, base, 192 + j // 16)
                return (
                    half(w, base, 208) * signed(scale) * (T.cast(quant, "float32") - 32)
                )
            else:
                raise ValueError("unsupported GGML encoding")

    return read_weight


def hardware_loads():
    import tilelang.language as T

    @T.macro
    def u16(w, offset):
        return T.call_extern("uint32", "tensor_load_u16", T.address_of(w[offset]))

    @T.macro
    def word(w, offset):
        return u16(w, offset) | (u16(w, offset + 2) << 16)

    @T.macro
    def half(w, offset):
        return T.cast(
            T.reinterpret("float16", T.cast(u16(w, offset), "uint16")), "float32"
        )

    return u16, word, half


def packed_dot(q, k, row_bytes, width=4):
    """One lane's packed weight dot product, inlined at each GEMV use."""
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
                    T.cast((packed >> (component * 8)) & 15, "float32") - 8,
                    dot,
                )
            for component in T.unroll(4):
                dot = T.ieee_fmaf(
                    activation_hi[component],
                    T.cast((packed >> (component * 8 + 4)) & 15, "float32") - 8,
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
                    T.cast((packed >> (component * 8)) & 15, "float32"),
                    lo,
                )
                hi = T.ieee_fmaf(
                    activation_hi[component],
                    T.cast((packed >> (component * 8 + 4)) & 15, "float32"),
                    hi,
                )
            s0 = T.if_then_else(
                group < 4,
                w[base + 4 + group] & 63,
                (w[base + 8 + group] & 15) | ((w[base + group] >> 6) << 4),
            )
            s1 = T.if_then_else(
                group < 4,
                w[base + 5 + group] & 63,
                (w[base + 9 + group] & 15) | ((w[base + 1 + group] >> 6) << 4),
            )
            m0 = T.if_then_else(
                group < 4,
                w[base + 8 + group] & 63,
                (w[base + 8 + group] >> 4) | ((w[base + 4 + group] >> 6) << 4),
            )
            m1 = T.if_then_else(
                group < 4,
                w[base + 9 + group] & 63,
                (w[base + 9 + group] >> 4) | ((w[base + 5 + group] >> 6) << 4),
            )
            sum0 = (activation[0] + activation[1]) + (activation[2] + activation[3])
            sum1 = (activation_hi[0] + activation_hi[1]) + (
                activation_hi[2] + activation_hi[3]
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
                low = word(w, base + half * 64 + (lane // 8 % 2) * 32 + lane % 8 * 4)
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
                        (low >> (component * 8 + lane // 16 * 4)) & 15, "int32"
                    ) | T.cast(
                        ((high >> (component * 8 + lane // 8 * 2)) & 3) << 4, "int32"
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
            lo = scale * (T.cast((packed >> shift) & 15, "float32") - 8)
            hi = scale * (T.cast((packed >> (shift + 8)) & 15, "float32") - 8)
            return T.call_extern("uint32", "tensor_pack_f16x2", lo, hi)
        elif q == 12:
            base = row * row_bytes + col // 256 * 144
            weight_col = col % 256
            group = weight_col // 32
            scale = T.if_then_else(
                group < 4,
                w[base + 4 + group] & 63,
                (w[base + 8 + group] & 15) | ((w[base + group] >> 6) << 4),
            )
            minimum = T.if_then_else(
                group < 4,
                w[base + 8 + group] & 63,
                (w[base + 8 + group] >> 4) | ((w[base + 4 + group] >> 6) << 4),
            )
            ds = half_value(w, base) * T.cast(scale, "float32")
            dm = half_value(w, base + 2) * T.cast(minimum, "float32")
            packed = u16(w, base + 16 + weight_col // 64 * 32 + weight_col % 32)
            shift = group % 2 * 4
            lo = ds * T.cast((packed >> shift) & 15, "float32") - dm
            hi = ds * T.cast((packed >> (shift + 8)) & 15, "float32") - dm
            return T.call_extern("uint32", "tensor_pack_f16x2", lo, hi)
        elif q == 14:
            base = row * row_bytes + col // 256 * 210
            weight_col = col % 256
            group = weight_col % 128 // 32
            low = u16(
                w, base + weight_col // 128 * 64 + group % 2 * 32 + weight_col % 32
            )
            high = u16(w, base + 128 + weight_col // 128 * 32 + weight_col % 32)
            a = T.cast((low >> (group // 2 * 4)) & 15, "int32") | T.cast(
                ((high >> (group * 2)) & 3) << 4, "int32"
            )
            b = T.cast((low >> (group // 2 * 4 + 8)) & 15, "int32") | T.cast(
                ((high >> (group * 2 + 8)) & 3) << 4, "int32"
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


def sample_weight(storage, kind, *, seed=31):
    """Fill resident bytes, including signed payloads and finite block scales."""
    from akas.checkpoints.gguf import TYPES, dequantize

    rng = np.random.default_rng(seed)
    if kind in (0, 1):
        storage[:] = rng.normal(0, 0.15, storage.shape)
        return storage.astype("float32").reshape(-1)
    raw = storage.view("uint8").reshape(-1)
    raw[:] = rng.integers(0, 256, raw.shape, dtype="uint8")
    blocks = raw.reshape(-1, TYPES[kind][2])
    scale_offset = 208 if kind == 14 else 0
    scales = rng.uniform(0.001, 0.004, (len(blocks), 1)).astype("float16")
    blocks[:, scale_offset:scale_offset + 2] = scales.view("uint8")
    if kind == 12:
        minimum = rng.uniform(0.001, 0.003, (len(blocks), 1)).astype("float16")
        blocks[:, 2:4] = minimum.view("uint8")
    return dequantize(raw, kind)
