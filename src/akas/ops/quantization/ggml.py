"""Packed GGML weights and baseline inference operators.

Factories are producer-only; consumers execute Tensor artifacts."""

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
