"""Portable packed GGML word loads and exact half rounding macros."""

def half_bits():
    import tilelang.language as T

    @T.macro
    def unpack(bits):
        return T.call_extern("float32", "tensor_unpack_f16", T.cast(bits, "uint32"))

    return unpack


def packed_loads():
    import tilelang.language as T

    unpack = half_bits()

    @T.macro
    def byte(w, address):
        return (w[address // 4] >> (address % 4 * 8)) & T.uint32(255)

    @T.macro
    def half(w, address):
        return unpack((w[address // 4] >> (address % 4 * 8)) & T.uint32(65535))

    @T.macro
    def word(w, address):
        # The 18-byte GGML Q4 blocks alternate between 2- and 4-byte alignment.
        return T.call_extern(
            "uint32",
            "tensor_byte_align_u32",
            w[address // 4],
            w[(address + 3) // 4],
            T.cast(address % 4, "uint32"),
        )

    return byte, half, word


def round_half():
    """Exact ties-to-even FP16 rounding, including FP16 subnormals.

    Signed exponent comparisons avoid costly unsigned division in WGSL lowering.
    This operation is shared by prefill operand staging and cache stores.
    """
    import tilelang.language as T

    @T.macro
    def rounded(value):
        bits = T.reinterpret("uint32", value)
        exponent = T.cast((bits >> 23) & 255, "int32")
        shift = T.min(T.max(126 - exponent, 1), 24)
        mantissa = (bits & 8388607) | 8388608
        rounding = (
            mantissa + (T.uint32(1) << (shift - 1)) - 1 + ((mantissa >> shift) & 1)
        ) >> shift
        quantum = T.reinterpret("float32", T.uint32(103 << 23))
        small = T.if_then_else(
            exponent < 102, 0.0, T.cast(rounding, "float32") * quantum
        ) * T.if_then_else((bits >> 31) != 0, -1.0, 1.0)
        normal = T.reinterpret(
            "float32", (bits + 4095 + ((bits >> 13) & 1)) & T.uint32(4294959104)
        )
        return T.if_then_else(exponent < 113, small, normal)

    return rounded
