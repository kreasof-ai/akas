"""Storage codec implementation and independent arithmetic."""


def half_bits():
    import tilelang.language as T

    @T.macro
    def unpack(bits):
        return T.call_extern("float32", "tensor_unpack_f16", T.cast(bits, "uint32"))

    return unpack


def round_half():
    """Exact ties-to-even FP16 rounding, including FP16 subnormals.

    Signed exponent comparisons avoid costly unsigned division in WGSL lowering.
    This operation is shared by prefill operand staging and cache stores.
    """
    import tilelang.language as T

    @T.macro
    def rounded(value):
        bits = T.reinterpret("uint32", value)
        exponent = T.cast(bits >> 23 & 255, "int32")
        shift = T.min(T.max(126 - exponent, 1), 24)
        mantissa = bits & 8388607 | 8388608
        rounding = (
            mantissa + (T.uint32(1) << shift - 1) - 1 + (mantissa >> shift & 1) >> shift
        )
        quantum = T.reinterpret("float32", T.uint32(103 << 23))
        small = T.if_then_else(
            exponent < 102, 0.0, T.cast(rounding, "float32") * quantum
        ) * T.if_then_else(bits >> 31 != 0, -1.0, 1.0)
        normal = T.reinterpret(
            "float32", bits + 4095 + (bits >> 13 & 1) & T.uint32(4294959104)
        )
        return T.if_then_else(exponent < 113, small, normal)

    return rounded


def round_bfloat():
    """Round finite F32 inputs to BF16 while keeping an F32 shader value."""
    import tilelang.language as T

    @T.macro
    def rounded(value):
        bits = T.reinterpret("uint32", value)
        return T.reinterpret(
            "float32", (bits + 32767 + (bits >> 16 & 1)) & T.uint32(4294901760)
        )

    return rounded


def round_half_branch():
    """Keep exact half rounding while bypassing subnormal work for normal data.

    In particular, do not replace this with pack2x16float: some Vulkan drivers
    truncate that conversion. Both branches return an exactly representable
    half value before the packed store, including ties and subnormals.
    """
    import tilelang.language as T

    fallback = round_half()

    @T.macro
    def rounded(value):
        bits = T.reinterpret("uint32", value)
        exponent = T.cast(bits >> 23 & 255, "int32")
        result = T.alloc_var("float32")
        if exponent >= 113:
            result = T.reinterpret(
                "float32", bits + 4095 + (bits >> 13 & 1) & T.uint32(4294959104)
            )
        elif exponent < 102:
            result = T.if_then_else(bits >> 31 != 0, -T.float32(0), T.float32(0))
        else:
            result = fallback(value)
        return result

    return rounded
