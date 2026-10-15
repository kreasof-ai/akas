"""Storage codec implementation and independent arithmetic."""


def decode_storage(bits):
    import numpy as np

    bits = np.asarray(bits, dtype="uint8")
    exponent = (bits >> 3 & 15).astype("int32")
    fraction = (bits & 7).astype("float32")
    magnitude = np.where(
        exponent == 0, fraction / 512, (1 + fraction / 8) * np.exp2(exponent - 7)
    ).astype("float32")
    return np.where(bits & 128, -magnitude, magnitude).astype("float32")


def encode_storage(values):
    """Nearest-even E4M3 with finite saturation after explicit F32 rounding."""
    import numpy as np

    values = np.asarray(values, dtype="float32")
    bits = values.view("uint32")
    magnitude = np.abs(values)
    rounded = (bits & np.uint32(2147483647)) + np.uint32(524287) + (bits >> 20 & 1)
    normal = (rounded >> 20).astype("int32") - 960
    subnormal = np.rint(magnitude * np.float32(512)).astype("int32")
    code = np.clip(
        np.where(magnitude < np.float32(0.015625), subnormal, normal), 0, 126
    )
    return code.astype("uint8") | (bits >> 24 & 128).astype("uint8")


def helpers():
    import tilelang.language as T

    @T.macro
    def encode(x):
        bits = T.reinterpret("uint32", x)
        magnitude = T.abs(x)
        rounded_bits = (
            (bits & T.uint32(2147483647)) + T.uint32(524287) + (bits >> 20 & 1)
        )
        normal = T.cast(rounded_bits >> 20, "int32") - 960
        subnormal = T.cast(T.round(magnitude * 512), "int32")
        return (
            T.cast(
                T.min(
                    T.max(T.if_then_else(magnitude < 0.015625, subnormal, normal), 0),
                    126,
                ),
                "uint32",
            )
            | bits >> 24 & 128
        )

    @T.macro
    def decode(bits):
        exponent = T.cast(bits >> 3 & 15, "int32")
        fraction = bits & 7
        magnitude = T.if_then_else(
            exponent == 0,
            T.cast(fraction, "float32") / 512,
            (1 + T.cast(fraction, "float32") / 8)
            * T.exp2(T.cast(exponent, "float32") - 7),
        )
        return T.if_then_else(bits & 128 != 0, -magnitude, magnitude)

    return (encode, decode)


def portable_scaling():
    """Deterministic F32/scale arithmetic without approximate shader division.

    Division by 448 reduces to integer division by seven and a power of two.
    Twenty-four-bit significands fit the calculation in U32. Encoding compares
    the exact ratio to E4M3
    midpoints, including the F32 rounding interval before the final FP8 rounding.
    """
    import tilelang.language as T

    @T.macro
    def scale(maximum):
        bits = T.alloc_var("uint32")
        significand = T.alloc_var("uint32")
        power = T.alloc_var("int32")
        numerator = T.alloc_var("uint32")
        mantissa = T.alloc_var("uint32")
        exponent = T.alloc_var("int32")
        result = T.alloc_var("uint32")
        bits = T.reinterpret("uint32", maximum)
        significand = bits & 8388607 | 8388608
        power = T.if_then_else(significand >= 14680064, 2, 3)
        numerator = significand << power
        mantissa = (numerator + 3) // 7
        exponent = T.max(T.cast(bits >> 23 & 255, "int32") - power - 6, 0)
        result = T.cast(exponent, "uint32") << 23 | mantissa - 8388608
        return T.reinterpret(
            "float32", T.if_then_else(bits <= 730643660, T.uint32(656463813), result)
        )

    @T.macro
    def encode_scaled(value, factor):
        bits = T.alloc_var("uint32")
        factor_bits = T.alloc_var("uint32")
        significand = T.alloc_var("uint32")
        denominator = T.alloc_var("uint32")
        difference = T.alloc_var("int32")
        exponent = T.alloc_var("int32")
        normal = T.alloc_var("int32")
        shift = T.alloc_var("int32")
        numerator = T.alloc_var("uint32")
        quotient = T.alloc_var("uint32")
        remainder = T.alloc_var("uint32")
        delta = T.alloc_var("int32")
        ulp_shift = T.alloc_var("int32")
        distance = T.alloc_var("uint32")
        halfway = T.alloc_var("int32")
        up = T.alloc_var("int32")
        nearest = T.alloc_var("int32")
        code = T.alloc_var("int32")
        bits = T.reinterpret("uint32", value)
        factor_bits = T.reinterpret("uint32", factor)
        significand = bits & 8388607 | 8388608
        denominator = factor_bits & 8388607 | 8388608
        difference = T.cast(bits >> 23 & 255, "int32") - T.cast(
            factor_bits >> 23 & 255, "int32"
        )
        exponent = difference - T.if_then_else(significand < denominator, 1, 0)
        normal = T.if_then_else(exponent >= -6, 1, 0)
        shift = T.if_then_else(normal != 0, difference + 3 - exponent, difference + 9)
        numerator = T.if_then_else(
            shift >= 0, significand << T.min(T.max(shift, 0), 4), T.uint32(0)
        )
        quotient = numerator // denominator
        remainder = numerator - quotient * denominator
        delta = T.if_then_else(
            shift == -1,
            T.cast(significand, "int32") - T.cast(denominator, "int32"),
            T.cast(remainder * 2, "int32") - T.cast(denominator, "int32"),
        )
        ulp_shift = T.if_then_else(
            normal != 0,
            20,
            T.if_then_else(
                quotient >= 4,
                21,
                T.if_then_else(
                    quotient >= 2, 22, T.if_then_else(quotient >= 1, 23, 24)
                ),
            ),
        )
        distance = T.cast(T.abs(delta), "uint32")
        halfway = T.if_then_else(
            (distance <= 16) & (distance << ulp_shift <= denominator), 1, 0
        )
        up = T.if_then_else(
            halfway != 0,
            T.if_then_else(quotient & 1 != 0, 1, 0),
            T.if_then_else(delta > 0, 1, 0),
        )
        nearest = T.cast(quotient, "int32") + up
        code = T.if_then_else(normal != 0, (exponent + 7) * 8 + nearest - 8, nearest)
        return T.cast(T.min(T.max(code, 0), 126), "uint32") | bits >> 24 & 128

    return (scale, encode_scaled)
