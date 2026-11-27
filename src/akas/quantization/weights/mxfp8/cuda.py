"""CUDA decodes four MXFP8 coefficients per aligned word load."""

from .definition import IMPLEMENTATIONS as IMPLEMENTATIONS
from .kernel import bind as portable, dequantize as dequantize

_PAIR_CUDA_SOURCE = r"""
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ unsigned int akas_mxfp8_bf16_pair(
    unsigned int word, float scale) {
    const __half2 values = static_cast<__half2>(
        __nv_cvt_fp8x2_to_halfraw2(static_cast<unsigned short>(word), __NV_E4M3)
    );
    const float2 v = __half22float2(values);
    const __nv_bfloat162_raw pair = __floats2bfloat162_rn(v.x * scale, v.y * scale);
    return static_cast<unsigned int>(pair.x) | (static_cast<unsigned int>(pair.y) << 16);
}
"""

CAPABILITIES = ("read", "read_tile", "dot_float", "stage_tile")


def bind(shape):
    import tilelang.language as T

    from akas.quantization.formats import Codec
    from akas.quantization.weights.cuda import aligned_pair_tile

    fallback = portable(shape)
    experts, rows, depth = shape
    fallback_tile = fallback.require("read_tile")

    @T.macro
    def read_tile(encoded, tile, row, column):
        if tile.shape[1] % 4 == 0 and column % 4 == 0:
            # The declared storage and these views have identical byte counts.
            words = T.meta_var(
                T.Tensor((experts * rows, depth // 4), "uint32", encoded[0].data)
            )
            scales = T.meta_var(
                T.Tensor((experts * rows, depth // 32), "uint8", encoded[1].data)
            )
            for i, j in T.Parallel(tile.shape[0], tile.shape[1] // 4):
                out_row = row + i
                k = column + j * 4
                if out_row < experts * rows and k < depth:
                    packed = words[out_row, k // 4]
                    exponent = T.cast(scales[out_row, k // 32], "int32")
                    # The smallest scale is 2^-127, an IEEE FP32 subnormal.
                    scale_bits = T.if_then_else(
                        exponent == 0,
                        T.uint32(0x00400000),
                        T.cast(exponent, "uint32") << 23,
                    )
                    scale = T.reinterpret("float32", scale_bits)
                    for lane in T.unroll(4):
                        code = T.cast(packed >> (8 * lane), "uint8")
                        coefficient = T.cast(
                            T.reinterpret("float8_e4m3fn", code), "float32"
                        )
                        tile[i, j * 4 + lane] = coefficient * scale
                else:
                    for lane in T.unroll(4):
                        tile[i, j * 4 + lane] = 0
        else:
            fallback_tile(encoded, tile, row, column)

    @T.macro
    def dot_float(encoded, x, offset, tile, lane, row, previous):
        words = T.meta_var(
            T.Tensor((experts * rows, depth // 4), "uint32", encoded[0].data)
        )
        scales = T.meta_var(
            T.Tensor((experts * rows, depth // 32), "uint8", encoded[1].data)
        )
        channel = (tile * 32 + lane) * 8
        acc = T.alloc_var("float32")
        acc = previous
        if channel < depth:
            exponent = T.cast(scales[row, channel // 32], "int32")
            scale_bits = T.if_then_else(
                exponent == 0,
                T.uint32(0x00400000),
                T.cast(exponent, "uint32") << 23,
            )
            scale = T.reinterpret("float32", scale_bits)
            for word in T.unroll(2):
                packed = words[row, channel // 4 + word]
                for i in T.unroll(4):
                    code = T.cast(packed >> (8 * i), "uint8")
                    coefficient = T.cast(
                        T.reinterpret("float8_e4m3fn", code), "float32"
                    )
                    coefficient_bf16 = T.cast(
                        T.cast(coefficient * scale, "bfloat16"), "float32"
                    )
                    acc = T.ieee_fmaf(
                        T.cast(x[offset + channel + word * 4 + i], "float32"),
                        coefficient_bf16,
                        acc,
                    )
        return acc

    @T.macro
    def decode_scale(bits):
        exponent = T.cast(bits, "int32")
        return T.reinterpret(
            "float32",
            T.if_then_else(
                exponent == 0, T.uint32(0x00400000), T.cast(exponent, "uint32") << 23
            ),
        )

    @T.macro
    def decode_pair(word, scale, outer):
        T.import_source(_PAIR_CUDA_SOURCE)
        return T.call_extern("uint32", "akas_mxfp8_bf16_pair", word, scale)

    return Codec(
        fallback.storage,
        fallback.read,
        read_tile,
        dot_float=dot_float,
        stage_tile=aligned_pair_tile(shape, 4, 32, decode_scale, decode_pair),
    )
