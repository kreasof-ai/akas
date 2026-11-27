"""CUDA decodes aligned NVFP4 words into matrix operand tiles."""

from .definition import IMPLEMENTATIONS as IMPLEMENTATIONS
from .kernel import bind as portable, dequantize as dequantize

CAPABILITIES = ("read", "read_tile", "dot_float", "stage_tile")


def bind(shape):
    import tilelang.language as T

    from akas.memory.state import BufferSpec
    from akas.quantization.formats import Codec
    from akas.quantization.weights.cuda import aligned_pair_tile

    fallback = portable(shape)
    experts, rows, depth = shape
    fallback_tile = fallback.require("read_tile")

    @T.macro
    def read_tile(encoded, tile, row, column):
        expert = row // rows
        if (
            tile.shape[1] % 8 == 0
            and column % 8 == 0
            and expert < experts
            and (row + tile.shape[0] - 1) // rows == expert
        ):
            # The declared storage and these views have identical byte counts.
            words = T.meta_var(
                T.Tensor((experts * rows, depth // 8), "uint32", encoded[0].data)
            )
            scales = T.meta_var(
                T.Tensor((experts * rows, depth // 16), "uint8", encoded[1].data)
            )
            outer = T.alloc_var("float32")
            outer = encoded[2][expert] / encoded[3][0]
            for i, j in T.Parallel(tile.shape[0], tile.shape[1] // 8):
                out_row = row + i
                k = column + j * 8
                if k < depth:
                    packed = words[out_row, k // 8]
                    scale = T.cast(
                        T.reinterpret("float8_e4m3fn", scales[out_row, k // 16]),
                        "float32",
                    )
                    for lane in T.unroll(8):
                        nibble = T.cast((packed >> (4 * lane)) & 15, "int32")
                        # Each nibble stores twice the corresponding E2M1 magnitude.
                        magnitude = (
                            T.cast(
                                (T.uint32(0xC8643210) >> (4 * (nibble & 7))) & 15,
                                "float32",
                            )
                            * 0.5
                        )
                        value = T.if_then_else(nibble < 8, magnitude, -magnitude)
                        tile[i, j * 8 + lane] = (value * scale) * outer
                else:
                    for lane in T.unroll(8):
                        tile[i, j * 8 + lane] = 0
        else:
            fallback_tile(encoded, tile, row, column)

    @T.macro
    def dot_float(encoded, x, offset, tile, lane, row, previous):
        words = T.meta_var(
            T.Tensor((experts * rows, depth // 8), "uint32", encoded[0].data)
        )
        scales = T.meta_var(
            T.Tensor((experts * rows, depth // 16), "uint8", encoded[1].data)
        )
        channel = (tile * 32 + lane) * 8
        acc = T.alloc_var("float32")
        acc = previous
        if channel < depth:
            packed = words[row, channel // 8]
            scale = T.cast(
                T.reinterpret("float8_e4m3fn", scales[row, channel // 16]),
                "float32",
            )
            outer = encoded[2][row // rows] / encoded[3][0]
            for i in T.unroll(8):
                nibble = T.cast((packed >> (4 * i)) & 15, "int32")
                magnitude = (
                    T.cast(
                        (T.uint32(0xC8643210) >> (4 * (nibble & 7))) & 15,
                        "float32",
                    )
                    * 0.5
                )
                signed_value = T.if_then_else(nibble < 8, magnitude, -magnitude)
                coefficient = T.cast(
                    T.cast((signed_value * scale) * outer, "bfloat16"), "float32"
                )
                acc = T.ieee_fmaf(
                    T.cast(x[offset + channel + i], "float32"), coefficient, acc
                )
        return acc

    @T.macro
    def decode_scale(bits):
        return bits

    @T.macro
    def prepare(encoded, expert, cache):
        outer = encoded[2][expert] / encoded[3][0]
        for i, j in T.Parallel(128, 8):
            scale = T.cast(
                T.reinterpret("float8_e4m3fn", T.cast(i, "uint8")), "float32"
            )
            magnitude = T.cast((T.uint32(0xC8643210) >> (4 * j)) & 15, "float32") * 0.5
            cache[i, j] = T.reinterpret(
                "uint16", T.cast((magnitude * scale) * outer, "bfloat16")
            )

    @T.macro
    def decode_pair(word, scale, cache):
        index = T.cast(scale, "uint32")
        signs = ((word & 8) << 12) | ((word & 128) << 24)
        scale_sign = T.if_then_else(index & 128 != 0, T.uint32(0x80008000), T.uint32(0))
        lo = T.cast(cache[index & 127, word & 7], "uint32")
        hi = T.cast(cache[index & 127, (word >> 4) & 7], "uint32")
        return (lo | (hi << 16)) ^ signs ^ scale_sign

    return Codec(
        fallback.storage,
        fallback.read,
        read_tile,
        dot_float=dot_float,
        stage_tile=aligned_pair_tile(
            shape,
            8,
            16,
            decode_scale,
            decode_pair,
            lookup=(BufferSpec((128, 8), "uint16"), prepare),
        ),
    )
