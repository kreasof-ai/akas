"""E4M3 software codec; byte payloads on CUDA, four packed bytes on WebGPU.

The reference ABI consumes F32 projections, rounded to BF16 before encoding,
matching the CUDA codec. Partial 128-channel scale blocks are zero padded.
"""


def helpers():
    import tilelang.language as T

    @T.macro
    def rounded(x):
        bits = T.reinterpret("uint32", x)
        return T.reinterpret(
            "float32",
            (bits + T.uint32(32767) + ((bits >> 16) & 1)) & T.uint32(4294901760),
        )

    @T.macro
    def encode(x):
        bits = T.reinterpret("uint32", x)
        magnitude = T.abs(x)
        rounded_bits = (
            (bits & T.uint32(2147483647)) + T.uint32(524287) + ((bits >> 20) & 1)
        )
        normal = T.cast((rounded_bits >> 20), "int32") - 960
        subnormal = T.cast(T.round(magnitude * 512), "int32")
        return T.cast(
            T.min(
                T.max(T.if_then_else(magnitude < 0.015625, subnormal, normal), 0), 126
            ),
            "uint32",
        ) | ((bits >> 24) & 128)

    @T.macro
    def decode(bits):
        exponent = T.cast((bits >> 3) & 15, "int32")
        fraction = bits & 7
        magnitude = T.if_then_else(
            exponent == 0,
            T.cast(fraction, "float32") / 512,
            (1 + T.cast(fraction, "float32") / 8)
            * T.exp2(T.cast(exponent, "float32") - 7),
        )
        return T.if_then_else((bits & 128) != 0, -magnitude, magnitude)

    return rounded, encode, decode


def reference_writer(head_dim, *, provider="cuda"):
    import tilelang.language as T
    from akas.ops.reductions.webgpu import tree_reduce

    d = head_dim
    blocks = (d + 127) // 128
    packed = provider == "webgpu"
    rounded, encode, _ = helpers()
    maximum = tree_reduce(128, "max")

    @T.macro
    def write(keys, values, slot, head, position, kc, vc, ks, vs, tx, enabled):
        scratch_k = T.alloc_shared((128,), "float32")
        scratch_v = T.alloc_shared((128,), "float32")
        scale_k = T.alloc_shared((1,), "float32")
        scale_v = T.alloc_shared((1,), "float32")
        for block in T.serial(blocks):
            scratch_k[tx] = T.if_then_else(
                block * 128 + tx < d,
                T.abs(rounded(keys[slot, head * d + block * 128 + tx])),
                0,
            )
            scratch_v[tx] = T.if_then_else(
                block * 128 + tx < d,
                T.abs(rounded(values[slot, head * d + block * 128 + tx])),
                0,
            )
            T.sync_threads()
            maximum(scratch_k, tx)
            maximum(scratch_v, tx)
            if tx == 0:
                scale_k[0] = T.max(scratch_k[0], T.float32(1e-12)) / 448
                scale_v[0] = T.max(scratch_v[0], T.float32(1e-12)) / 448
                if enabled[slot] != 0:
                    ks[slot, head, position, block] = scale_k[0]
                    vs[slot, head, position, block] = scale_v[0]
            T.sync_threads()
            if packed:
                if tx < 32 and block * 128 + tx * 4 < d and enabled[slot] != 0:
                    word_k = T.alloc_var("uint32")
                    word_v = T.alloc_var("uint32")
                    word_k = 0
                    word_v = 0
                    for i in T.serial(4):
                        if block * 128 + tx * 4 + i < d:
                            word_k = word_k | (
                                encode(
                                    rounded(
                                        keys[slot, head * d + block * 128 + tx * 4 + i]
                                    )
                                    / scale_k[0]
                                )
                                << (i * 8)
                            )
                            word_v = word_v | (
                                encode(
                                    rounded(
                                        values[
                                            slot, head * d + block * 128 + tx * 4 + i
                                        ]
                                    )
                                    / scale_v[0]
                                )
                                << (i * 8)
                            )
                    kc[slot, head, position, block * 32 + tx] = word_k
                    vc[slot, head, position, block * 32 + tx] = word_v
            else:
                if block * 128 + tx < d and enabled[slot] != 0:
                    kc[slot, head, position, block * 128 + tx] = T.cast(
                        encode(
                            rounded(keys[slot, head * d + block * 128 + tx])
                            / scale_k[0]
                        ),
                        "uint8",
                    )
                    vc[slot, head, position, block * 128 + tx] = T.cast(
                        encode(
                            rounded(values[slot, head * d + block * 128 + tx])
                            / scale_v[0]
                        ),
                        "uint8",
                    )
            T.sync_threads()

    return write


def reference_decoder(*, provider="cuda"):
    import tilelang.language as T

    _, _, decode = helpers()
    packed = provider == "webgpu"

    @T.macro
    def read(cache, metadata, slot, head, position, channel):
        if packed:
            bits = (
                cache[slot, head, position, channel // 4] >> ((channel % 4) * 8)
            ) & 255
        else:
            bits = T.cast(cache[slot, head, position, channel], "uint32")
        return decode(bits) * metadata[slot, head, position, channel // 128]

    return read
