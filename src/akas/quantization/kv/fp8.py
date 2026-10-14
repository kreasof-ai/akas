"""Block-128 E4M3 storage and numerical policies for KV caches.

Native BF16 writers and portable F32 writers preserve their declared rounding.
Both use the same format declaration and codec interface.
"""

from functools import partial
from types import SimpleNamespace

from akas.memory.state import BufferSpec
from akas.quantization.weights.block_fp8 import decode_storage, encode_storage
from akas.quantization.kv.codec import Format, Storage, get

BLOCK = 128
PACKED_DTYPE = "float8_e4m3fn"

def layout(shape):
    scales = shape[:-1] + ((shape[-1] + BLOCK - 1) // BLOCK,)
    return Storage(BufferSpec(shape, "uint8"), (BufferSpec(scales, "float32"),))


def webgpu_layout(shape):
    original = layout(shape)
    return Storage(
        BufferSpec((*shape[:-1], (shape[-1] + 3) // 4), "uint32"), original.metadata
    )


def writer(head_dim, *, value_offset=0):
    import tilelang.language as T

    d = head_dim
    if d % BLOCK:
        raise ValueError("FP8 KV writer currently requires a multiple of 128 channels")
    blocks = d // BLOCK

    @T.macro
    def encode_pair(keys, values, row, head, slot, position, kc, vc, ks, vs):
        ka = T.alloc_fragment((blocks, 128), "float32")
        va = T.alloc_fragment((blocks, 128), "float32")
        km = T.alloc_fragment((blocks,), "float32")
        vm = T.alloc_fragment((blocks,), "float32")
        scales_k = T.alloc_shared((blocks,), "float32")
        scales_v = T.alloc_shared((blocks,), "float32")
        for block, j in T.Parallel(blocks, 128):
            ka[block, j] = T.abs(T.cast(keys[block * 128 + j], "float32"))
            va[block, j] = T.abs(
                T.cast(
                    T.cast(
                        values[row, value_offset + (head * d + block * 128 + j)],
                        "bfloat16",
                    ),
                    "float32",
                )
            )
        T.reduce_max(ka, km, dim=1)
        T.reduce_max(va, vm, dim=1)
        for block in T.Parallel(blocks):
            scales_k[block] = T.max(km[block], T.float32(1e-12)) / 448
            scales_v[block] = T.max(vm[block], T.float32(1e-12)) / 448
            ks[slot, head, position, block] = scales_k[block]
            vs[slot, head, position, block] = scales_v[block]
        T.sync_threads()
        for j in T.Parallel(d):
            quantized_key = T.cast(keys[j], "float32") / scales_k[j // 128]
            quantized_value = (
                T.cast(
                    T.cast(values[row, value_offset + (head * d + j)], "bfloat16"),
                    "float32",
                )
                / scales_v[j // 128]
            )
            kc[slot, head, position, j] = T.cast(
                T.call_extern("uint32", "tensor_encode_e4m3", quantized_key), "uint8"
            )
            vc[slot, head, position, j] = T.cast(
                T.call_extern("uint32", "tensor_encode_e4m3", quantized_value), "uint8"
            )

    return encode_pair


def decoder():
    import tilelang.language as T

    @T.macro
    def read(cache, metadata, slot, head, position, channel):
        return (
            T.call_extern(
                "float32",
                "tensor_decode_e4m3",
                T.cast(cache[slot, head, position, channel], "uint32"),
            )
            * metadata[slot, head, position, channel // BLOCK]
        )

    return read


def packed_decoder():
    import tilelang.language as T

    @T.macro
    def read(encoded, metadata, row, channel):
        return (
            T.cast(encoded[row, channel], "float32") * metadata[row, channel // BLOCK]
        )

    return read


def helpers():
    import tilelang.language as T

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

    return encode, decode


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
        significand = (bits & 8388607) | 8388608
        power = T.if_then_else(significand >= 14680064, 2, 3)
        numerator = significand << power
        mantissa = (numerator + 3) // 7  # odd divisor: there is no halfway tie
        exponent = T.max(T.cast((bits >> 23) & 255, "int32") - power - 6, 0)
        result = (T.cast(exponent, "uint32") << 23) | (mantissa - 8388608)
        # float32(float32(1e-12) / float32(448)), the zero-head scale floor.
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
        significand = (bits & 8388607) | 8388608
        denominator = (factor_bits & 8388607) | 8388608
        difference = T.cast((bits >> 23) & 255, "int32") - T.cast(
            (factor_bits >> 23) & 255, "int32"
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
        # F32 can round a ratio just beside an E4M3 midpoint onto that midpoint.
        # The interval is half an F32 ULP, expressed in integer ratio units.
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
            (distance <= 16) & ((distance << ulp_shift) <= denominator), 1, 0
        )
        up = T.if_then_else(
            halfway != 0,
            T.if_then_else((quotient & 1) != 0, 1, 0),
            T.if_then_else(delta > 0, 1, 0),
        )
        nearest = T.cast(quotient, "int32") + up
        code = T.if_then_else(normal != 0, (exponent + 7) * 8 + nearest - 8, nearest)
        return T.cast(T.min(T.max(code, 0), 126), "uint32") | ((bits >> 24) & 128)

    return scale, encode_scaled


def reference_writer(head_dim, *, provider="cuda", chunk=1):
    import tilelang.language as T
    from akas.ops.reductions.webgpu import tree_reduce

    d = head_dim
    blocks = (d + 127) // 128
    packed = provider == "webgpu"
    encode, _ = helpers()
    portable_scale, encode_scaled = portable_scaling()
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
                T.abs(keys[slot, head * d + block * 128 + tx]),
                0,
            )
            scratch_v[tx] = T.if_then_else(
                block * 128 + tx < d,
                T.abs(values[slot, head * d + block * 128 + tx]),
                0,
            )
            T.sync_threads()
            maximum(scratch_k, tx)
            maximum(scratch_v, tx)
            if tx == 0:
                if packed:
                    scale_k[0] = portable_scale(scratch_k[0])
                    scale_v[0] = portable_scale(scratch_v[0])
                else:
                    scale_k[0] = T.max(scratch_k[0], T.float32(1e-12)) / 448
                    scale_v[0] = T.max(scratch_v[0], T.float32(1e-12)) / 448
                if enabled[slot] != 0:
                    ks[slot // chunk, head, position, block] = scale_k[0]
                    vs[slot // chunk, head, position, block] = scale_v[0]
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
                                encode_scaled(
                                    keys[slot, head * d + block * 128 + tx * 4 + i],
                                    scale_k[0],
                                )
                                << (i * 8)
                            )
                            word_v = word_v | (
                                encode_scaled(
                                    values[slot, head * d + block * 128 + tx * 4 + i],
                                    scale_v[0],
                                )
                                << (i * 8)
                            )
                    kc[slot // chunk, head, position, block * 32 + tx] = word_k
                    vc[slot // chunk, head, position, block * 32 + tx] = word_v
            else:
                if block * 128 + tx < d and enabled[slot] != 0:
                    kc[slot // chunk, head, position, block * 128 + tx] = T.cast(
                        encode(keys[slot, head * d + block * 128 + tx] / scale_k[0]),
                        "uint8",
                    )
                    vc[slot // chunk, head, position, block * 128 + tx] = T.cast(
                        encode(values[slot, head * d + block * 128 + tx] / scale_v[0]),
                        "uint8",
                    )
            T.sync_threads()

    return write


def reference_decoder(*, provider="cuda"):
    import tilelang.language as T

    _, decode = helpers()
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






def make_kernel(kind, p):
    import tilelang.language as T

    slots, heads, d, capacity = (
        p[name] for name in ("slots", "kv_heads", "d", "capacity")
    )
    chunk = p.get("chunk", 1)
    if type(chunk) is not int or chunk <= 0:
        raise ValueError("KV operation requires a positive chunk")
    rows = slots * chunk
    codec = get(p.get("kv_dtype", "fp8"))
    storage = codec.storage((slots, heads, capacity, d))
    if len(storage.metadata) != 1:
        raise ValueError("KV pair operation requires one metadata buffer per cache")
    shape, dtype = storage.values.shape, storage.values.dtype
    metadata_shape, metadata_dtype = (
        storage.metadata[0].shape,
        storage.metadata[0].dtype,
    )
    backend = codec.backend("cuda")
    if kind == "kv_encode":
        encode = backend.writer(d)

        @T.prim_func
        def kernel(
            keys: T.Tensor((rows, heads * d), "bfloat16"),
            values: T.Tensor((rows, heads * d), "float32"),
            positions: T.Tensor((rows,), "int32"),
            active: T.Tensor((rows,), "int32"),
            kc: T.Tensor(shape, dtype),
            vc: T.Tensor(shape, dtype),
            ks: T.Tensor(metadata_shape, metadata_dtype),
            vs: T.Tensor(metadata_shape, metadata_dtype),
        ):
            with T.Kernel(rows, heads, threads=256) as (row, head):
                if active[row] != 0:
                    prepared = T.alloc_shared((d,), "bfloat16")
                    for j in T.Parallel(d):
                        prepared[j] = keys[row, head * d + j]
                    encode(
                        prepared,
                        values,
                        row,
                        head,
                        row // chunk,
                        positions[row],
                        kc,
                        vc,
                        ks,
                        vs,
                    )

        return kernel
    if kind == "kv_decode":
        decode = backend.decoder()

        @T.prim_func
        def kernel(
            kc: T.Tensor(shape, dtype),
            vc: T.Tensor(shape, dtype),
            ks: T.Tensor(metadata_shape, metadata_dtype),
            vs: T.Tensor(metadata_shape, metadata_dtype),
            keys: T.Tensor((slots, heads, capacity, d), "bfloat16"),
            values: T.Tensor((slots, heads, capacity, d), "bfloat16"),
        ):
            with T.Kernel(slots, heads, capacity, threads=256) as (
                slot,
                head,
                position,
            ):
                for j in T.Parallel(d):
                    keys[slot, head, position, j] = decode(
                        kc, ks, slot, head, position, j
                    )
                    values[slot, head, position, j] = decode(
                        vc, vs, slot, head, position, j
                    )

        return kernel
    raise ValueError("unknown KV codec operation: " + kind)


def backend(provider):
    if provider not in ("cuda", "webgpu"):
        raise ValueError("FP8 KV does not support " + provider)
    methods = dict(reference_writer=partial(reference_writer, provider=provider),
                   reference_decoder=partial(reference_decoder, provider=provider))
    if provider == "cuda":
        methods.update(writer=writer, decoder=decoder, packed_decoder=packed_decoder,
                       PACKED_DTYPE=PACKED_DTYPE)
    return SimpleNamespace(**methods)


FORMAT = Format(
    "fp8", layout,
    (("cuda", __name__), ("webgpu", __name__)),
    layouts=(("webgpu", webgpu_layout),),
)

from akas.ops.registry import variants

IMPLEMENTATIONS = variants(("kv_encode", "kv_decode"), "akas.quantization.kv.fp8", "kv-pair")
