"""Shared TileLang computations for fp8."""

from akas.quantization._shared.e4m3 import helpers, portable_scaling


def reference_writer(head_dim, *, provider="cuda", chunk=1, round_inputs=False):
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    from akas.quantization._shared.rounding import round_bfloat

    rounded = round_bfloat()

    @T.macro
    def input_value(buffer, row, column):
        if round_inputs:
            return rounded(buffer[row, column])
        else:
            return buffer[row, column]

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
                T.abs(input_value(keys, slot, head * d + block * 128 + tx)),
                0,
            )
            scratch_v[tx] = T.if_then_else(
                block * 128 + tx < d,
                T.abs(input_value(values, slot, head * d + block * 128 + tx)),
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
                if tx < 32 and block * 128 + tx * 4 < d and (enabled[slot] != 0):
                    word_k = T.alloc_var("uint32")
                    word_v = T.alloc_var("uint32")
                    word_k = 0
                    word_v = 0
                    for i in T.serial(4):
                        if block * 128 + tx * 4 + i < d:
                            word_k = (
                                word_k
                                | encode_scaled(
                                    input_value(
                                        keys, slot, head * d + block * 128 + tx * 4 + i
                                    ),
                                    scale_k[0],
                                )
                                << i * 8
                            )
                            word_v = (
                                word_v
                                | encode_scaled(
                                    input_value(
                                        values,
                                        slot,
                                        head * d + block * 128 + tx * 4 + i,
                                    ),
                                    scale_v[0],
                                )
                                << i * 8
                            )
                    kc[slot // chunk, head, position, block * 32 + tx] = word_k
                    vc[slot // chunk, head, position, block * 32 + tx] = word_v
            elif block * 128 + tx < d and enabled[slot] != 0:
                kc[slot // chunk, head, position, block * 128 + tx] = T.cast(
                    encode(
                        input_value(keys, slot, head * d + block * 128 + tx)
                        / scale_k[0]
                    ),
                    "uint8",
                )
                vc[slot // chunk, head, position, block * 128 + tx] = T.cast(
                    encode(
                        input_value(values, slot, head * d + block * 128 + tx)
                        / scale_v[0]
                    ),
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
            bits = cache[slot, head, position, channel // 4] >> channel % 4 * 8 & 255
        else:
            bits = T.cast(cache[slot, head, position, channel], "uint32")
        return decode(bits) * metadata[slot, head, position, channel // 128]

    return read


def binding(shape, *, provider, chunk=1, value_offset=0):
    """Bind uniform cache access; preserve the qualified CUDA pair writer."""
    import tilelang.language as T
    from akas.quantization.formats import Codec
    from akas.quantization._shared.access import cache_reader, cache_tile
    from .definition import FORMAT

    if len(shape) != 4:
        raise ValueError("KV access requires [slots, heads, capacity, channels]")
    if type(chunk) is not int or chunk <= 0:
        raise ValueError("KV access requires a positive chunk")
    storage = FORMAT.storage(shape, provider=provider)
    native = provider == "cuda"
    backend = FORMAT.backend(provider)
    scalar = backend.decoder() if native else reference_decoder(provider=provider)
    read = cache_reader(scalar)
    writer = reference_writer(shape[-1], provider=provider, chunk=chunk)

    @T.macro
    def write(
        keys, values, row, head, position, key_encoded, value_encoded, tx, active
    ):
        writer(
            keys,
            values,
            row,
            head,
            position,
            key_encoded[0],
            value_encoded[0],
            key_encoded[1],
            value_encoded[1],
            tx,
            active,
        )

    pair = None
    packed = None
    if native:
        pair = (
            backend.writer(shape[-1], value_offset=value_offset)
            if shape[-1] % 128 == 0
            else None
        )
        packed = backend.packed_decoder()

    @T.macro
    def write_pair(keys, values, row, head, slot, position, key_encoded, value_encoded):
        pair(
            keys,
            values,
            row,
            head,
            slot,
            position,
            key_encoded[0],
            value_encoded[0],
            key_encoded[1],
            value_encoded[1],
        )

    @T.macro
    def read_packed(encoded, row, column):
        return packed(encoded[0], encoded[1], row, column)

    return Codec(
        storage,
        read,
        cache_tile(read),
        write=write,
        write_pair=write_pair if pair is not None else None,
        read_packed=read_packed if packed is not None else None,
        packed_dtype=backend.PACKED_DTYPE if native else None,
    )


def quantize(spec, schedule, *, provider="webgpu"):
    """Portable pair conversion with the native writer's BF16 input policy."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel
    from .definition import FORMAT

    p = {**spec.parameters, **schedule}
    slots, heads, depth, capacity = (
        p[n] for n in ("slots", "kv_heads", "d", "capacity")
    )
    chunk = p.get("chunk", 1)
    if type(chunk) is not int or chunk <= 0:
        raise ValueError("KV operation requires a positive chunk")
    rows = slots * chunk
    storage = FORMAT.storage((slots, heads, capacity, depth), provider=provider)
    payload, scale = storage.buffers
    encode = reference_writer(depth, provider=provider, chunk=chunk, round_inputs=True)

    @kernel
    def compute(
        keys: Tensor((rows, heads * depth), "float32"),
        values: Tensor((rows, heads * depth), "float32"),
        positions: Tensor((rows,), "int32"),
        active: Tensor((rows,), "int32"),
        kc: Tensor(payload.shape, payload.dtype),
        vc: Tensor(payload.shape, payload.dtype),
        ks: Tensor(scale.shape, scale.dtype),
        vs: Tensor(scale.shape, scale.dtype),
    ) -> tuple[
        Tensor(payload.shape, payload.dtype),
        Tensor(payload.shape, payload.dtype),
        Tensor(scale.shape, scale.dtype),
        Tensor(scale.shape, scale.dtype),
    ]:
        with T.Kernel(heads, rows, threads=128) as (head, row):
            tx = T.get_thread_binding()
            encode(keys, values, row, head, positions[row], kc, vc, ks, vs, tx, active)
        return kc, vc, ks, vs

    return compute


def dequantize(spec, schedule):
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.quantization._shared.rounding import round_bfloat
    from .definition import FORMAT

    p = {**spec.parameters, **schedule}
    slots, heads, depth, capacity = (
        p[n] for n in ("slots", "kv_heads", "d", "capacity")
    )
    shape = (slots, heads, capacity, depth)
    binding = FORMAT.bind(shape, provider="webgpu")
    payload, scale = binding.storage.buffers
    read, rounded = binding.read, round_bfloat()

    @kernel
    def compute(
        kc: Tensor(payload.shape, payload.dtype),
        vc: Tensor(payload.shape, payload.dtype),
        ks: Tensor(scale.shape, scale.dtype),
        vs: Tensor(scale.shape, scale.dtype),
    ) -> tuple[Tensor(shape, "float32"), Tensor(shape, "float32")]:
        keys = output(shape, "float32")
        values = output(shape, "float32")
        with T.Kernel(slots, heads, capacity, threads=128) as (slot, head, position):
            for channel in T.Parallel(depth):
                keys[slot, head, position, channel] = rounded(
                    read((kc, ks), slot, head, position, channel)
                )
                values[slot, head, position, channel] = rounded(
                    read((vc, vs), slot, head, position, channel)
                )
        return keys, values

    return compute


def stage(spec, schedule, *, provider="webgpu"):
    """Decode a live cache prefix to F16, zero padding, and return unit scales.

    The consumer supplies valid prefix lengths. Persistent payloads and their
    F32 scales are read only; this temporary representation does not requantize
    either cache or change its declared precision.
    """
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from .definition import FORMAT

    p = {**spec.parameters, **schedule}
    slots, heads, capacity, depth = (
        p[name] for name in ("slots", "kv_heads", "capacity", "d")
    )
    shape = (slots, heads, capacity, depth)
    binding = FORMAT.bind(shape, provider=provider)
    payload, metadata = binding.storage.buffers
    read = binding.read
    block = 128
    units = (slots, heads, capacity, 1)

    @kernel
    def compute(
        kc: Tensor(payload.shape, payload.dtype),
        vc: Tensor(payload.shape, payload.dtype),
        ks: Tensor(metadata.shape, metadata.dtype),
        vs: Tensor(metadata.shape, metadata.dtype),
        lengths: Tensor((slots,), "int32"),
    ) -> tuple[
        Tensor(shape, "float16"),
        Tensor(shape, "float16"),
        Tensor(units, "float32"),
    ]:
        keys = output(shape, "float16")
        values = output(shape, "float16")
        scales = output(units, "float32")
        with T.Kernel(T.ceildiv(capacity, block), heads, slots, threads=128) as (
            tile,
            head,
            slot,
        ):
            for i, j in T.Parallel(block, depth):
                position = tile * block + i
                if position < capacity:
                    keys[slot, head, position, j] = T.if_then_else(
                        position < lengths[slot],
                        read((kc, ks), slot, head, position, j),
                        0,
                    )
                    values[slot, head, position, j] = T.if_then_else(
                        position < lengths[slot],
                        read((vc, vs), slot, head, position, j),
                        0,
                    )
            for i in T.Parallel(block):
                if tile * block + i < capacity:
                    scales[slot, head, tile * block + i, 0] = 1
        return keys, values, scales

    return compute
