"""CUDA specializations and shared bindings for fp8."""

from .definition import FORMAT, IMPLEMENTATIONS as _CHOICES

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output
from functools import partial
from akas.quantization.kv.fp8.kernel import reference_decoder as _decoder
from akas.quantization.kv.fp8.kernel import reference_writer as _writer


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


def kv_encode(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.kv.codec import get

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, heads, d, capacity = (
        p[name] for name in ("slots", "kv_heads", "d", "capacity")
    )
    chunk = p.get("chunk", 1)
    if type(chunk) is not int or chunk <= 0:
        raise ValueError("KV operation requires a positive chunk")
    rows = slots * chunk
    codec = get(p.get("kv_dtype", "fp8"))
    binding = codec.bind((slots, heads, capacity, d), provider="cuda")
    storage = binding.storage
    if len(storage.metadata) != 1:
        raise ValueError("KV pair operation requires one metadata buffer per cache")
    shape, dtype = (storage.values.shape, storage.values.dtype)
    metadata_shape, metadata_dtype = (
        storage.metadata[0].shape,
        storage.metadata[0].dtype,
    )
    if binding.write_pair is None:
        from .kernel import quantize

        return quantize(spec, schedule, provider="cuda")
    from akas.quantization._shared.access import pair_arguments

    encode = pair_arguments(binding.require("write_pair"))

    @kernel
    def compute(
        keys: Tensor((rows, heads * d), "bfloat16"),
        values: Tensor((rows, heads * d), "float32"),
        positions: Tensor((rows,), "int32"),
        active: Tensor((rows,), "int32"),
        kc: Tensor(shape, dtype),
        vc: Tensor(shape, dtype),
        ks: Tensor(metadata_shape, metadata_dtype),
        vs: Tensor(metadata_shape, metadata_dtype),
    ) -> tuple[
        Tensor(shape, dtype),
        Tensor(shape, dtype),
        Tensor(metadata_shape, metadata_dtype),
        Tensor(metadata_shape, metadata_dtype),
    ]:
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
        return (kc, vc, ks, vs)

    return compute


def kv_decode(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.kv.codec import get

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, heads, d, capacity = (
        p[name] for name in ("slots", "kv_heads", "d", "capacity")
    )
    chunk = p.get("chunk", 1)
    if type(chunk) is not int or chunk <= 0:
        raise ValueError("KV operation requires a positive chunk")
    rows = slots * chunk
    codec = get(p.get("kv_dtype", "fp8"))
    binding = codec.bind((slots, heads, capacity, d), provider="cuda")
    storage = binding.storage
    if len(storage.metadata) != 1:
        raise ValueError("KV pair operation requires one metadata buffer per cache")
    shape, dtype = (storage.values.shape, storage.values.dtype)
    metadata_shape, metadata_dtype = (
        storage.metadata[0].shape,
        storage.metadata[0].dtype,
    )
    from akas.quantization._shared.access import cache_arguments

    decode = cache_arguments(binding.read)

    @kernel
    def compute(
        kc: Tensor(shape, dtype),
        vc: Tensor(shape, dtype),
        ks: Tensor(metadata_shape, metadata_dtype),
        vs: Tensor(metadata_shape, metadata_dtype),
    ) -> tuple[
        Tensor((slots, heads, capacity, d), "bfloat16"),
        Tensor((slots, heads, capacity, d), "bfloat16"),
    ]:
        keys = output((slots, heads, capacity, d), "bfloat16")
        values = output((slots, heads, capacity, d), "bfloat16")
        with T.Kernel(slots, heads, capacity, threads=256) as (slot, head, position):
            for j in T.Parallel(d):
                keys[slot, head, position, j] = decode(kc, ks, slot, head, position, j)
                values[slot, head, position, j] = decode(
                    vc, vs, slot, head, position, j
                )
        return (keys, values)

    return compute


BLOCK = 128
reference_writer = partial(_writer, provider="cuda")
reference_decoder = partial(_decoder, provider="cuda")
PACKED_DTYPE = "float8_e4m3fn"


def bind(shape, *, chunk=1, value_offset=0):
    from .kernel import binding

    return binding(shape, provider="cuda", chunk=chunk, value_offset=value_offset)


def quantize(spec, schedule):
    return kv_encode(spec, schedule)


def dequantize(spec, schedule):
    return kv_decode(spec, schedule)


AVAILABLE = "cuda" in dict(FORMAT.providers)
CAPABILITIES = (
    "read",
    "read_tile",
    "write",
    "write_pair",
    "read_packed",
    "quantize",
    "dequantize",
)
IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "cuda")
