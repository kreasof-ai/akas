"""Block-128 E4M3 CUDA encode/decode helpers, independent of model geometry."""

from akas.ops.kv_quantization.fp8 import BLOCK

PACKED_DTYPE = "float8_e4m3fn"


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
