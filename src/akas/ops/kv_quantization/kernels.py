"""Standalone KV encode/decode operations using a selected codec's CUDA helpers."""

from akas.ops.kv_quantization import get


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
