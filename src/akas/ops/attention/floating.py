"""Shared partitioned decode attention over the floating cache ABI.

The cache codec supplies reads; head geometry, positions and masking are data.
CUDA and WebGPU differ only in subgroup shuffle builtins.
"""


def cache_reader(p):
    """Bind the cache layout and codec without coupling them to a backend kernel."""
    import tilelang.language as T
    from akas.ops.attention.reference import cache_storage

    storage = cache_storage(p)
    if p["kv_dtype"] not in ("float16", "float32"):
        from akas.quantization.kv import get

        backend = get(p["kv_dtype"]).backend(p["provider"])
        read = getattr(backend, "decoder", backend.reference_decoder)()
    else:

        @T.macro
        def read(cache, metadata, slot, head, position, channel):
            return (
                T.cast(cache[slot, head, position, channel], "float32")
                * metadata[slot, head, position, 0]
            )

    if len(storage.metadata) != 1:
        raise ValueError("floating attention requires one metadata buffer per cache")
    return storage, read


def cache_arguments(p, storage):
    r, h, d = p["r"], p["h"], p["d"]
    return [
        ("q", (r, h * d), "float32"),
        ("kc", storage.values.shape, storage.values.dtype),
        ("vc", storage.values.shape, storage.values.dtype),
        ("ks", storage.metadata[0].shape, "float32"),
        ("vs", storage.metadata[0].shape, "float32"),
    ]


def make_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    if d != 64:
        raise ValueError("floating attention requires 64-channel heads")
    splits = p.get("splits", 16)
    storage, read = cache_reader(p)
    args = cache_arguments(p, storage)
    controls = [("positions", r, "int32"), ("enabled", r, "int32")]
    if p["provider"] == "webgpu":

        @T.macro
        def shuffle(value, offset):
            return T.call_extern(
                "float32", "subgroupShuffleXor", value, T.uint32(offset)
            )

        @T.macro
        def broadcast(value):
            tx = T.get_thread_binding()
            size = T.call_extern("uint32", "tensor_subgroup_size")
            return T.call_extern(
                "float32",
                "subgroupShuffle",
                value,
                T.cast((tx % size) // 32 * 32, "uint32"),
            )
    else:

        @T.macro
        def shuffle(value, offset):
            return T.shfl_down(value, offset)

        @T.macro
        def broadcast(value):
            return T.shfl_sync(value, 0)

    if kind == "attention_partial":

        @T.macro
        def partial(q, kc, vc, ks, vs, out, positions, enabled):
            with T.Kernel(h * r, splits, threads=128) as (index, split):
                head, slot = index % h, index // h
                tx = T.get_thread_binding()
                lane, warp = tx % 32, tx // 32
                length = T.if_then_else(
                    enabled[slot] != 0, T.min(positions[slot] + 1, cap), 0
                )
                span = T.ceildiv(length, splits)
                begin, end = split * span, T.min((split + 1) * span, length)
                maxima = T.alloc_shared((4,), "float32")
                sums = T.alloc_shared((4,), "float32")
                values = T.alloc_shared((4, 64), "float32")
                maximum = T.alloc_var("float32")
                normalizer = T.alloc_var("float32")
                a0 = T.alloc_var("float32")
                a1 = T.alloc_var("float32")
                maximum, normalizer, a0, a1 = -T.infinity("float32"), 0, 0, 0
                q0, q1 = q[slot, head * d + lane], q[slot, head * d + lane + 32]
                for offset in T.serial(T.ceildiv(T.max(end - begin - warp, 0), 4)):
                    token = begin + warp + offset * 4
                    score = T.alloc_var("float32")
                    score = q0 * read(
                        kc, ks, slot, head // (h // kh), token, lane
                    ) + q1 * read(kc, ks, slot, head // (h // kh), token, lane + 32)
                    for step in T.unroll(5):
                        score = score + shuffle(score, 16 >> step)
                    score = broadcast(score) * 0.125
                    next_max = T.max(maximum, score)
                    correction, probability = (
                        T.exp(maximum - next_max),
                        T.exp(score - next_max),
                    )
                    a0 = a0 * correction + probability * read(
                        vc, vs, slot, head // (h // kh), token, lane
                    )
                    a1 = a1 * correction + probability * read(
                        vc, vs, slot, head // (h // kh), token, lane + 32
                    )
                    normalizer = normalizer * correction + probability
                    maximum = next_max
                values[warp, lane], values[warp, lane + 32] = a0, a1
                if lane == 0:
                    maxima[warp], sums[warp] = maximum, normalizer
                T.sync_threads()
                if warp == 0:
                    merged_max = T.max(
                        T.max(maxima[0], maxima[1]), T.max(maxima[2], maxima[3])
                    )
                    total = T.alloc_var("float32")
                    result0 = T.alloc_var("float32")
                    result1 = T.alloc_var("float32")
                    total, result0, result1 = 0, 0, 0
                    for other in T.unroll(4):
                        factor = T.if_then_else(
                            end > begin, T.exp(maxima[other] - merged_max), 0
                        )
                        total = total + sums[other] * factor
                        result0 = result0 + values[other, lane] * factor
                        result1 = result1 + values[other, lane + 32] * factor
                    base = (index * splits + split) * 66
                    out[base + lane], out[base + lane + 32] = result0, result1
                    if lane == 0:
                        out[base + 64], out[base + 65] = merged_max, total

        return primitive(
            [*args, ("out", r * h * splits * 66, "float32"), *controls], partial
        )
    if kind == "attention_merge":

        @T.macro
        def merge(parts, out, enabled):
            with T.Kernel(h * r, threads=128) as index:
                maxima = T.alloc_fragment((splits,), "float32")
                maximum = T.alloc_fragment((1,), "float32")
                sums = T.alloc_fragment((splits,), "float32")
                normalizer = T.alloc_fragment((1,), "float32")
                products = T.alloc_fragment((splits, d), "float32")
                result = T.alloc_fragment((d,), "float32")
                for i in T.Parallel(splits):
                    maxima[i] = parts[(index * splits + i) * 66 + d]
                T.reduce_max(maxima, maximum, dim=0)
                for i in T.Parallel(splits):
                    sums[i] = parts[(index * splits + i) * 66 + d + 1] * T.exp(
                        maxima[i] - maximum[0]
                    )
                T.reduce_sum(sums, normalizer, dim=0)
                for i, j in T.Parallel(splits, d):
                    products[i, j] = parts[(index * splits + i) * 66 + j] * T.exp(
                        maxima[i] - maximum[0]
                    )
                T.reduce_sum(products, result, dim=0)
                for j in T.Parallel(d):
                    out[index * d + j] = T.if_then_else(
                        enabled[index // h] != 0, result[j] / normalizer[0], 0
                    )

        return primitive(
            [
                ("parts", r * h * splits * 66, "float32"),
                ("out", r * h * d, "float32"),
                ("enabled", r, "int32"),
            ],
            merge,
        )
    raise ValueError("partitioned attention requires partial or merge")
