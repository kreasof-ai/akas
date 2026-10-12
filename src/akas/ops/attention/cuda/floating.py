"""Batched adapters of the established warp and tensor-core online attention.

The cache codec supplies reads; head geometry, positions and masking are data.
Decode partitions live history, and prefill shares K/V across query tiles.
"""


def make_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive
    from akas.ops.attention.floating import cache_reader, cache_arguments

    if kind != "attention":
        from akas.ops.attention.floating import make_kernel as decode

        return decode(kind, p)

    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    if d != 64:
        raise ValueError("floating attention requires 64-channel heads")
    chunk = p.get("chunk", 1)
    slots = r // chunk
    storage, read = cache_reader(p)
    args = cache_arguments(p, storage)
    controls = [("positions", r, "int32"), ("enabled", r, "int32")]
    if kind != "attention" or chunk == 1:
        raise ValueError("floating prefill requires a chunk")

    @T.macro
    def prefill(q, kc, vc, ks, vs, out, positions, enabled):
        with T.Kernel(T.ceildiv(chunk, 32), h, slots, threads=128) as (bx, head, slot):
            query = T.alloc_shared((32, d), "float16")
            key = T.alloc_shared((64, d), "float16")
            value = T.alloc_shared((64, d), "float16")
            prob = T.alloc_shared((32, 64), "float16")
            scores = T.alloc_fragment((32, 64), "float32")
            result = T.alloc_fragment((32, d), "float32")
            maximum = T.alloc_fragment((32,), "float32")
            previous = T.alloc_fragment((32,), "float32")
            factor = T.alloc_fragment((32,), "float32")
            normalizer = T.alloc_fragment((32,), "float32")
            total = T.alloc_fragment((32,), "float32")
            for i, j in T.Parallel(32, d):
                row = slot * chunk + bx * 32 + i
                query[i, j] = T.if_then_else(
                    bx * 32 + i < chunk, q[row, head * d + j], 0
                )
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            # Padded rows have clamped positions; only live cache is loaded.
            for tile in T.serial(
                T.ceildiv(
                    positions[slot * chunk + T.min(chunk - 1, (bx + 1) * 32 - 1)] + 1,
                    64,
                )
            ):
                for i, j in T.Parallel(64, d):
                    key[i, j] = T.if_then_else(
                        tile * 64 + i < cap,
                        read(kc, ks, slot, head // (h // kh), tile * 64 + i, j),
                        0,
                    )
                    value[i, j] = T.if_then_else(
                        tile * 64 + i < cap,
                        read(vc, vs, slot, head // (h // kh), tile * 64 + i, j),
                        0,
                    )
                T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                T.copy(maximum, previous)
                for i, j in T.Parallel(32, 64):
                    row = slot * chunk + bx * 32 + i
                    scores[i, j] = T.if_then_else(
                        bx * 32 + i < chunk and tile * 64 + j <= positions[row],
                        scores[i, j] * d ** (-0.5),
                        -T.infinity("float32"),
                    )
                T.reduce_max(scores, maximum, dim=1, clear=False)
                for i in T.Parallel(32):
                    factor[i] = T.exp(previous[i] - maximum[i])
                for i, j in T.Parallel(32, 64):
                    scores[i, j] = T.exp(scores[i, j] - maximum[i])
                T.reduce_sum(scores, total, dim=1)
                for i in T.Parallel(32):
                    normalizer[i] = normalizer[i] * factor[i] + total[i]
                for i, j in T.Parallel(32, d):
                    result[i, j] *= factor[i]
                T.copy(scores, prob)
                T.gemm(prob, value, result)
            for i, j in T.Parallel(32, d):
                row = slot * chunk + bx * 32 + i
                if bx * 32 + i < chunk:
                    out[(row * h + head) * d + j] = T.if_then_else(
                        enabled[row] != 0, result[i, j] / normalizer[i], 0
                    )

    return primitive([*args, ("out", r * h * d, "float32"), *controls], prefill)
