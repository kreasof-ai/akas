"""Portable online attention over 64-position tiles.

Each lane scores one position, then accumulates one output channel. Shared tree
reductions require barriers per tile, avoiding a barrier for every cached token.
Trip counts remain static and causal masks use each slot's live position.
"""


def attention(p, read):
    import tilelang.language as T
    from akas.ops.reductions.webgpu import tree_reduce

    r, h, kh, d, cap = (p[k] for k in ("r", "h", "kh", "d", "capacity"))
    chunk = p.get("chunk", 1)
    maximum = tree_reduce(64, "max")
    total = tree_reduce(64)

    @T.macro
    def attend(q, kc, vc, ks, vs, out, positions, enabled):
        with T.Kernel(h, r, threads=64) as (head, row):
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((64,), "float32")
            probabilities = T.alloc_shared((64,), "float32")
            running_max = T.alloc_shared((1,), "float32")
            normalizer = T.alloc_shared((1,), "float32")
            factor = T.alloc_shared((1,), "float32")
            accum = T.alloc_var("float32")
            score = T.alloc_var("float32")
            contribution = T.alloc_var("float32")
            accum = 0
            if tx == 0:
                running_max[0] = -T.infinity("float32")
                normalizer[0] = 0
            T.sync_threads()
            for tile in T.serial(
                T.ceildiv(T.min(positions[row] + 1, cap), 64)
                if p.get("optimized")
                else T.ceildiv(cap, 64)
            ):
                position = tile * 64 + tx
                score = 0
                if position < cap:
                    for channel in T.serial(d):
                        score += q[row, head * d + channel] * read(
                            kc, ks, row // chunk, head // (h // kh), position, channel
                        )
                score = T.if_then_else(
                    position < cap and position <= positions[row],
                    score * d ** (-0.5),
                    T.float32(-1e30),
                )
                scratch[tx] = score
                T.sync_threads()
                maximum(scratch, tx)
                if tx == 0:
                    next_max = T.max(running_max[0], scratch[0])
                    factor[0] = T.exp(running_max[0] - next_max)
                    running_max[0] = next_max
                T.sync_threads()
                probabilities[tx] = T.if_then_else(
                    position < cap and position <= positions[row],
                    T.exp(score - running_max[0]),
                    0,
                )
                scratch[tx] = probabilities[tx]
                T.sync_threads()
                total(scratch, tx)
                if tx == 0:
                    normalizer[0] = normalizer[0] * factor[0] + scratch[0]
                T.sync_threads()
                contribution = 0
                for i in T.serial(64):
                    if tile * 64 + i < cap:
                        contribution += probabilities[i] * read(
                            vc, vs, row // chunk, head // (h // kh), tile * 64 + i, tx
                        )
                accum = accum * factor[0] + contribution
                T.sync_threads()
            out[row * h * d + head * d + tx] = T.if_then_else(
                enabled[row] != 0, accum / normalizer[0], 0
            )

    return attend
