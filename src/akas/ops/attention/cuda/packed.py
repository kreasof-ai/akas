"""Packed projections, attention, normalization, and reductions for CUDA.

Factories are producer-only; consumers execute Tensor artifacts."""


def warp_partial_kernel(p, splits):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    h, kh, d, cap = (p[n] for n in ("h", "kh", "d", "cap"))
    if d != 64 or type(splits) is not int or splits < 1:
        raise ValueError("unsupported warp attention schedule")

    @T.macro
    def algorithm(q, kc, vc, parts, pos):
        with T.Kernel(h, splits, threads=128) as (head, split):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            chunk = T.ceildiv(pos[0] + 1, splits)
            begin = split * chunk
            end = T.min(begin + chunk, pos[0] + 1)
            kvhead = head // (h // kh)
            q0 = q[head * 64 + lane]
            q1 = q[head * 64 + lane + 32]
            maxima = T.alloc_shared((4,), "float32")
            sums = T.alloc_shared((4,), "float32")
            values = T.alloc_shared((4, 64), "float32")
            maximum = T.alloc_var("float32")
            normalizer = T.alloc_var("float32")
            a0 = T.alloc_var("float32")
            a1 = T.alloc_var("float32")
            maximum = -T.infinity("float32")
            normalizer = 0
            a0 = 0
            a1 = 0
            for offset in T.serial(T.ceildiv(T.max(end - begin - warp, 0), 4)):
                token = begin + warp + offset * 4
                base = (token * kh + kvhead) * 64
                score = q0 * T.cast(kc[base + lane], "float32") + q1 * T.cast(
                    kc[base + lane + 32], "float32"
                )
                score = score + T.shfl_down(score, 16)
                score = score + T.shfl_down(score, 8)
                score = score + T.shfl_down(score, 4)
                score = score + T.shfl_down(score, 2)
                score = score + T.shfl_down(score, 1)
                score = T.shfl_sync(score, 0) * 0.125
                next_max = T.max(maximum, score)
                correction = T.exp(maximum - next_max)
                probability = T.exp(score - next_max)
                a0 = a0 * correction + probability * T.cast(vc[base + lane], "float32")
                a1 = a1 * correction + probability * T.cast(
                    vc[base + lane + 32], "float32"
                )
                normalizer = normalizer * correction + probability
                maximum = next_max
            values[warp, lane] = a0
            values[warp, lane + 32] = a1
            if lane == 0:
                maxima[warp] = maximum
                sums[warp] = normalizer
            T.sync_threads()
            if warp == 0:
                merged_max = T.max(
                    T.max(maxima[0], maxima[1]), T.max(maxima[2], maxima[3])
                )
                total = T.alloc_var("float32")
                result0 = T.alloc_var("float32")
                result1 = T.alloc_var("float32")
                total = 0
                result0 = 0
                result1 = 0
                for other in T.unroll(4):
                    factor = T.if_then_else(
                        T.isfinite(merged_max), T.exp(maxima[other] - merged_max), 0
                    )
                    total = total + sums[other] * factor
                    result0 = result0 + values[other, lane] * factor
                    result1 = result1 + values[other, lane + 32] * factor
                base = (head * splits + split) * 66
                parts[base + lane] = result0
                parts[base + lane + 32] = result1
                if lane == 0:
                    parts[base + 64] = merged_max
                    parts[base + 65] = total

    return primitive(
        [
            ("q", h * d, "float32"),
            ("kc", cap * kh * d, "float16"),
            ("vc", cap * kh * d, "float16"),
            ("parts", h * splits * 66, "float32"),
            ("pos", 2, "int32"),
        ],
        algorithm,
    )


def grouped_kernel(p, stage=32, warps=2):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    h, kh, d, cap, splits = (p[n] for n in ("h", "kh", "d", "cap", "splits"))
    group = h // kh
    threads = group * warps * 32
    if (
        d != 64
        or h % kh
        or group not in (2, 4)
        or (stage not in (16, 32, 64))
        or (warps not in (2, 4))
    ):
        raise ValueError("unsupported grouped attention schedule")

    @T.macro
    def algorithm(q, kc, vc, parts, pos):
        with T.Kernel(kh, splits, threads=threads) as (kvhead, split):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            local = warp // warps
            worker = warp % warps
            head = kvhead * group + local
            chunk = T.ceildiv(pos[0] + 1, splits)
            begin = split * chunk
            end = T.min(begin + chunk, pos[0] + 1)
            keys = T.alloc_shared((stage, 64), "float16")
            vals = T.alloc_shared((stage, 64), "float16")
            maxima = T.alloc_shared((group * warps,), "float32")
            sums = T.alloc_shared((group * warps,), "float32")
            values = T.alloc_shared((group * warps, 64), "float32")
            q0 = q[head * 64 + lane]
            q1 = q[head * 64 + lane + 32]
            maximum = T.alloc_var("float32")
            normalizer = T.alloc_var("float32")
            a0 = T.alloc_var("float32")
            a1 = T.alloc_var("float32")
            maximum = -T.infinity("float32")
            normalizer = 0
            a0 = 0
            a1 = 0
            for block in T.serial(T.ceildiv(T.max(end - begin, 0), stage)):
                base = begin + block * stage
                for token, col in T.Parallel(stage, 64):
                    keys[token, col] = T.if_then_else(
                        base + token < end,
                        kc[((base + token) * kh + kvhead) * 64 + col],
                        0,
                    )
                    vals[token, col] = T.if_then_else(
                        base + token < end,
                        vc[((base + token) * kh + kvhead) * 64 + col],
                        0,
                    )
                T.sync_threads()
                for offset in T.serial(
                    T.ceildiv(T.max(T.min(stage, end - base) - worker, 0), warps)
                ):
                    token = worker + offset * warps
                    score = q0 * T.cast(keys[token, lane], "float32") + q1 * T.cast(
                        keys[token, lane + 32], "float32"
                    )
                    score = score + T.shfl_down(score, 16)
                    score = score + T.shfl_down(score, 8)
                    score = score + T.shfl_down(score, 4)
                    score = score + T.shfl_down(score, 2)
                    score = score + T.shfl_down(score, 1)
                    score = T.shfl_sync(score, 0) * 0.125
                    next_max = T.max(maximum, score)
                    correction = T.exp(maximum - next_max)
                    probability = T.exp(score - next_max)
                    a0 = a0 * correction + probability * T.cast(
                        vals[token, lane], "float32"
                    )
                    a1 = a1 * correction + probability * T.cast(
                        vals[token, lane + 32], "float32"
                    )
                    normalizer = normalizer * correction + probability
                    maximum = next_max
                T.sync_threads()
            values[warp, lane] = a0
            values[warp, lane + 32] = a1
            if lane == 0:
                maxima[warp] = maximum
                sums[warp] = normalizer
            T.sync_threads()
            if worker == 0:
                merged_max = T.alloc_var("float32")
                total = T.alloc_var("float32")
                result0 = T.alloc_var("float32")
                result1 = T.alloc_var("float32")
                merged_max = -T.infinity("float32")
                total = 0
                result0 = 0
                result1 = 0
                for other in T.unroll(warps):
                    merged_max = T.max(merged_max, maxima[local * warps + other])
                for other in T.unroll(warps):
                    index = local * warps + other
                    factor = T.if_then_else(
                        T.isfinite(merged_max), T.exp(maxima[index] - merged_max), 0
                    )
                    total = total + sums[index] * factor
                    result0 = result0 + values[index, lane] * factor
                    result1 = result1 + values[index, lane + 32] * factor
                base = (head * splits + split) * 66
                parts[base + lane] = result0
                parts[base + lane + 32] = result1
                if lane == 0:
                    parts[base + 64] = merged_max
                    parts[base + 65] = total

    return primitive(
        [
            ("q", h * d, "float32"),
            ("kc", cap * kh * d, "float16"),
            ("vc", cap * kh * d, "float16"),
            ("parts", h * splits * 66, "float32"),
            ("pos", 2, "int32"),
        ],
        algorithm,
    )


def merge_kernel(p, splits):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    h, d = (p["h"], p["d"])
    stride = d + 2

    @T.macro
    def algorithm(parts, out):
        with T.Kernel(h, threads=128) as head:
            maxima = T.alloc_fragment((splits,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            sums = T.alloc_fragment((splits,), "float32")
            normalizer = T.alloc_fragment((1,), "float32")
            products = T.alloc_fragment((splits, d), "float32")
            result = T.alloc_fragment((d,), "float32")
            for i in T.Parallel(splits):
                maxima[i] = parts[(head * splits + i) * stride + d]
            T.reduce_max(maxima, maximum, dim=0)
            for i in T.Parallel(splits):
                sums[i] = parts[(head * splits + i) * stride + (d + 1)] * T.exp(
                    maxima[i] - maximum[0]
                )
            T.reduce_sum(sums, normalizer, dim=0)
            for i, j in T.Parallel(splits, d):
                products[i, j] = parts[(head * splits + i) * stride + j] * T.exp(
                    maxima[i] - maximum[0]
                )
            T.reduce_sum(products, result, dim=0)
            for j in T.Parallel(d):
                out[head * d + j] = result[j] / normalizer[0]

    return primitive(
        [("parts", h * splits * stride, "float32"), ("out", h * d, "float32")],
        algorithm,
    )


def make_kernel(kind, p):
    if kind == "attention_partial":
        return warp_partial_kernel(p, p["splits"])
    if kind == "attention_grouped":
        return grouped_kernel(p, p.get("stage", 32), p.get("warps", 2))
    if kind == "attention_merge":
        return merge_kernel(p, p["splits"])
    raise ValueError("unsupported operation: " + kind)
