"""WebGPU specializations and shared bindings for attention."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def qk_norm_rotary_cache(spec, schedule):
    """Prepare normalized/rotated Q and the final FP16 KV in one dispatch."""
    import tilelang.language as T
    from akas.kernels.webgpu import subgroup_reduce
    from akas.quantization._shared.rounding import round_half

    p = {**spec.parameters, **schedule}
    rows, heads, kh, d, rot, cap, stride, blocks, cache_cap, chunk = (
        p[n]
        for n in (
            "r",
            "h",
            "kh",
            "d",
            "rotary",
            "capacity",
            "stride",
            "blocks",
            "cache_capacity",
            "chunk",
        )
    )
    if d not in (64, 128) or chunk != 1:
        raise ValueError("fused QK preparation requires D64/D128 decode")
    eps, half = p["eps"], rot // 2
    pair = d // 2 if p.get("full_head_pairs", False) else half
    shape = (rows, kh, cache_cap, d)
    reduce, rounded = subgroup_reduce(d), round_half()

    @T.macro
    def prepare(
        value: T.Ref, x, w, positions, fine, coarse, scratch, row, head, count, channel
    ):
        total = T.alloc_var("float32")
        base = (row * count + head) * d
        total = x[base + channel] * x[base + channel]
        reduce(scratch, channel, total)
        scale = T.rsqrt(scratch[0] / d + eps)
        if channel % pair < half and channel < 2 * pair:
            pos = T.min(T.max(positions[row], 0), cap - 1)
            column = channel % pair
            c0, s0 = fine[pos % stride, column], fine[pos % stride, column + half]
            c1, s1 = coarse[pos // stride, column], coarse[pos // stride, column + half]
            cosine, sine = c0 * c1 - s0 * s1, s0 * c1 + c0 * s1
            left = x[base + column] * scale * w[column]
            right = x[base + column + pair] * scale * w[column + pair]
            value = T.if_then_else(
                channel < pair,
                left * cosine - right * sine,
                left * sine + right * cosine,
            )
        else:
            value = x[base + channel] * scale * w[channel]

    @kernel
    def compute(
        q: Tensor((rows * heads * d,), "float32"),
        k: Tensor((rows * kh * d,), "float32"),
        v: Tensor((rows, kh * d), "float32"),
        wq: Tensor((d,), "float32"),
        wk: Tensor((d,), "float32"),
        kc: Tensor(shape, "float16"),
        vc: Tensor(shape, "float16"),
        positions: Tensor((rows,), "int32"),
        enabled: Tensor((rows,), "int32"),
        fine: Tensor((stride, rot), "float32"),
        coarse: Tensor((blocks, rot), "float32"),
    ) -> tuple[
        Tensor((rows, heads * d), "float32"),
        Tensor(shape, "float16"),
        Tensor(shape, "float16"),
    ]:
        out = output((rows, heads * d), "float32", position=5)
        with T.Kernel(heads + kh, rows, threads=d) as (group, row):
            channel = T.get_thread_binding()
            scratch = T.alloc_shared((d,), "float32")
            value = T.alloc_var("float32")
            if group < heads:
                prepare(
                    value,
                    q,
                    wq,
                    positions,
                    fine,
                    coarse,
                    scratch,
                    row,
                    group,
                    heads,
                    channel,
                )
                out[row, group * d + channel] = value
            else:
                prepare(
                    value,
                    k,
                    wk,
                    positions,
                    fine,
                    coarse,
                    scratch,
                    row,
                    group - heads,
                    kh,
                    channel,
                )
                if enabled[row] != 0:
                    kc[row, group - heads, positions[row] % cache_cap, channel] = (
                        rounded(value)
                    )
                    vc[row, group - heads, positions[row] % cache_cap, channel] = (
                        rounded(v[row, (group - heads) * d + channel])
                    )
        return out, kc, vc

    return compute


def window_prefill(spec, schedule):
    """Reuse circular KV tiles across four adjacent queries and all GQA heads."""
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import registers, vector

    p = {**spec.parameters, **schedule}
    rows, heads, kh, d, cap, chunk, window, splits = (
        p[n] for n in ("r", "h", "kh", "d", "capacity", "chunk", "window", "splits")
    )
    bm, group = p.get("query_rows", 4), heads // kh
    shared = 4 * bm * group * (d + 32) + 2 * d * 65
    if (
        d not in (64, 128)
        or bm not in (1, 2, 4, 8)
        or chunk % bm
        or rows % chunk
        or heads % kh
        or group not in (1, 2, 4, 8)
        or shared > 32768
        or p.get("key_rows", 32) != 32
    ):
        raise ValueError("unsupported tiled circular attention geometry")
    shape = (rows // chunk, kh, cap, d)

    @T.macro
    def set_value(value: T.Ref, source):
        value = source

    def live_end(positions, enabled, row0, first):
        last = first
        for i in range(bm):
            last = T.max(
                last,
                T.if_then_else(enabled[row0 + i] != 0, positions[row0 + i] + 1, first),
            )
        return last

    def initialize(maxima):
        for value in maxima.values():
            set_value(value[0], -1e30)

    def qvector(queries, i, head, channel):
        return vector([queries[i, head, channel * 4 + j] for j in range(4)])

    def kvector(keys, channel, lane):
        return vector(
            [T.cast(keys[channel * 4 + j, lane], "float32") for j in range(4)]
        )

    @T.macro
    def add_score(value: T.Ref, queries, right, i, head, channel):
        value += T.call_extern(
            "float32", "dot", qvector(queries, i, head, channel), right
        )

    def score_rows(dots, queries, right, head, channel):
        for i in range(bm):
            add_score(dots[i,][0], queries, right, i, head, channel)

    @T.macro
    def dot_tiles(dots, queries, keys, head, lane):
        for channel in T.serial(d // 4):
            right = T.alloc_var("float32x4")
            right = kvector(keys, channel, lane)
            score_rows(dots, queries, right, head, channel)

    @T.macro
    def exchange(value, offset):
        return T.call_extern("float32", "subgroupShuffleXor", value, T.uint32(offset))

    @T.macro
    def probabilities(
        score: T.Ref,
        maximum: T.Ref,
        norm: T.Ref,
        factor: T.Ref,
        probs,
        positions,
        enabled,
        row0,
        base,
        head,
        lane,
        i,
    ):
        row = row0 + i
        live = (
            (enabled[row] != 0)
            & (base + lane <= positions[row])
            & ((base + lane > positions[row] - window) if window else True)
        )
        score = T.if_then_else(live, score * d**-0.5, -1e30)
        peak = T.alloc_var("float32")
        peak = score
        for step in T.unroll(5):
            peak = T.max(peak, exchange(peak, 16 >> step))
        next_max = T.max(maximum, peak)
        factor = T.exp(maximum - next_max)
        probability = T.if_then_else(live, T.exp(score - next_max), 0)
        total = T.alloc_var("float32")
        total = probability
        for step in T.unroll(5):
            total += exchange(total, 16 >> step)
        norm = norm * factor + total
        maximum = next_max
        probs[i, head, lane] = probability

    def softmax_rows(
        dots, maxima, norms, factors, probs, positions, enabled, row0, base, head, lane
    ):
        for i in range(bm):
            probabilities(
                dots[i,][0],
                maxima[i,][0],
                norms[i,][0],
                factors[i,][0],
                probs,
                positions,
                enabled,
                row0,
                base,
                head,
                lane,
                i,
            )

    @T.macro
    def rescale(value: T.Ref, factor):
        value *= factor

    def scale_rows(accum, factors):
        for i in range(bm):
            for side in range(d // 32):
                rescale(accum[i, side][0], factors[i,][0])

    def probability_vector(probs, i, head, key):
        return vector([probs[i, head, key * 4 + j] for j in range(4)])

    def value_vector(values, key, lane, side):
        return vector(
            [T.cast(values[key * 4 + j, lane + side * 32], "float32") for j in range(4)]
        )

    @T.macro
    def add_value(value: T.Ref, probs, right, i, head, key):
        value += T.call_extern(
            "float32", "dot", probability_vector(probs, i, head, key), right
        )

    def value_rows(accum, probs, values, head, lane, key):
        for side in range(d // 32):
            right = value_vector(values, key, lane, side)
            for i in range(bm):
                add_value(accum[i, side][0], probs, right, i, head, key)

    @T.macro
    def value_tiles(accum, probs, values, head, lane):
        for key in T.serial(8):
            value_rows(accum, probs, values, head, lane, key)

    @T.macro
    def write_value(out, value, row0, head, part, lane, i, side):
        offset = (((row0 + i) * heads + head) * splits + part) * (d + 2)
        out[offset + lane + side * 32] = value

    @T.macro
    def write_stats(out, maximum, norm, row0, head, part, lane, i):
        if lane == 0:
            offset = (((row0 + i) * heads + head) * splits + part) * (d + 2)
            out[offset + d] = maximum
            out[offset + d + 1] = norm

    def store_rows(out, accum, maxima, norms, row0, head, part, lane):
        for i in range(bm):
            for side in range(d // 32):
                write_value(out, accum[i, side][0], row0, head, part, lane, i, side)
            write_stats(out, maxima[i,][0], norms[i,][0], row0, head, part, lane, i)

    @kernel
    def compute(
        q: Tensor((rows, heads * d), "float32"),
        kc: Tensor(shape, "float16"),
        vc: Tensor(shape, "float16"),
        positions: Tensor((rows,), "int32"),
        enabled: Tensor((rows,), "int32"),
    ) -> Tensor((rows * heads * splits * (d + 2),), "float32"):
        out = output((rows * heads * splits * (d + 2),), "float32")
        with T.Kernel(kh, rows // bm, splits, threads=group * 32) as (
            kv_head,
            block,
            part,
        ):
            tx = T.get_thread_binding()
            head, lane = tx // 32, tx % 32
            row0 = block * bm
            first = T.max(0, positions[row0] - window + 1) if window else 0
            last = live_end(positions, enabled, row0, first)
            span = T.ceildiv(T.ceildiv(T.max(last - first, 0), 32), splits) * 32
            begin, end = first + part * span, T.min(last, first + (part + 1) * span)
            queries = T.alloc_shared((bm, group, d), "float32")
            keys = T.alloc_shared((d, 33), "float16")
            values = T.alloc_shared((32, d), "float16")
            probs = T.alloc_shared((bm, group, 32), "float32")
            maxima, norms, factors = (
                registers((bm,)),
                registers((bm,)),
                registers((bm,)),
            )
            accum = registers((bm, d // 32))
            initialize(maxima)
            for i in T.unroll(bm):
                for side in T.unroll(d // 32):
                    queries[i, head, lane + side * 32] = q[
                        row0 + i, (kv_head * group + head) * d + lane + side * 32
                    ]
            T.sync_threads()
            for tile in T.serial(T.ceildiv(T.max(end - begin, 0), 32)):
                base = begin + tile * 32
                for key, channel in T.Parallel(32, d):
                    keys[channel, key] = T.if_then_else(
                        base + key < end,
                        kc[row0 // chunk, kv_head, (base + key) % cap, channel],
                        0,
                    )
                    values[key, channel] = T.if_then_else(
                        base + key < end,
                        vc[row0 // chunk, kv_head, (base + key) % cap, channel],
                        0,
                    )
                T.sync_threads()
                dots = registers((bm,))
                dot_tiles(dots, queries, keys, head, lane)
                softmax_rows(
                    dots,
                    maxima,
                    norms,
                    factors,
                    probs,
                    positions,
                    enabled,
                    row0,
                    base,
                    head,
                    lane,
                )
                scale_rows(accum, factors)
                T.sync_threads()
                value_tiles(accum, probs, values, head, lane)
                T.sync_threads()
            store_rows(
                out, accum, maxima, norms, row0, kv_head * group + head, part, lane
            )
        return out

    return compute.with_attr("tirx.noalias", True)


def window_merge(spec, schedule):
    """Merge split softmax statistics without fragment reductions."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows, heads, d, splits = (p[n] for n in ("r", "h", "d", "splits"))

    @kernel
    def compute(
        parts: Tensor((rows * heads * splits * (d + 2),), "float32"),
        enabled: Tensor((rows,), "int32"),
    ) -> Tensor((rows * heads * d,), "float32"):
        out = output((rows * heads * d,), "float32", position=1)
        with T.Kernel(heads * rows, threads=d) as index:
            tx = T.get_thread_binding()
            maximum, total, result = (
                T.alloc_var("float32"),
                T.alloc_var("float32"),
                T.alloc_var("float32"),
            )
            maximum, total, result = -1e30, 0, 0
            for part in T.serial(splits):
                maximum = T.max(maximum, parts[(index * splits + part) * (d + 2) + d])
            for part in T.serial(splits):
                base = (index * splits + part) * (d + 2)
                factor = T.exp(parts[base + d] - maximum)
                total += parts[base + d + 1] * factor
                result += parts[base + tx] * factor
            out[index * d + tx] = T.if_then_else(
                enabled[index // heads] != 0, result / T.max(total, 1e-30), 0
            )
        return out

    return compute


def window_merge_subgroup(spec, schedule):
    """Compute each split weight once and share it across the output channels."""
    import tilelang.language as T
    from akas.kernels.webgpu import subgroup_reduce

    p = {**spec.parameters, **schedule}
    rows, heads, d, splits = (p[n] for n in ("r", "h", "d", "splits"))
    maximum_reduce = subgroup_reduce(d, "max")
    sum_reduce = subgroup_reduce(d)

    @kernel
    def compute(
        parts: Tensor((rows * heads * splits * (d + 2),), "float32"),
        enabled: Tensor((rows,), "int32"),
    ) -> Tensor((rows * heads * d,), "float32"):
        out = output((rows * heads * d,), "float32", position=1)
        with T.Kernel(heads * rows, threads=d) as index:
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((d,), "float32")
            weights = T.alloc_shared((splits,), "float32")
            maximum, normalizer, result = (
                T.alloc_var("float32"),
                T.alloc_var("float32"),
                T.alloc_var("float32"),
            )
            maximum, normalizer, result = -1e30, 0, 0
            subgroup = T.cast(T.call_extern("uint32", "tensor_subgroup_size"), "int32")
            if T.min(subgroup, d) >= splits:
                part = tx % splits
                base = (index * splits + part) * (d + 2)
                maximum = T.call_extern("float32", "subgroupMax", parts[base + d])
                factor = T.exp(parts[base + d] - maximum)
                normalizer = T.call_extern(
                    "float32", "subgroupAdd", parts[base + d + 1] * factor
                )
                normalizer *= T.cast(splits, "float32") / T.cast(
                    T.min(subgroup, d), "float32"
                )
                if tx < splits:
                    weights[tx] = factor
                T.sync_threads()
            else:
                for tile in T.serial(T.ceildiv(splits, d)):
                    part = tile * d + tx
                    if part < splits:
                        maximum = T.max(
                            maximum, parts[(index * splits + part) * (d + 2) + d]
                        )
                maximum_reduce(scratch, tx, maximum)
                maximum = scratch[0]
                for tile in T.serial(T.ceildiv(splits, d)):
                    part = tile * d + tx
                    if part < splits:
                        base = (index * splits + part) * (d + 2)
                        factor = T.exp(parts[base + d] - maximum)
                        weights[part] = factor
                        normalizer += parts[base + d + 1] * factor
                T.sync_threads()
                sum_reduce(scratch, tx, normalizer)
                normalizer = scratch[0]
            for part in T.serial(splits):
                result += parts[(index * splits + part) * (d + 2) + tx] * weights[part]
            out[index * d + tx] = T.if_then_else(
                enabled[index // heads] != 0, result / T.max(normalizer, 1e-30), 0
            )
        return out

    return compute


def window_decode(spec, schedule):
    """Coalesced warp-per-key dots avoid transposing short decode KV tiles."""
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import registers, vector

    p = {**spec.parameters, **schedule}
    rows, heads, kh, d, cap, chunk, window, splits = (
        p[n] for n in ("r", "h", "kh", "d", "capacity", "chunk", "window", "splits")
    )
    if chunk != 1 or d not in (64, 128):
        raise ValueError(
            "coalesced circular attention requires decode and 64/128 channels"
        )
    shape = (rows, kh, cap, d)
    unroll = p.get("key_unroll", 1)
    if unroll not in (1, 2, 4, 8):
        raise ValueError("decode key unroll must be 1, 2, 4 or 8")

    @T.macro
    def exchange(value, offset):
        return T.call_extern("float32", "subgroupShuffleXor", value, T.uint32(offset))

    @T.macro
    def score_key(score: T.Ref, q0, q1, q2, q3, kc, row, head, token, lane, end):
        physical = token % cap
        score = q0 * T.cast(
            kc[row, head // (heads // kh), physical, lane], "float32"
        ) + q1 * T.cast(kc[row, head // (heads // kh), physical, lane + 32], "float32")
        if d == 128:
            score += q2 * T.cast(
                kc[row, head // (heads // kh), physical, lane + 64], "float32"
            ) + q3 * T.cast(
                kc[row, head // (heads // kh), physical, lane + 96], "float32"
            )
        for step in T.unroll(5):
            score += exchange(score, 16 >> step)
        score = T.if_then_else(token < end, score * d**-0.5, -1e30)

    def score_keys(scores, q0, q1, q2, q3, kc, row, head, base, lane, end):
        for j in range(unroll):
            score_key(
                scores[j,][0], q0, q1, q2, q3, kc, row, head, base + j * 4, lane, end
            )

    @T.macro
    def max_key(peak: T.Ref, score):
        peak = T.max(peak, score)

    def max_keys(peak, scores):
        for j in range(unroll):
            max_key(peak, scores[j,][0])

    @T.macro
    def probability(value: T.Ref, score, peak, token, end):
        value = T.if_then_else(token < end, T.exp(score - peak), 0)

    @T.macro
    def sum_probability(norm: T.Ref, value):
        norm += value

    def probabilities(probs, scores, peak, base, end, norm):
        for j in range(unroll):
            probability(probs[j,][0], scores[j,][0], peak, base + j * 4, end)
            sum_probability(norm, probs[j,][0])

    def probability_vector(probs):
        return vector([probs[j,][0] for j in range(min(unroll, 4))])

    def upper_probability_vector(probs):
        return vector([probs[j,][0] for j in range(4, 8)])

    def value_vector(vc, row, head, base, channel, end):
        return vector(
            [
                T.if_then_else(
                    base + j * 4 < end,
                    T.cast(
                        vc[row, head // (heads // kh), (base + j * 4) % cap, channel],
                        "float32",
                    ),
                    0,
                )
                for j in range(min(unroll, 4))
            ]
        )

    def upper_value_vector(vc, row, head, base, channel, end):
        return vector(
            [
                T.if_then_else(
                    base + j * 4 < end,
                    T.cast(
                        vc[row, head // (heads // kh), (base + j * 4) % cap, channel],
                        "float32",
                    ),
                    0,
                )
                for j in range(4, 8)
            ]
        )

    @T.macro
    def add_value(acc: T.Ref, probs, vc, row, head, base, channel, end):
        left, right = (
            probability_vector(probs),
            value_vector(vc, row, head, base, channel, end),
        )
        if unroll == 1:
            acc += left * right
        else:
            acc += T.call_extern("float32", "dot", left, right)
        if unroll == 8:
            acc += T.call_extern(
                "float32",
                "dot",
                upper_probability_vector(probs),
                upper_value_vector(vc, row, head, base, channel, end),
            )

    @kernel
    def compute(
        q: Tensor((rows, heads * d), "float32"),
        kc: Tensor(shape, "float16"),
        vc: Tensor(shape, "float16"),
        positions: Tensor((rows,), "int32"),
        enabled: Tensor((rows,), "int32"),
    ) -> Tensor((rows * heads * splits * (d + 2),), "float32"):
        out = output((rows * heads * splits * (d + 2),), "float32")
        with T.Kernel(heads * rows, splits, threads=128) as (index, part):
            row, head = index // heads, index % heads
            tx = T.get_thread_binding()
            lane, warp = tx % 32, tx // 32
            first = T.max(0, positions[row] - window + 1) if window else 0
            last = T.if_then_else(enabled[row] != 0, positions[row] + 1, first)
            span = T.ceildiv(T.ceildiv(T.max(last - first, 0), 32), splits) * 32
            begin, end = first + part * span, T.min(last, first + (part + 1) * span)
            maxima = T.alloc_shared((4,), "float32")
            sums = T.alloc_shared((4,), "float32")
            values = T.alloc_shared((4, d), "float32")
            maximum, normalizer = T.alloc_var("float32"), T.alloc_var("float32")
            a0, a1 = T.alloc_var("float32"), T.alloc_var("float32")
            a2, a3 = T.alloc_var("float32"), T.alloc_var("float32")
            maximum, normalizer, a0, a1, a2, a3 = -1e30, 0, 0, 0, 0, 0
            q0, q1 = q[row, head * d + lane], q[row, head * d + lane + 32]
            q2, q3 = T.alloc_var("float32"), T.alloc_var("float32")
            q2, q3 = 0, 0
            if d == 128:
                q2, q3 = q[row, head * d + lane + 64], q[row, head * d + lane + 96]
            for offset in T.serial(T.ceildiv(T.max(end - begin, 0), 4 * unroll)):
                token = begin + offset * (4 * unroll) + warp
                scores, probs = registers((unroll,)), registers((unroll,))
                score_keys(scores, q0, q1, q2, q3, kc, row, head, token, lane, end)
                next_max = T.alloc_var("float32")
                next_max = maximum
                max_keys(next_max, scores)
                correction = T.exp(maximum - next_max)
                normalizer = normalizer * correction
                a0, a1, a2, a3 = (
                    a0 * correction,
                    a1 * correction,
                    a2 * correction,
                    a3 * correction,
                )
                probabilities(probs, scores, next_max, token, end, normalizer)
                add_value(a0, probs, vc, row, head, token, lane, end)
                add_value(a1, probs, vc, row, head, token, lane + 32, end)
                if d == 128:
                    add_value(a2, probs, vc, row, head, token, lane + 64, end)
                    add_value(a3, probs, vc, row, head, token, lane + 96, end)
                maximum = next_max
            values[warp, lane], values[warp, lane + 32] = a0, a1
            if d == 128:
                values[warp, lane + 64], values[warp, lane + 96] = a2, a3
            if lane == 0:
                maxima[warp], sums[warp] = maximum, normalizer
            T.sync_threads()
            if warp == 0:
                peak = T.max(T.max(maxima[0], maxima[1]), T.max(maxima[2], maxima[3]))
                total = T.alloc_var("float32")
                total, a0, a1, a2, a3 = 0, 0, 0, 0, 0
                for other in T.unroll(4):
                    factor = T.exp(maxima[other] - peak)
                    total += sums[other] * factor
                    a0 += values[other, lane] * factor
                    a1 += values[other, lane + 32] * factor
                    if d == 128:
                        a2 += values[other, lane + 64] * factor
                        a3 += values[other, lane + 96] * factor
                base = (index * splits + part) * (d + 2)
                out[base + lane], out[base + lane + 32] = a0, a1
                if d == 128:
                    out[base + lane + 64], out[base + lane + 96] = a2, a3
                if lane == 0:
                    out[base + d], out[base + d + 1] = peak, total
        return out

    return compute


def tiled_prefill(spec: KernelSpec, schedule: Mapping[str, object]):
    """Share KV tiles across query rows and grouped heads with online softmax."""
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import registers, vector
    from akas.ops.attention.kernel import cache_reader

    p = {**spec.parameters, **schedule}
    rows, heads, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    chunk = p.get("chunk", 1)
    group = heads // kh
    bk, pad = p.get("block_k", 32), p.get("key_pad", 1)
    dot_width = p.get("dot_width", 4)
    dtype = p.get(
        "shared_dtype", "float16" if p["kv_dtype"] == "float16" else "float32"
    )
    packed = p.get("packed_shared", False)
    if packed and not (
        dtype == "float16" and p["kv_dtype"] == "float16" and p.get("unit_scales")
    ):
        raise ValueError("packed prefill staging requires unit-scale native F16 KV")
    bm = p.get("block_q", min(8, chunk))
    if dtype not in ("float16", "float32") or (
        dtype == "float16" and p["kv_dtype"] != "float16"
    ):
        raise ValueError("half staging requires native half KV")
    scalar_bytes = 2 if dtype == "float16" else 4

    def shared_bytes(q):
        return 4 * q * group * (d + 32) + scalar_bytes * (d * (bk + pad) + bk * d)

    while "block_q" not in p and bm > 4 and shared_bytes(bm) > 32768:
        bm //= 2
    if (
        d != 64
        or group not in (1, 2, 4, 8)
        or bm not in (4, 8, 16)
        or chunk % bm
        or rows % chunk
        or bk not in (16, 32)
        or pad not in (0, 1)
        or dot_width not in (1, 4)
        or shared_bytes(bm) > 32768
    ):
        raise ValueError("unsupported tiled WebGPU attention geometry")
    storage, read = cache_reader(p)
    scale_shape = storage.metadata[0].shape

    def reduction(maximum):
        @T.macro
        def reduce(value: T.Ref, scratch, row, head, lane):
            if T.call_extern("uint32", "tensor_subgroup_size") >= 32:
                for step in T.unroll(5):
                    other = T.call_extern(
                        "float32", "subgroupShuffleXor", value, T.uint32(16 >> step)
                    )
                    if maximum:
                        value = T.max(value, other)
                    else:
                        value = value + other
            else:
                scratch[row, head, lane] = value
                T.sync_threads()
                for step in T.unroll(5):
                    if lane < (16 >> step):
                        if maximum:
                            scratch[row, head, lane] = T.max(
                                scratch[row, head, lane],
                                scratch[row, head, lane + (16 >> step)],
                            )
                        else:
                            scratch[row, head, lane] += scratch[
                                row, head, lane + (16 >> step)
                            ]
                    T.sync_threads()
                value = scratch[row, head, 0]
                T.sync_threads()

        return reduce

    maximum, total = reduction(True), reduction(False)

    def query_vector(queries, i, warp, channel):
        return vector(
            [queries[i, warp, channel * dot_width + j] for j in range(dot_width)]
        )

    def key_vector(keys, channel, lane):
        if packed:
            return vector(
                [
                    T.call_extern(
                        "float32",
                        "tensor_unpack_f16",
                        keys[(channel * dot_width + j) // 2, lane]
                        >> (((channel * dot_width + j) % 2) * 16),
                    )
                    for j in range(dot_width)
                ]
            )
        return vector(
            [
                T.cast(keys[channel * dot_width + j, lane], "float32")
                for j in range(dot_width)
            ]
        )

    def probability_vector(scores, i, warp, key):
        return vector([scores[i, warp, key * dot_width + j] for j in range(dot_width)])

    def value_vector(values, key, lane):
        if packed:
            return vector(
                [
                    T.call_extern(
                        "float32",
                        "tensor_unpack_f16",
                        values[key * dot_width + j, lane // 2] >> ((lane % 2) * 16),
                    )
                    for j in range(dot_width)
                ]
            )
        return vector(
            [
                T.cast(values[key * dot_width + j, lane], "float32")
                for j in range(dot_width)
            ]
        )

    def span(positions, enabled, base):
        last = T.int32(-1)
        for i in range(bm):
            row = T.min(base + i, rows - 1)
            last = T.max(last, T.if_then_else(enabled[row] != 0, positions[row], -1))
        return T.max(0, T.min(last + 1, cap))

    @T.macro
    def probabilities(
        score: T.Ref,
        scores,
        positions,
        enabled,
        running_max: T.Ref,
        normalizer: T.Ref,
        correction: T.Ref,
        base,
        tile,
        warp,
        lane,
        i,
    ):
        row = T.min(base + i, rows - 1)
        valid = (
            base + i < rows
            and enabled[row] != 0
            and lane < bk
            and tile * bk + lane < cap
            and tile * bk + lane <= positions[row]
        )
        score = T.if_then_else(valid, score * 0.125, -T.infinity("float32"))
        peak = T.alloc_var("float32")
        peak = score
        maximum(peak, scores, i, warp, lane)
        next_max = T.alloc_var("float32")
        next_max = T.max(running_max, peak)
        correction = T.if_then_else(
            running_max == -T.infinity("float32"), 0, T.exp(running_max - next_max)
        )
        probability = T.alloc_var("float32")
        probability = T.if_then_else(valid, T.exp(score - next_max), 0)
        norm = T.alloc_var("float32")
        norm = probability
        total(norm, scores, i, warp, lane)
        normalizer = normalizer * correction + norm
        running_max = next_max
        scores[i, warp, lane] = probability

    def make_probabilities(
        queries,
        keys,
        scores,
        positions,
        enabled,
        maxima,
        norms,
        factors,
        base,
        tile,
        warp,
        lane,
    ):
        dots = registers((bm,))
        dot_tiles(queries, keys, dots, warp, lane)
        for i in range(bm):
            probabilities(
                dots[i,][0],
                scores,
                positions,
                enabled,
                maxima[i,][0],
                norms[i,][0],
                factors[i,][0],
                base,
                tile,
                warp,
                lane,
                i,
            )

    @T.macro
    def add_score(acc: T.Ref, queries, right, warp, channel, i):
        left = query_vector(queries, i, warp, channel)
        if dot_width == 1:
            acc += left * right
        else:
            acc += T.call_extern("float32", "dot", left, right)

    def score_rows(dots, queries, right, warp, channel):
        for i in range(bm):
            add_score(dots[i,][0], queries, right, warp, channel, i)

    @T.macro
    def dot_tiles(queries, keys, dots, warp, lane):
        for channel in T.serial(d // dot_width):
            right = T.alloc_var("float32x4" if dot_width == 4 else "float32")
            right = key_vector(keys, channel, lane)
            score_rows(dots, queries, right, warp, channel)

    @T.macro
    def rescale(acc: T.Ref, factor):
        acc *= factor

    @T.macro
    def add_value(acc0: T.Ref, acc1: T.Ref, scores, right0, right1, warp, key, i):
        left = probability_vector(scores, i, warp, key)
        if dot_width == 1:
            acc0 += left * right0
            acc1 += left * right1
        else:
            acc0 += T.call_extern("float32", "dot", left, right0)
            acc1 += T.call_extern("float32", "dot", left, right1)

    def value_rows(accum, scores, right0, right1, warp, key):
        for i in range(bm):
            add_value(
                accum[i, 0][0], accum[i, 1][0], scores, right0, right1, warp, key, i
            )

    @T.macro
    def value_tiles(accum, scores, values, warp, lane):
        for key in T.serial(bk // dot_width):
            right0 = T.alloc_var("float32x4" if dot_width == 4 else "float32")
            right1 = T.alloc_var("float32x4" if dot_width == 4 else "float32")
            right0 = value_vector(values, key, lane)
            right1 = value_vector(values, key, lane + 32)
            value_rows(accum, scores, right0, right1, warp, key)

    def accumulate(accum, factors, scores, values, warp, lane):
        for i in range(bm):
            for side in range(2):
                rescale(accum[i, side][0], factors[i,][0])
        value_tiles(accum, scores, values, warp, lane)

    @T.macro
    def store(out, enabled, value, norm, base, head, lane, i, side):
        row = base + i
        if row < rows:
            out[(row * heads + head) * d + lane + side * 32] = T.if_then_else(
                enabled[row] != 0, value / T.max(norm, T.float32(1e-20)), 0
            )

    def stores(out, enabled, accum, norms, base, head, lane):
        for i in range(bm):
            for side in range(2):
                store(
                    out,
                    enabled,
                    accum[i, side][0],
                    norms[i,][0],
                    base,
                    head,
                    lane,
                    i,
                    side,
                )

    def initialize_maxima(maxima):
        for i in range(bm):
            set_maximum(maxima[i,][0])

    @T.macro
    def set_maximum(value: T.Ref):
        value = -T.infinity("float32")

    @kernel
    def compute(
        q: Tensor((rows, heads * d), "float32"),
        kc: Tensor(storage.values.shape, storage.values.dtype),
        vc: Tensor(storage.values.shape, storage.values.dtype),
        ks: Tensor(scale_shape, "float32"),
        vs: Tensor(scale_shape, "float32"),
        positions: Tensor((rows,), "int32"),
        enabled: Tensor((rows,), "int32"),
    ) -> Tensor((rows * heads * d,), "float32"):
        out = output((rows * heads * d,), "float32", position=5)
        with T.Kernel(kh, T.ceildiv(rows, bm), threads=group * 32) as (kv_head, block):
            tx = T.get_thread_binding()
            warp, lane = tx // 32, tx % 32
            base = block * bm
            slot = base // chunk
            length = span(positions, enabled, base)
            queries = T.alloc_shared((bm, group, d), "float32")
            keys = T.alloc_shared(
                (d // 2 if packed else d, bk + pad), "uint32" if packed else dtype
            )
            values = T.alloc_shared(
                (bk, d // 2 if packed else d), "uint32" if packed else dtype
            )
            scores = T.alloc_shared((bm, group, 32), "float32")
            maxima, norms, factors = (
                registers((bm,)),
                registers((bm,)),
                registers((bm,)),
            )
            accum = registers((bm, 2))
            initialize_maxima(maxima)
            if packed:
                for load in T.serial(T.ceildiv(bm * group * d, group * 32)):
                    flat = load * (group * 32) + tx
                    i, channel = flat // d, flat % d
                    if i < bm * group:
                        queries[i // group, i % group, channel] = q[
                            (base + i // group),
                            (kv_head * group + i % group) * d + channel,
                        ]
            else:
                for i, channel in T.Parallel(bm * group, d):
                    queries[i // group, i % group, channel] = q[
                        (base + i // group), (kv_head * group + i % group) * d + channel
                    ]
            T.sync_threads()
            for tile in T.serial(T.truncdiv(length + bk - 1, bk)):
                if packed:
                    for load in T.serial(T.ceildiv(bk * (d // 2), group * 32)):
                        flat = load * (group * 32) + tx
                        key, pair = flat // (d // 2), flat % (d // 2)
                        position = tile * bk + key
                        a = T.if_then_else(
                            position < length,
                            read(kc, ks, slot, kv_head, position, pair * 2),
                            0,
                        )
                        b = T.if_then_else(
                            position < length,
                            read(kc, ks, slot, kv_head, position, pair * 2 + 1),
                            0,
                        )
                        c = T.if_then_else(
                            position < length,
                            read(vc, vs, slot, kv_head, position, pair * 2),
                            0,
                        )
                        e = T.if_then_else(
                            position < length,
                            read(vc, vs, slot, kv_head, position, pair * 2 + 1),
                            0,
                        )
                        keys[pair, key] = T.call_extern(
                            "uint32",
                            "pack2x16float",
                            T.call_extern("float32x2", "vec2<f32>", a, b),
                        )
                        values[key, pair] = T.call_extern(
                            "uint32",
                            "pack2x16float",
                            T.call_extern("float32x2", "vec2<f32>", c, e),
                        )
                else:
                    for key, channel in T.Parallel(bk, d):
                        position = tile * bk + key
                        keys[channel, key] = T.if_then_else(
                            position < length,
                            read(kc, ks, slot, kv_head, position, channel),
                            0,
                        )
                        values[key, channel] = T.if_then_else(
                            position < length,
                            read(vc, vs, slot, kv_head, position, channel),
                            0,
                        )
                T.sync_threads()
                make_probabilities(
                    queries,
                    keys,
                    scores,
                    positions,
                    enabled,
                    maxima,
                    norms,
                    factors,
                    base,
                    tile,
                    warp,
                    lane,
                )
                T.sync_threads()
                accumulate(accum, factors, scores, values, warp, lane)
                T.sync_threads()
            stores(out, enabled, accum, norms, base, kv_head * group + warp, lane)
        return out

    return compute.with_attr("tirx.noalias", True)


def _attention(p):
    import tilelang.language as T
    from akas.ops.attention.webgpu import _score_shuffle as score_shuffle
    from akas.kernels.reductions import tree_reduce
    from akas.kernels.webgpu import subgroup_reduce

    r, h, kh, d, cap = (p.get("r", 1), p["h"], p["kh"], p["d"], p["cap"])
    partitioned = r == 1 and p.get("attention_schedule") == "partitioned_values"
    channels, parts = (p["channels"], p["value_parts"]) if partitioned else (d, 1)
    threads = channels * parts if partitioned else 128
    if partitioned and (
        channels not in (16, 32, 64)
        or d % channels
        or parts not in (1, 2, 4, 8, 16)
        or (threads > 1024)
    ):
        raise ValueError("unsupported decode attention distribution")
    fused_scores = partitioned and p.get("fused_scores", False)
    if fused_scores and (threads % 32 or d % 32):
        raise ValueError("fused attention requires complete 32-lane reductions")
    split = (
        partitioned
        and (not fused_scores)
        or (not partitioned and p.get("sg") and (r == 1))
    )
    sg = partitioned or p.get("sg", False)
    max_reduce = subgroup_reduce(threads, "max") if sg else tree_reduce(threads, "max")
    sum_reduce = subgroup_reduce(threads) if sg else tree_reduce(threads)
    shuffle = score_shuffle()
    score_loop = T.unroll if partitioned else T.serial

    @T.macro
    def scores_serial(q, kc, scores, maximum: T.Ref, dot: T.Ref, head, row, pos, tx):
        for tile in T.serial(T.ceildiv(cap, threads)):
            token = tile * threads + tx
            if token < cap:
                dot = -T.infinity("float32")
                if token <= pos[0] + row:
                    dot = 0
                    for j in T.serial(d):
                        dot = dot + q[(row * h + head) * d + j] * T.cast(
                            kc[(token * kh + head // (h // kh)) * d + j], "float32"
                        )
                    dot = dot * d ** (-0.5)
                scores[token] = dot
                maximum = T.max(maximum, dot)

    @T.macro
    def scores_subgroup(q, kc, scores, maximum: T.Ref, dot: T.Ref, head, row, pos, tx):
        if not partitioned:
            for token in T.Parallel(cap):
                scores[token] = -T.infinity("float32")
            T.sync_threads()
        if T.call_extern("uint32", "tensor_subgroup_size") >= 32:
            lane = tx % 32
            for tile in T.serial(T.ceildiv(pos[0] + row + 1, threads // 32)):
                token = tile * (threads // 32) + tx // 32
                dot = 0
                if token <= pos[0] + row:
                    for j in score_loop(d // 32):
                        channel = j * 32 + lane
                        dot = dot + q[(row * h + head) * d + channel] * T.cast(
                            kc[(token * kh + head // (h // kh)) * d + channel],
                            "float32",
                        )
                shuffle(dot)
                if (lane == 0) & (token <= pos[0] + row):
                    scores[token] = dot * d ** (-0.5)
        elif partitioned:
            for tile in T.serial(T.ceildiv(pos[0] + 1, threads)):
                token = tile * threads + tx
                if token <= pos[0]:
                    dot = 0
                    for j in T.serial(d):
                        dot = dot + q[head * d + j] * T.cast(
                            kc[(token * kh + head // (h // kh)) * d + j], "float32"
                        )
                    scores[token] = dot * d ** (-0.5)
        else:
            scores_serial(q, kc, scores, maximum, dot, head, row, pos, tx)
        T.sync_threads()
        for tile in T.serial(T.ceildiv(pos[0] + row + 1, threads)):
            token = tile * threads + tx
            if token < cap:
                if not partitioned or token <= pos[0]:
                    maximum = T.max(maximum, scores[token])

    @T.macro
    def attention(q, vc, out, pos, kc=None):
        with T.Kernel(h, d // channels if partitioned else r, threads=threads) as (
            head,
            block,
        ):
            row = 0 if partitioned else block
            tx = T.get_thread_binding()
            dot = T.alloc_var("float32")
            maximum = T.alloc_var("float32")
            total = T.alloc_var("float32")
            result = T.alloc_var("float32")
            scores = T.alloc_shared((cap,), "float32")
            scratch = T.alloc_shared((threads,), "float32")
            if partitioned:
                partials = T.alloc_shared((threads,), "float32")
                channel = tx % channels
                part = tx // channels
            maximum = -T.infinity("float32")
            if split:
                for tile in T.serial(
                    T.ceildiv(pos[0] + row + 1 if sg else cap, threads)
                ):
                    token = tile * threads + tx
                    if token < cap:
                        if not partitioned or token <= pos[0]:
                            scores[token] = q[head * cap + token]
                            maximum = T.max(maximum, scores[token])
            elif sg:
                scores_subgroup(q, kc, scores, maximum, dot, head, row, pos, tx)
            else:
                scores_serial(q, kc, scores, maximum, dot, head, row, pos, tx)
            if sg:
                max_reduce(scratch, tx, maximum)
            else:
                scratch[tx] = maximum
                T.sync_threads()
                max_reduce(scratch, tx)
            total = 0
            for tile in T.serial(T.ceildiv(pos[0] + row + 1 if sg else cap, threads)):
                token = tile * threads + tx
                if token < cap:
                    if not partitioned or token <= pos[0]:
                        scores[token] = T.exp(scores[token] - scratch[0])
                        total = total + scores[token]
            T.sync_threads()
            if sg:
                sum_reduce(scratch, tx, total)
            else:
                scratch[tx] = total
                T.sync_threads()
                sum_reduce(scratch, tx)
            result = 0
            if partitioned:
                for tile in T.serial(T.ceildiv(pos[0] + 1, parts)):
                    token = tile * parts + part
                    if token <= pos[0]:
                        result = result + scores[token] * T.cast(
                            vc[
                                (token * kh + head // (h // kh)) * d
                                + block * channels
                                + channel
                            ],
                            "float32",
                        )
                partials[tx] = result
                T.sync_threads()
                if part == 0:
                    result = 0
                    for other in T.unroll(parts):
                        result = result + partials[other * channels + channel]
                    out[head * d + block * channels + channel] = result / scratch[0]
            elif tx < d:
                if sg:
                    for token in T.serial(pos[0] + row + 1):
                        result = result + scores[token] * T.cast(
                            vc[(token * kh + head // (h // kh)) * d + tx], "float32"
                        )
                else:
                    for token in T.serial(cap):
                        if token <= pos[0] + row:
                            result = result + scores[token] * T.cast(
                                vc[(token * kh + head // (h // kh)) * d + tx], "float32"
                            )
                out[(row * h + head) * d + tx] = result / scratch[0]

    if split:

        @kernel
        def compute(
            q: Tensor(h * cap, "float32"),
            vc: Tensor(cap * kh * d, "float16"),
            pos: Tensor(2, "int32"),
        ) -> Tensor(r * h * d, "float32"):
            out = output(r * h * d, "float32", position=2)
            attention(q, vc, out, pos)
            return out

        return compute
    if not split:

        @kernel
        def compute(
            q: Tensor(h * cap if split else r * h * d, "float32"),
            kc: Tensor(cap * kh * d, "float16"),
            vc: Tensor(cap * kh * d, "float16"),
            pos: Tensor(2, "int32"),
        ) -> Tensor(r * h * d, "float32"):
            out = output(r * h * d, "float32", position=3)
            attention(q, vc, out, pos, kc)
            return out

        return compute


def attention(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.attention.webgpu import _attention as attention_kernel

    p = {**spec.parameters, **schedule}
    return attention_kernel(p)


def qkv(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    raise ValueError("WebGPU uses separate query/cache normalization kernels")


def _score_shuffle():
    import tilelang.language as T

    def reduce(dot):
        for stride in (16, 8, 4, 2, 1):
            step(dot, stride)

    @T.macro
    def step(dot: T.Ref, stride):
        dot = dot + T.call_extern(
            "float32", "subgroupShuffleXor", dot, T.uint32(stride)
        )

    return reduce


def scores(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.attention.webgpu import _score_shuffle as score_shuffle

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    h, kh, d, cap = (p[n] for n in ("h", "kh", "d", "cap"))
    shuffle = score_shuffle()

    @kernel
    def compute(
        q: Tensor((h * d,), "float32"),
        kc: Tensor((cap * kh * d,), "float16"),
        pos: Tensor((2,), "int32"),
    ) -> Tensor((h * cap,), "float32"):
        out = output((h * cap,), "float32", position=2)
        with T.Kernel(h, T.ceildiv(cap, 32), threads=128) as (head, block):
            tx = T.get_thread_binding()
            dot = T.alloc_var("float32")
            if T.call_extern("uint32", "tensor_subgroup_size") >= 32:
                lane = tx % 32
                for tile in T.serial(8):
                    token = block * 32 + tile * 4 + tx // 32
                    dot = 0
                    if token <= pos[0]:
                        for j in T.serial(d // 32):
                            channel = j * 32 + lane
                            dot = dot + q[head * d + channel] * T.cast(
                                kc[(token * kh + head // (h // kh)) * d + channel],
                                "float32",
                            )
                    shuffle(dot)
                    if (lane == 0) & (token < cap):
                        out[head * cap + token] = T.if_then_else(
                            token <= pos[0], dot * d ** (-0.5), -T.infinity("float32")
                        )
            elif tx < 32:
                token = block * 32 + tx
                if token < cap:
                    dot = -T.infinity("float32")
                    if token <= pos[0]:
                        dot = 0
                        for j in T.serial(d):
                            dot = dot + q[head * d + j] * T.cast(
                                kc[(token * kh + head // (h // kh)) * d + j], "float32"
                            )
                        dot = dot * d ** (-0.5)
                    out[head * cap + token] = dot
        return out

    return compute


def attention_scores(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.attention.webgpu import scores as attention_scores_kernel
    from akas.kernels.definition import KernelSpec

    p = {**spec.parameters, **schedule}
    return attention_scores_kernel(KernelSpec("scores", p), {})


def qkv_mapped_webgpu(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    heads, kh, d, cap, eps, theta = (
        p[n] for n in ("heads", "kv_heads", "d", "capacity", "eps", "theta")
    )
    projection = 2 * heads * d + 2 * kh * d
    if p.get("kv_dtype") == "fp8":
        from akas.kernels.reductions import tree_reduce
        from akas.quantization._shared.e4m3 import portable_scaling

        portable_scale, encode_scaled = portable_scaling()
        maximum = tree_reduce(128, "max")

        @kernel
        def compute(
            proj: Tensor((rows, projection), "float32"),
            qw: Tensor((d,), "float16"),
            kw: Tensor((d,), "float16"),
            mapping: Tensor((slots,), "int32"),
            positions: Tensor((rows,), "int32"),
            active: Tensor((rows,), "int32"),
            kc: Tensor((pool, kh, cap, d // 4), "uint32"),
            vc: Tensor((pool, kh, cap, d // 4), "uint32"),
            ks: Tensor((pool, kh, cap, 2), "float32"),
            vs: Tensor((pool, kh, cap, 2), "float32"),
        ) -> tuple[
            Tensor((pool, kh, cap, d // 4), "uint32"),
            Tensor((pool, kh, cap, d // 4), "uint32"),
            Tensor((pool, kh, cap, 2), "float32"),
            Tensor((pool, kh, cap, 2), "float32"),
            Tensor((rows, heads, d), "float16"),
        ]:
            q = output((rows, heads, d), "float16")
            with T.Kernel(rows, heads + kh, threads=256) as (row, head):
                tx = T.get_thread_binding()
                normal = T.alloc_shared((d,), "float16")
                rotated = T.alloc_shared((d,), "float16")
                vals_v = T.alloc_shared((d,), "float16")
                scratch_k = T.alloc_shared((128,), "float32")
                scratch_v = T.alloc_shared((128,), "float32")
                scales_k = T.alloc_shared((2,), "float32")
                scales_v = T.alloc_shared((2,), "float32")
                squares = T.alloc_fragment((d,), "float32")
                total = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(d):
                    offset = T.if_then_else(
                        head < heads, head * 2 * d, 2 * heads * d + (head - heads) * d
                    )
                    value = T.cast(T.cast(proj[row, offset + j], "float16"), "float32")
                    squares[j] = value * value
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(d):
                    offset = T.if_then_else(
                        head < heads, head * 2 * d, 2 * heads * d + (head - heads) * d
                    )
                    value = T.cast(T.cast(proj[row, offset + j], "float16"), "float32")
                    weight = T.if_then_else(head < heads, qw[j], kw[j])
                    normal[j] = (
                        value
                        * T.rsqrt(total[0] / d + eps)
                        * (1 + T.cast(weight, "float32"))
                    )
                T.sync_threads()
                for j in T.Parallel(d):
                    value = T.alloc_var("float32")
                    value = T.cast(normal[j], "float32")
                    if j < 64:
                        angle = positions[row] * T.exp(
                            T.cast(-2 * (j % 32), "float32")
                            / 64
                            * T.log(T.float32(theta))
                        )
                        pair = T.cast(normal[(j + 32) % 64], "float32")
                        value = value * T.cos(angle) + T.if_then_else(
                            j < 32, -pair, pair
                        ) * T.sin(angle)
                    rotated[j] = value
                    vals_v[j] = T.cast(
                        proj[
                            row,
                            2 * heads * d + kh * d + T.max(head - heads, 0) * d + j,
                        ],
                        "float16",
                    )
                    if (active[row] != 0) & (head < heads):
                        q[row, head, j] = rotated[j]
                T.sync_threads()
                for block in T.serial(2):
                    if tx < 128:
                        scratch_k[tx] = T.abs(
                            T.cast(rotated[block * 128 + tx], "float32")
                        )
                        scratch_v[tx] = T.abs(
                            T.cast(vals_v[block * 128 + tx], "float32")
                        )
                    T.sync_threads()
                    maximum(scratch_k, tx)
                    maximum(scratch_v, tx)
                    if tx == 0:
                        scales_k[block] = portable_scale(scratch_k[0])
                        scales_v[block] = portable_scale(scratch_v[0])
                        if (active[row] != 0) & (head >= heads):
                            ks[
                                mapping[row // chunk],
                                head - heads,
                                positions[row],
                                block,
                            ] = scales_k[block]
                            vs[
                                mapping[row // chunk],
                                head - heads,
                                positions[row],
                                block,
                            ] = scales_v[block]
                    T.sync_threads()
                if (tx < 64) & (active[row] != 0) & (head >= heads):
                    word_k = T.alloc_var("uint32")
                    word_v = T.alloc_var("uint32")
                    word_k = 0
                    word_v = 0
                    blk = tx // 32
                    for i in T.serial(4):
                        ch = tx * 4 + i
                        word_k = word_k | (
                            encode_scaled(T.cast(rotated[ch], "float32"), scales_k[blk])
                            << (i * 8)
                        )
                        word_v = word_v | (
                            encode_scaled(T.cast(vals_v[ch], "float32"), scales_v[blk])
                            << (i * 8)
                        )
                    kc[mapping[row // chunk], head - heads, positions[row], tx] = word_k
                    vc[mapping[row // chunk], head - heads, positions[row], tx] = word_v
            return (kc, vc, ks, vs, q)

        return compute

    @kernel
    def compute(
        proj: Tensor((rows, projection), "float32"),
        qw: Tensor((d,), "float16"),
        kw: Tensor((d,), "float16"),
        mapping: Tensor((slots,), "int32"),
        positions: Tensor((rows,), "int32"),
        active: Tensor((rows,), "int32"),
        kc: Tensor((pool, kh, cap, d), "float16"),
        vc: Tensor((pool, kh, cap, d), "float16"),
    ) -> tuple[
        Tensor((pool, kh, cap, d), "float16"),
        Tensor((pool, kh, cap, d), "float16"),
        Tensor((rows, heads, d), "float16"),
    ]:
        q = output((rows, heads, d), "float16")
        with T.Kernel(rows, heads + kh, threads=256) as (row, head):
            if active[row] != 0:
                values = T.alloc_shared((d,), "float16")
                squares = T.alloc_fragment((d,), "float32")
                total = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(d):
                    raw = T.cast(
                        T.cast(
                            proj[
                                row,
                                T.if_then_else(
                                    head < heads,
                                    head * 2 * d,
                                    2 * heads * d + (head - heads) * d,
                                )
                                + j,
                            ],
                            "float16",
                        ),
                        "float32",
                    )
                    squares[j] = raw * raw
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(d):
                    raw = T.cast(
                        T.cast(
                            proj[
                                row,
                                T.if_then_else(
                                    head < heads,
                                    head * 2 * d,
                                    2 * heads * d + (head - heads) * d,
                                )
                                + j,
                            ],
                            "float16",
                        ),
                        "float32",
                    )
                    weight = T.if_then_else(head < heads, qw[j], kw[j])
                    values[j] = (
                        raw
                        * T.rsqrt(total[0] / d + eps)
                        * (1 + T.cast(weight, "float32"))
                    )
                for j in T.Parallel(d):
                    value = T.alloc_var("float32")
                    value = T.cast(values[j], "float32")
                    if j < 64:
                        angle = positions[row] * T.exp(
                            T.cast(-2 * (j % 32), "float32")
                            / 64
                            * T.log(T.float32(theta))
                        )
                        pair = T.cast(values[(j + 32) % 64], "float32")
                        value = value * T.cos(angle) + T.if_then_else(
                            j < 32, -pair, pair
                        ) * T.sin(angle)
                    if head < heads:
                        q[row, head, j] = value
                    else:
                        kc[mapping[row // chunk], head - heads, positions[row], j] = (
                            value
                        )
                        vc[mapping[row // chunk], head - heads, positions[row], j] = (
                            T.cast(
                                proj[
                                    row, 2 * heads * d + kh * d + (head - heads) * d + j
                                ],
                                "float16",
                            )
                        )
        return (kc, vc, q)

    return compute


def mapped_webgpu(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    r = p.get("r", 1)
    slots, chunk, pool = (p.get("slots", r), p.get("chunk", 1), p.get("pool", 256))
    rows = slots * chunk
    heads, kh, d, cap = (p[n] for n in ("heads", "kv_heads", "d", "capacity"))
    group = heads // kh
    projection = 2 * heads * d + 2 * kh * d
    reduce_sum = tree_reduce(256)
    quantized = p.get("kv_dtype") == "fp8"
    if quantized:
        from akas.quantization.kv.fp8.kernel import reference_decoder

        read_kv = reference_decoder(provider="webgpu")
    else:

        @T.macro
        def read_kv(cache, metadata, slot, head, pos, ch):
            return T.cast(cache[slot, head, pos, ch], "float32")

    @T.macro
    def body(q, kc, vc, ks, vs, proj, mapping, positions, lengths, out):
        with T.Kernel(rows, heads, threads=256) as (row, head):
            tx = T.get_thread_binding()
            batch = row // chunk
            t = row % chunk
            request = mapping[batch]
            valid_row = (request >= 0) & (t < lengths[batch])
            count = T.if_then_else(valid_row, T.min(cap, positions[batch] + t + 1), 0)
            kv_head = head // group
            scratch = T.alloc_shared((256,), "float32")
            running_max = T.alloc_shared((1,), "float32")
            normalizer = T.alloc_shared((1,), "float32")
            factor = T.alloc_shared((1,), "float32")
            prob = T.alloc_shared((1,), "float32")
            acc = T.alloc_var("float32")
            acc = 0
            q_val = T.cast(q[row, head, tx], "float32")
            if tx == 0:
                running_max[0] = -T.infinity("float32")
                normalizer[0] = 0
            T.sync_threads()
            for pos in T.serial(count):
                r_slot = T.max(request, 0)
                k_val = T.cast(
                    T.cast(read_kv(kc, ks, r_slot, kv_head, pos, tx), "float16"),
                    "float32",
                )
                v_val = T.cast(
                    T.cast(read_kv(vc, vs, r_slot, kv_head, pos, tx), "float16"),
                    "float32",
                )
                scratch[tx] = q_val * k_val
                T.sync_threads()
                reduce_sum(scratch, tx)
                if tx == 0:
                    score = scratch[0] * d ** (-0.5)
                    next_max = T.max(running_max[0], score)
                    factor[0] = T.if_then_else(
                        running_max[0] > -T.infinity("float32"),
                        T.exp(running_max[0] - next_max),
                        0,
                    )
                    prob[0] = T.cast(
                        T.cast(T.exp(score - next_max), "float16"), "float32"
                    )
                    normalizer[0] = normalizer[0] * factor[0] + prob[0]
                    running_max[0] = next_max
                T.sync_threads()
                acc = acc * factor[0] + prob[0] * v_val
                T.sync_threads()
            if valid_row:
                gate = T.cast(
                    T.cast(proj[row, head * 2 * d + d + tx], "float16"), "float32"
                )
                normal = T.cast(T.cast(acc / normalizer[0], "float16"), "float32")
                out[row, head * d + tx] = normal / (1 + T.exp(-gate))
            else:
                out[row, head * d + tx] = 0

    if quantized:

        @kernel
        def compute(
            q: Tensor((rows, heads, d), "float16"),
            kc: Tensor((pool, kh, cap, d // 4), "uint32"),
            vc: Tensor((pool, kh, cap, d // 4), "uint32"),
            ks: Tensor((pool, kh, cap, 2), "float32"),
            vs: Tensor((pool, kh, cap, 2), "float32"),
            proj: Tensor((rows, projection), "float32"),
            mapping: Tensor((slots,), "int32"),
            positions: Tensor((slots,), "int32"),
            lengths: Tensor((slots,), "int32"),
        ) -> Tensor((rows, heads * d), "float16"):
            out = output((rows, heads * d), "float16")
            body(q, kc, vc, ks, vs, proj, mapping, positions, lengths, out)
            return out

        return compute

    @kernel
    def compute(
        q: Tensor((rows, heads, d), "float16"),
        kc: Tensor((pool, kh, cap, d), "float16"),
        vc: Tensor((pool, kh, cap, d), "float16"),
        proj: Tensor((rows, projection), "float32"),
        mapping: Tensor((slots,), "int32"),
        positions: Tensor((slots,), "int32"),
        lengths: Tensor((slots,), "int32"),
    ) -> Tensor((rows, heads * d), "float16"):
        out = output((rows, heads * d), "float16")
        body(q, kc, vc, kc, vc, proj, mapping, positions, lengths, out)
        return out

    return compute


from .kernel import (
    attention as shared_attention,
    attention_flat,
    cache_extent,
    cache_pad,
    cache_write,
    chunk_active,
    merge_floating,
    partial_floating,
    prepare_flat,
    rope,
)
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))


def blocked_partial(spec, schedule):
    """Batch independent key dots before updating online softmax state."""
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import registers
    from akas.ops.attention.kernel import cache_reader

    p = {**spec.parameters, **schedule}
    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    splits, unroll = p.get("splits", 16), p.get("key_unroll", 4)
    if d != 64 or h % kh or unroll not in (2, 4, 8):
        raise ValueError("unsupported register-tiled attention")
    storage, read = cache_reader(p)

    @T.macro
    def exchange(value, offset):
        return T.call_extern("float32", "subgroupShuffleXor", value, T.uint32(offset))

    @T.macro
    def score_key(score: T.Ref, q0, q1, kc, ks, slot, head, token, lane, end):
        score = T.if_then_else(
            token < end,
            q0 * read(kc, ks, slot, head, token, lane)
            + q1 * read(kc, ks, slot, head, token, lane + 32),
            0,
        )
        for step in T.unroll(5):
            score += exchange(score, 16 >> step)
        score = T.if_then_else(token < end, score * 0.125, -T.infinity("float32"))

    def score_keys(scores, q0, q1, kc, ks, slot, head, base, lane, end):
        for j in range(unroll):
            score_key(
                scores[j,][0], q0, q1, kc, ks, slot, head, base + j * 4, lane, end
            )

    @T.macro
    def max_key(peak: T.Ref, score):
        peak = T.max(peak, score)

    def maximum_key(peak, scores):
        for j in range(unroll):
            max_key(peak, scores[j,][0])

    @T.macro
    def accumulate(
        a0: T.Ref,
        a1: T.Ref,
        norm: T.Ref,
        score,
        peak,
        vc,
        vs,
        slot,
        head,
        token,
        lane,
        end,
    ):
        probability = T.if_then_else(token < end, T.exp(score - peak), 0)
        a0 += probability * T.if_then_else(
            token < end, read(vc, vs, slot, head, token, lane), 0
        )
        a1 += probability * T.if_then_else(
            token < end, read(vc, vs, slot, head, token, lane + 32), 0
        )
        norm += probability

    def accumulate_keys(
        a0, a1, norm, scores, peak, vc, vs, slot, head, base, lane, end
    ):
        for j in range(unroll):
            accumulate(
                a0,
                a1,
                norm,
                scores[j,][0],
                peak,
                vc,
                vs,
                slot,
                head,
                base + j * 4,
                lane,
                end,
            )

    @kernel
    def compute(
        q: Tensor((r, h * d), "float32"),
        kc: Tensor(storage.values.shape, storage.values.dtype),
        vc: Tensor(storage.values.shape, storage.values.dtype),
        ks: Tensor(storage.metadata[0].shape, "float32"),
        vs: Tensor(storage.metadata[0].shape, "float32"),
        positions: Tensor(r, "int32"),
        enabled: Tensor(r, "int32"),
    ) -> Tensor(r * h * splits * 66, "float32"):
        out = output(r * h * splits * 66, "float32", position=5)
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
            maximum, normalizer = T.alloc_var("float32"), T.alloc_var("float32")
            a0, a1 = T.alloc_var("float32"), T.alloc_var("float32")
            maximum, normalizer, a0, a1 = -T.infinity("float32"), 0, 0, 0
            q0, q1 = q[slot, head * d + lane], q[slot, head * d + lane + 32]
            scores = registers((unroll,))
            for tile in T.serial(T.ceildiv(T.max(end - begin, 0), 4 * unroll)):
                base = begin + tile * 4 * unroll + warp
                score_keys(
                    scores, q0, q1, kc, ks, slot, head // (h // kh), base, lane, end
                )
                peak = T.alloc_var("float32")
                peak = maximum
                maximum_key(peak, scores)
                correction = T.if_then_else(
                    maximum == -T.infinity("float32"), 0, T.exp(maximum - peak)
                )
                normalizer, a0, a1 = (
                    normalizer * correction,
                    a0 * correction,
                    a1 * correction,
                )
                accumulate_keys(
                    a0,
                    a1,
                    normalizer,
                    scores,
                    peak,
                    vc,
                    vs,
                    slot,
                    head // (h // kh),
                    base,
                    lane,
                    end,
                )
                maximum = peak
            values[warp, lane], values[warp, lane + 32] = a0, a1
            if lane == 0:
                maxima[warp], sums[warp] = maximum, normalizer
            T.sync_threads()
            if warp == 0:
                merged = T.max(T.max(maxima[0], maxima[1]), T.max(maxima[2], maxima[3]))
                norm, result0, result1 = (
                    T.alloc_var("float32"),
                    T.alloc_var("float32"),
                    T.alloc_var("float32"),
                )
                norm, result0, result1 = 0, 0, 0
                for other in T.unroll(4):
                    factor = T.if_then_else(
                        end > begin, T.exp(maxima[other] - merged), 0
                    )
                    norm += sums[other] * factor
                    result0 += values[other, lane] * factor
                    result1 += values[other, lane + 32] * factor
                offset = (index * splits + split) * 66
                out[offset + lane], out[offset + lane + 32] = result0, result1
                if lane == 0:
                    out[offset + 64], out[offset + 65] = merged, norm
        return out

    return compute


def channel_partial(spec, schedule):
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import registers, vector
    from akas.ops.attention.kernel import cache_reader

    p = {**spec.parameters, **schedule}
    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    group, splits, bk = h // kh, p.get("splits", 16), p.get("key_rows", 16)
    adaptive = spec.kind == "attention_adaptive_partial"
    granule = p.get("granule", 512)
    packed_shared = p.get("packed_shared", False)
    chains = p.get("dot_chains", 1)
    if chains not in (1, 4):
        raise ValueError("attention dot chains must be one or four")
    if adaptive and (granule not in (64, 128, 512) or granule % bk):
        raise ValueError("unsupported adaptive cache block size")
    if d != 64 or h % kh or group not in (1, 2, 4, 8) or bk not in (16, 32, 64, 128):
        raise ValueError("unsupported channel-cooperative attention")
    if p["kv_dtype"] != "float16" or not p.get("unit_scales", False):
        raise ValueError("channel-cooperative attention requires unit-scale half KV")
    if bk == 128 and not packed_shared:
        raise ValueError("128-token tiles require packed shared KV")
    lanes = 2 if bk >= 64 else 64 // bk
    terms = d // lanes
    storage, read = cache_reader(p)

    @T.macro
    def exchange(value, stride):
        return T.call_extern("float32", "subgroupShuffleXor", value, T.uint32(stride))

    @T.macro
    def load_query(value: T.Ref, q, slot, head, channel):
        value = q[slot, head * d + channel]

    def load_queries(queries, q, slot, head, channel_group):
        for j in range(terms):
            load_query(queries[j,][0], q, slot, head, channel_group * terms + j)

    def tile_range(length, split):
        if adaptive:
            begin = split * granule
            return (
                begin,
                length,
                T.ceildiv(T.max(length - begin, 0), splits * granule) * (granule // bk),
            )
        span = T.ceildiv(length, splits)
        begin, end = split * span, T.min((split + 1) * span, length)
        return begin, end, T.ceildiv(T.max(end - begin, 0), bk)

    def tile_start(begin, tile):
        if adaptive:
            return (
                begin
                + tile // (granule // bk) * (splits * granule)
                + tile % (granule // bk) * bk
            )
        return begin + tile * bk

    @T.macro
    def dot_term(score: T.Ref, left, right):
        score += T.call_extern("float32", "dot", left, right)

    def dot_keys(score, queries, keys, key, channel_group):
        if chains == 4:
            partial = registers((4,))
            for j in range(terms // 4):
                for component in range(4):
                    channel = channel_group * terms + j * 4 + component
                    right = (
                        T.call_extern(
                            "float32",
                            "tensor_unpack_f16",
                            keys[(channel // 2) * (bk + 1) + key]
                            >> (16 * (channel % 2)),
                        )
                        if packed_shared
                        else T.cast(keys[channel * (bk + 1) + key], "float32")
                    )
                    fma_term(
                        partial[component,][0], queries[j * 4 + component,][0], right
                    )
            combine_chains(score, *(partial[j,][0] for j in range(4)))
            return
        for j in range(terms // 4):
            left = vector([queries[j * 4 + k,][0] for k in range(4)])
            if packed_shared:
                right = vector(
                    [
                        T.call_extern(
                            "float32",
                            "tensor_unpack_f16",
                            keys[
                                (channel_group * (terms // 2) + j * 2 + k // 2)
                                * (bk + 1)
                                + key
                            ]
                            >> (16 * (k % 2)),
                        )
                        for k in range(4)
                    ]
                )
            else:
                right = vector(
                    [
                        T.cast(
                            keys[(channel_group * terms + j * 4 + k) * (bk + 1) + key],
                            "float32",
                        )
                        for k in range(4)
                    ]
                )
            dot_term(score, left, right)

    def dot_values(acc, probabilities, values, keys, head, channel):
        if chains == 4:
            partial = registers((4,))
            for j in range(bk // 4):
                for component in range(4):
                    token = j * 4 + component
                    word = (
                        (
                            keys[token * (d // 2) + channel // 2]
                            if bk == 128
                            else values[token, channel // 2]
                        )
                        if packed_shared
                        else None
                    )
                    right = (
                        T.call_extern(
                            "float32", "tensor_unpack_f16", word >> ((channel % 2) * 16)
                        )
                        if packed_shared
                        else T.cast(values[token, channel], "float32")
                    )
                    fma_term(partial[component,][0], probabilities[head, token], right)
            combine_chains(acc, *(partial[j,][0] for j in range(4)))
            return
        for j in range(bk // 4):
            left = vector([probabilities[head, j * 4 + k] for k in range(4)])
            if packed_shared:
                right = vector(
                    [
                        T.call_extern(
                            "float32",
                            "tensor_unpack_f16",
                            (
                                keys[(j * 4 + k) * (d // 2) + channel // 2]
                                if bk == 128
                                else values[j * 4 + k, channel // 2]
                            )
                            >> ((channel % 2) * 16),
                        )
                        for k in range(4)
                    ]
                )
            else:
                right = vector(
                    [T.cast(values[j * 4 + k, channel], "float32") for k in range(4)]
                )
            dot_term(acc, left, right)

    @T.macro
    def fma_term(acc: T.Ref, left, right):
        acc = T.call_extern("float32", "fma", left, right, acc)

    @T.macro
    def combine_chains(acc: T.Ref, a, b, c, e):
        acc += (a + b) + (c + e)

    @T.macro
    def raw_score(scores, probabilities, head, token):
        if bk >= 64:
            return scores[head, token]
        else:
            return probabilities[head, token]

    @kernel
    def compute(
        q: Tensor((r, h * d), "float32"),
        kc: Tensor(storage.values.shape, storage.values.dtype),
        vc: Tensor(storage.values.shape, storage.values.dtype),
        ks: Tensor(storage.metadata[0].shape, "float32"),
        vs: Tensor(storage.metadata[0].shape, "float32"),
        positions: Tensor(r, "int32"),
        enabled: Tensor(r, "int32"),
    ) -> Tensor(r * h * splits * 66, "float32"):
        out = output(r * h * splits * 66, "float32", position=5)
        with T.Kernel(kh, r, splits, threads=group * 64) as (kv_head, slot, split):
            tx = T.get_thread_binding()
            head, local = tx // 64, tx % 64
            key, channel_group = local // lanes, local % lanes
            length = T.if_then_else(
                enabled[slot] != 0, T.min(positions[slot] + 1, cap), 0
            )
            begin, end, tile_count = tile_range(length, split)
            keys = T.alloc_shared(
                ((d // 2 if packed_shared else d) * (bk + 1),),
                "uint32" if packed_shared else "float16",
            )
            values = T.alloc_shared(
                (
                    1 if bk == 128 else bk,
                    1 if bk == 128 else d // 2 if packed_shared else d,
                ),
                "uint32" if packed_shared else "float16",
            )
            probabilities = T.alloc_shared((group, bk), "float32")
            scores = T.alloc_shared(
                (group if bk >= 64 else 1, bk if bk >= 64 else 1), "float32"
            )
            queries = registers((terms,))
            if not adaptive:
                load_queries(queries, q, slot, kv_head * group + head, channel_group)
            elif begin < length:
                load_queries(queries, q, slot, kv_head * group + head, channel_group)
            maximum, normalizer, acc = (
                T.alloc_var("float32"),
                T.alloc_var("float32"),
                T.alloc_var("float32"),
            )
            maximum, normalizer, acc = -T.infinity("float32"), 0, 0
            for tile in T.serial(tile_count):
                base = tile_start(begin, tile)
                if base < end:
                    if packed_shared:
                        for load in T.serial(T.ceildiv(bk * (d // 2), group * 64)):
                            flat = load * (group * 64) + tx
                            token, pair = flat // (d // 2), flat % (d // 2)
                            a = T.if_then_else(
                                base + token < end,
                                read(kc, ks, slot, kv_head, base + token, pair * 2),
                                0,
                            )
                            b = T.if_then_else(
                                base + token < end,
                                read(kc, ks, slot, kv_head, base + token, pair * 2 + 1),
                                0,
                            )
                            if bk != 128:
                                c = T.if_then_else(
                                    base + token < end,
                                    read(vc, vs, slot, kv_head, base + token, pair * 2),
                                    0,
                                )
                                e = T.if_then_else(
                                    base + token < end,
                                    read(
                                        vc,
                                        vs,
                                        slot,
                                        kv_head,
                                        base + token,
                                        pair * 2 + 1,
                                    ),
                                    0,
                                )
                            keys[pair * (bk + 1) + token] = T.call_extern(
                                "uint32",
                                "pack2x16float",
                                T.call_extern("float32x2", "vec2<f32>", a, b),
                            )
                            if bk != 128:
                                values[token, pair] = T.call_extern(
                                    "uint32",
                                    "pack2x16float",
                                    T.call_extern("float32x2", "vec2<f32>", c, e),
                                )
                    else:
                        for load in T.serial(T.ceildiv(bk * d, group * 64)):
                            flat = load * (group * 64) + tx
                            token, channel = flat // d, flat % d
                            keys[channel * (bk + 1) + token] = T.if_then_else(
                                base + token < end,
                                read(kc, ks, slot, kv_head, base + token, channel),
                                0,
                            )
                            values[token, channel] = T.if_then_else(
                                base + token < end,
                                read(vc, vs, slot, kv_head, base + token, channel),
                                0,
                            )
                    T.sync_threads()
                    for score_tile in T.unroll(bk // 32 if bk >= 64 else 1):
                        key_index = key + score_tile * 32
                        score = T.alloc_var("float32")
                        score = 0
                        dot_keys(score, queries, keys, key_index, channel_group)
                        for step in T.unroll(lanes.bit_length() - 1):
                            score += exchange(score, lanes >> (step + 1))
                        if channel_group == 0:
                            if bk >= 64:
                                scores[head, key_index] = T.if_then_else(
                                    base + key_index < end,
                                    score * 0.125,
                                    -T.infinity("float32"),
                                )
                            else:
                                probabilities[head, key_index] = T.if_then_else(
                                    base + key_index < end,
                                    score * 0.125,
                                    -T.infinity("float32"),
                                )
                    T.sync_threads()
                    if bk == 128:
                        for load in T.serial(T.ceildiv(bk * (d // 2), group * 64)):
                            flat = load * (group * 64) + tx
                            token, pair = flat // (d // 2), flat % (d // 2)
                            a = T.if_then_else(
                                base + token < end,
                                read(vc, vs, slot, kv_head, base + token, pair * 2),
                                0,
                            )
                            b = T.if_then_else(
                                base + token < end,
                                read(vc, vs, slot, kv_head, base + token, pair * 2 + 1),
                                0,
                            )
                            keys[flat] = T.call_extern(
                                "uint32",
                                "pack2x16float",
                                T.call_extern("float32x2", "vec2<f32>", a, b),
                            )
                    probability = T.alloc_var("float32")
                    probability = raw_score(scores, probabilities, head, local % bk)
                    peak = T.alloc_var("float32")
                    peak = probability
                    if bk == 128:
                        peak = T.max(peak, scores[head, local + 64])
                    if bk >= 64:
                        if T.call_extern("uint32", "tensor_subgroup_size") >= 64:
                            peak = T.max(peak, exchange(peak, 32))
                        else:
                            peak = T.max(peak, scores[head, local ^ 32])
                            if bk == 128:
                                peak = T.max(peak, scores[head, (local ^ 32) + 64])
                    for step in T.unroll(min(bk, 32).bit_length() - 1):
                        peak = T.max(peak, exchange(peak, min(bk, 32) >> (step + 1)))
                    next_max = T.max(maximum, peak)
                    correction = T.if_then_else(
                        maximum == -T.infinity("float32"), 0, T.exp(maximum - next_max)
                    )
                    probability = T.if_then_else(
                        base + local % bk < end, T.exp(probability - next_max), 0
                    )
                    norm = T.alloc_var("float32")
                    norm = probability
                    second = T.alloc_var("float32")
                    if bk == 128:
                        second = T.if_then_else(
                            base + local + 64 < end,
                            T.exp(scores[head, local + 64] - next_max),
                            0,
                        )
                        norm += second
                    if bk >= 64:
                        if T.call_extern("uint32", "tensor_subgroup_size") >= 64:
                            norm += exchange(probability, 32)
                            if bk == 128:
                                norm += exchange(second, 32)
                        else:
                            norm += T.if_then_else(
                                base + (local ^ 32) < end,
                                T.exp(scores[head, local ^ 32] - next_max),
                                0,
                            )
                            if bk == 128:
                                norm += T.if_then_else(
                                    base + (local ^ 32) + 64 < end,
                                    T.exp(scores[head, (local ^ 32) + 64] - next_max),
                                    0,
                                )
                    for step in T.unroll(min(bk, 32).bit_length() - 1):
                        norm += exchange(norm, min(bk, 32) >> (step + 1))
                    normalizer = normalizer * correction + norm
                    maximum = next_max
                    if local < bk:
                        probabilities[head, local] = probability
                    if bk == 128:
                        probabilities[head, local + 64] = second
                    acc *= correction
                    T.sync_threads()
                    dot_values(acc, probabilities, values, keys, head, local)
                    T.sync_threads()
            offset = ((slot * h + kv_head * group + head) * splits + split) * 66
            out[offset + local] = acc
            if local == 0:
                out[offset + 64], out[offset + 65] = maximum, normalizer
        return out

    return compute


def packed_channel_partial(spec, schedule):
    return channel_partial(spec, {**schedule, "packed_shared": True})


def two_pass_partial(spec, schedule):
    """Bounded shared scores: one K pass, one softmax, one coalesced V pass."""
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import registers, vector
    from akas.ops.attention.kernel import cache_reader

    p = {**spec.parameters, **schedule}
    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    group, splits = h // kh, p.get("splits", 128)
    granule, lanes = p.get("granule", 128), p.get("key_lanes", 8)
    unroll = p.get("value_unroll", 8)
    chains = p.get("value_chains", 4)
    cached = p.get("stage_cache", False)
    bk = p.get("cache_rows", 64)
    cache_dtype = p.get("cache_dtype", "float16")
    cache_bytes = bk * 65 * (2 if cache_dtype == "float16" else 4) if cached else 0
    extent = (cap + splits * granule - 1) // (splits * granule) * granule
    if (
        d != 64
        or h % kh
        or group not in (1, 2, 4, 8)
        or lanes not in (4, 8, 16)
        or unroll not in (4, 8, 16)
        or chains not in (1, 4)
        or unroll % chains
        or granule not in (64, 128, 512)
        or (cap > 16384 and cap % 128)
        or (
            cached and (bk not in (32, 64) or cache_dtype not in ("float16", "float32"))
        )
        or ((extent + 1) * group + group * 2) * 4 + cache_bytes > 32768
    ):
        raise ValueError(
            "unsupported two-pass decode geometry, capacity alignment or shared storage"
        )
    if p["kv_dtype"] != "float16" or not p.get("unit_scales", False):
        raise ValueError("two-pass decode requires unit-scale half KV")
    keys_per_step, terms = 64 // lanes, d // lanes
    storage, read = cache_reader(p)

    @T.macro
    def read(cache, metadata, slot, head, token, channel):
        return T.cast(cache[slot, head, token, channel], "float32")

    @T.macro
    def exchange(value, stride):
        return T.call_extern("float32", "subgroupShuffleXor", value, T.uint32(stride))

    @T.macro
    def query(value: T.Ref, q, slot, head, channel):
        value = q[slot, head * d + channel]

    def load_query(values, q, slot, head, lane):
        for j in range(terms):
            query(values[j,][0], q, slot, head, lane * terms + j)

    @T.macro
    def add_dot(score: T.Ref, left, right):
        score += T.call_extern("float32", "dot", left, right)

    def score_dot(score, values, kc, ks, slot, kv_head, token, lane, cache=None, key=0):
        for j in range(terms // 4):
            left = vector([values[j * 4 + k,][0] for k in range(4)])
            if cached:
                right = vector(
                    [
                        T.cast(cache[key, lane * terms + j * 4 + k], "float32")
                        for k in range(4)
                    ]
                )
            else:
                right = vector(
                    [
                        read(
                            kc,
                            ks,
                            slot,
                            kv_head,
                            T.min(token, cap - 1),
                            lane * terms + j * 4 + k,
                        )
                        for k in range(4)
                    ]
                )
            add_dot(score, left, right)

    @T.macro
    def add_value(acc: T.Ref, weight, value):
        acc += weight * T.cast(value, "float32")

    def value_step(
        accum,
        scores,
        vc,
        vs,
        slot,
        kv_head,
        head,
        channel,
        tile,
        length,
        count,
        begin,
        cache=None,
        base=0,
    ):
        for u in range(unroll):
            index = base + tile * unroll + u
            token = begin + index // granule * (splits * granule) + index % granule
            if cached:
                add_value(
                    accum[u % chains,][0],
                    scores[head, index],
                    cache[tile * unroll + u, channel],
                )
            else:
                with T.If((index < count) & (token < length)):
                    with T.Then():
                        add_value(
                            accum[u % chains,][0],
                            scores[head, index],
                            read(vc, vs, slot, kv_head, T.min(token, cap - 1), channel),
                        )

    @T.macro
    def load_cache(cache, staged, slot, kv_head, base, length):
        tx = T.get_thread_binding()
        for load in T.serial(T.ceildiv(bk * d, group * 64)):
            flat = load * (group * 64) + tx
            token, channel = flat // d, flat % d
            staged[token, channel] = T.if_then_else(
                base + token < length,
                T.cast(
                    cache[slot, kv_head, T.min(base + token, cap - 1), channel],
                    cache_dtype,
                ),
                0,
            )

    @T.macro
    def sum_values(value: T.Ref, other):
        value += other

    def finish_values(accum, value):
        for chain in range(chains):
            sum_values(value, accum[chain,][0])

    @kernel
    def compute(
        q: Tensor((r, h * d), "float32"),
        kc: Tensor(storage.values.shape, storage.values.dtype),
        vc: Tensor(storage.values.shape, storage.values.dtype),
        ks: Tensor(storage.metadata[0].shape, "float32"),
        vs: Tensor(storage.metadata[0].shape, "float32"),
        positions: Tensor(r, "int32"),
        enabled: Tensor(r, "int32"),
    ) -> Tensor(r * h * splits * 66, "float32"):
        out = output(r * h * splits * 66, "float32", position=5)
        with T.Kernel(kh, r, splits, threads=group * 64) as (kv_head, slot, split):
            tx = T.get_thread_binding()
            head, local = tx // 64, tx % 64
            lane = local % lanes
            length = T.if_then_else(
                enabled[slot] != 0, T.min(positions[slot] + 1, cap), 0
            )
            begin = split * granule
            count = T.ceildiv(T.max(length - begin, 0), splits * granule) * granule
            scores = T.alloc_shared((group, extent + 1), "float32")
            reductions = T.alloc_shared((group, 2), "float32")
            if cached:
                cache_tile = T.alloc_shared((bk, d + 1), cache_dtype)
            queries = registers((terms,))
            accum = registers((chains,))
            maximum = T.alloc_var("float32")
            normalizer = T.alloc_var("float32")
            result = T.alloc_var("float32")
            maximum, normalizer, result = -T.infinity("float32"), 0, 0
            if begin < length:
                load_query(queries, q, slot, kv_head * group + head, lane)
                if cached:
                    for tile in T.serial(T.ceildiv(count, bk)):
                        base_index = tile * bk
                        base = (
                            begin
                            + base_index // granule * (splits * granule)
                            + base_index % granule
                        )
                        load_cache(kc, cache_tile, slot, kv_head, base, length)
                        T.sync_threads()
                        for item in T.serial(bk // keys_per_step):
                            key = item * keys_per_step + local // lanes
                            index, token = base_index + key, base + key
                            score = T.alloc_var("float32")
                            score = 0
                            score_dot(
                                score,
                                queries,
                                kc,
                                ks,
                                slot,
                                kv_head,
                                token,
                                lane,
                                cache_tile,
                                key,
                            )
                            for step in T.unroll(lanes.bit_length() - 1):
                                score += exchange(score, lanes >> (step + 1))
                            score = T.if_then_else(
                                token < length,
                                score * 0.125 * ks[slot, kv_head, 0, 0],
                                -T.infinity("float32"),
                            )
                            maximum = T.max(maximum, score)
                            if lane == 0:
                                scores[head, index] = score
                        T.sync_threads()
                else:
                    for tile in T.serial(T.ceildiv(count, keys_per_step)):
                        index = tile * keys_per_step + local // lanes
                        token = (
                            begin
                            + index // granule * (splits * granule)
                            + index % granule
                        )
                        score = T.alloc_var("float32")
                        score = 0
                        if token < length:
                            score_dot(
                                score, queries, kc, ks, slot, kv_head, token, lane
                            )
                        for step in T.unroll(lanes.bit_length() - 1):
                            score += exchange(score, lanes >> (step + 1))
                        score = T.if_then_else(
                            token < length,
                            score * 0.125 * ks[slot, kv_head, 0, 0],
                            -T.infinity("float32"),
                        )
                        maximum = T.max(maximum, score)
                        if lane == 0:
                            scores[head, index] = score
                for step in T.unroll(5):
                    maximum = T.max(maximum, exchange(maximum, 16 >> step))
                if local % 32 == 0:
                    reductions[head, local // 32] = maximum
                T.sync_threads()
                maximum = T.max(reductions[head, 0], reductions[head, 1])
                for tile in T.serial(T.ceildiv(count, 64)):
                    index = tile * 64 + local
                    if index < count:
                        probability = T.exp(scores[head, index] - maximum)
                        scores[head, index] = probability
                        normalizer += probability
                for step in T.unroll(5):
                    normalizer += exchange(normalizer, 16 >> step)
                if local % 32 == 0:
                    reductions[head, local // 32] = normalizer
                T.sync_threads()
                normalizer = reductions[head, 0] + reductions[head, 1]
                if cached:
                    for tile in T.serial(T.ceildiv(count, bk)):
                        base_index = tile * bk
                        base = (
                            begin
                            + base_index // granule * (splits * granule)
                            + base_index % granule
                        )
                        load_cache(vc, cache_tile, slot, kv_head, base, length)
                        T.sync_threads()
                        for item in T.serial(bk // unroll):
                            value_step(
                                accum,
                                scores,
                                vc,
                                vs,
                                slot,
                                kv_head,
                                head,
                                local,
                                item,
                                length,
                                count,
                                begin,
                                cache_tile,
                                base_index,
                            )
                        T.sync_threads()
                else:
                    for tile in T.serial(T.ceildiv(count, unroll)):
                        value_step(
                            accum,
                            scores,
                            vc,
                            vs,
                            slot,
                            kv_head,
                            head,
                            local,
                            tile,
                            length,
                            count,
                            begin,
                        )
                finish_values(accum, result)
                result *= vs[slot, kv_head, 0, 0]
            offset = ((slot * h + kv_head * group + head) * splits + split) * 66
            out[offset + local] = result
            if local == 0:
                out[offset + 64], out[offset + 65] = maximum, normalizer
        return out

    return compute


def buffered_partial(spec, schedule):
    return two_pass_partial(spec, {**schedule, "stage_cache": True})


def stream_merge(spec, schedule):
    """Reduce split statistics without a splits-by-depth shared product array."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, h, d, splits = (p[n] for n in ("r", "h", "d", "splits"))
    if d != 64 or splits > 512:
        raise ValueError("streaming merge requires D64 and at most 512 splits")

    @T.macro
    def exchange(value, stride):
        return T.call_extern("float32", "subgroupShuffleXor", value, T.uint32(stride))

    @kernel
    def compute(
        parts: Tensor(r * h * splits * 66, "float32"), enabled: Tensor(r, "int32")
    ) -> Tensor(r * h * d, "float32"):
        out = output(r * h * d, "float32", position=1)
        with T.Kernel(r * h, threads=64) as head:
            tx = T.get_thread_binding()
            weights = T.alloc_shared((splits,), "float32")
            reductions = T.alloc_shared((2,), "float32")
            limits = T.alloc_shared((2,), "float32")
            active_end = T.alloc_var("float32")
            maximum, normalizer, result = (
                T.alloc_var("float32"),
                T.alloc_var("float32"),
                T.alloc_var("float32"),
            )
            maximum, normalizer, result = -T.infinity("float32"), 0, 0
            active_end = 0
            for tile in T.serial(T.ceildiv(splits, 64)):
                part = tile * 64 + tx
                if part < splits:
                    if parts[(head * splits + part) * 66 + 65] > 0:
                        maximum = T.max(
                            maximum, parts[(head * splits + part) * 66 + 64]
                        )
                        active_end = T.max(active_end, T.cast(part + 1, "float32"))
            for step in T.unroll(5):
                maximum = T.max(maximum, exchange(maximum, 16 >> step))
                active_end = T.max(active_end, exchange(active_end, 16 >> step))
            if tx % 32 == 0:
                reductions[tx // 32] = maximum
                limits[tx // 32] = active_end
            T.sync_threads()
            maximum = T.max(reductions[0], reductions[1])
            last = T.cast(T.max(limits[0], limits[1]), "int32")
            for tile in T.serial(T.ceildiv(last, 64)):
                part = tile * 64 + tx
                if part < last:
                    norm = parts[(head * splits + part) * 66 + 65]
                    factor = T.if_then_else(
                        norm > 0,
                        T.exp(parts[(head * splits + part) * 66 + 64] - maximum),
                        0,
                    )
                    weights[part] = factor
                    normalizer += norm * factor
            for step in T.unroll(5):
                normalizer += exchange(normalizer, 16 >> step)
            if tx % 32 == 0:
                reductions[tx // 32] = normalizer
            T.sync_threads()
            normalizer = reductions[0] + reductions[1]
            for part in T.serial(last):
                result += parts[(head * splits + part) * 66 + tx] * weights[part]
            out[head * d + tx] = T.if_then_else(
                (enabled[head // h] != 0) & (normalizer > 0), result / normalizer, 0
            )
        return out

    return compute


def cooperative_scores(spec, schedule):
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import registers, vector

    p = {**spec.parameters, **schedule}
    if p.get("score_tile") == 256:
        return cooperative_scores_wide(spec, schedule)
    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    group = h // kh
    tile = p.get("score_tile", 64)
    key_dtype = "float32" if p.get("score_float_shared", False) else "float16"
    if tile not in (32, 64, 128) or (tile >= 64 and key_dtype == "float32"):
        raise ValueError("unsupported cooperative score tile/shared storage")
    if d not in (64, 128) or cap % tile or h % kh or group not in (1, 2, 4, 8):
        raise ValueError("unsupported cooperative score geometry")

    def query_vector(q, queries, row, kv_head, head, band, chain):
        if tile == 128:
            return vector(
                [
                    q[row, (kv_head * group + head) * d + band * 16 + chain * 4 + j]
                    for j in range(4)
                ]
            )
        return vector([queries[head, band * 16 + chain * 4 + j] for j in range(4)])

    def key_vector(keys, lane, band, chain):
        if tile == 128:
            return vector(
                [
                    T.cast(
                        keys[
                            band * 16 + chain * 4 + j,
                            lane ^ ((band * 16 + chain * 4 + j) % tile),
                        ],
                        "float32",
                    )
                    for j in range(4)
                ]
            )
        return vector(
            [T.cast(keys[band * 16 + chain * 4 + j, lane], "float32") for j in range(4)]
        )

    @T.macro
    def dot_term(acc: T.Ref, q, queries, keys, row, kv_head, head, lane, band, chain):
        acc += T.call_extern(
            "float32",
            "dot",
            query_vector(q, queries, row, kv_head, head, band, chain),
            key_vector(keys, lane, band, chain),
        )

    def dot_terms(accum, q, queries, keys, row, kv_head, head, lane, band):
        for chain in range(4):
            dot_term(
                accum[chain,][0],
                q,
                queries,
                keys,
                row,
                kv_head,
                head,
                lane,
                band,
                chain,
            )

    bands = T.unroll if p.get("score_unroll", False) else T.serial

    @kernel
    def compute(
        q: Tensor((r, h * d), "float32"),
        kc: Tensor((r, kh, cap, d), "float16"),
        positions: Tensor((r,), "int32"),
        enabled: Tensor((r,), "int32"),
    ) -> Tensor((r * h * cap,), "float32"):
        out = output((r * h * cap,), "float32")
        with T.Kernel(kh, cap // tile, r, threads=group * tile) as (
            kv_head,
            block,
            row,
        ):
            tx = T.get_thread_binding()
            head, lane = tx // tile, tx % tile
            if tile == 128:
                queries = T.alloc_local((1, 1), "float32")
                keys = T.alloc_shared((d, tile), key_dtype)
            else:
                queries = T.alloc_shared((group, d), "float32")
                keys = T.alloc_shared((d, tile + 1), key_dtype)
                for query, channel in T.Parallel(group, d):
                    queries[query, channel] = q[
                        row, (kv_head * group + query) * d + channel
                    ]
            for key, channel in T.Parallel(tile, d):
                if tile == 128:
                    keys[channel, key ^ (channel % tile)] = T.if_then_else(
                        (enabled[row] != 0) & (block * tile + key <= positions[row]),
                        kc[row, kv_head, block * tile + key, channel],
                        0,
                    )
                else:
                    keys[channel, key] = T.cast(
                        T.if_then_else(
                            (enabled[row] != 0)
                            & (block * tile + key <= positions[row]),
                            kc[row, kv_head, block * tile + key, channel],
                            0,
                        ),
                        key_dtype,
                    )
            T.sync_threads()
            accum = registers((4,))
            for band in bands(d // 16):
                dot_terms(accum, q, queries, keys, row, kv_head, head, lane, band)
            score = (accum[0,][0] + accum[1,][0]) + (accum[2,][0] + accum[3,][0])
            out[(row * h + kv_head * group + head) * cap + block * tile + lane] = (
                T.if_then_else(
                    (enabled[row] != 0) & (block * tile + lane <= positions[row]),
                    score * d**-0.5,
                    -1e30,
                )
            )
        return out

    return compute


def cooperative_scores_wide(spec, schedule):
    """256 keys per group; stage 64 channels at a time within 32 KiB LDS."""
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import registers, vector

    p = {**spec.parameters, **schedule}
    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    group = h // kh
    if d not in (64, 128) or cap % 256 or h % kh or group not in (1, 2, 4):
        raise ValueError("unsupported wide cooperative score geometry")

    def query_vector(q, row, head, dimension, band, chain):
        return vector(
            [
                q[row, head * d + dimension * 64 + band * 16 + chain * 4 + j]
                for j in range(4)
            ]
        )

    def key_vector(keys, lane, band, chain):
        return vector(
            [
                T.cast(
                    keys[band * 16 + chain * 4 + j, lane ^ (band * 16 + chain * 4 + j)],
                    "float32",
                )
                for j in range(4)
            ]
        )

    @T.macro
    def dot_term(acc: T.Ref, q, keys, row, head, lane, dimension, band, chain):
        acc += T.call_extern(
            "float32",
            "dot",
            query_vector(q, row, head, dimension, band, chain),
            key_vector(keys, lane, band, chain),
        )

    def dot_terms(accum, q, keys, row, head, lane, dimension, band):
        for chain in range(4):
            dot_term(accum[chain,][0], q, keys, row, head, lane, dimension, band, chain)

    bands = T.unroll if p.get("score_unroll", False) else T.serial

    @kernel
    def compute(
        q: Tensor((r, h * d), "float32"),
        kc: Tensor((r, kh, cap, d), "float16"),
        positions: Tensor((r,), "int32"),
        enabled: Tensor((r,), "int32"),
    ) -> Tensor((r * h * cap,), "float32"):
        out = output((r * h * cap,), "float32")
        with T.Kernel(kh, cap // 256, r, threads=group * 256) as (kv_head, block, row):
            tx = T.get_thread_binding()
            head, lane = tx // 256, tx % 256
            keys = T.alloc_shared((64, 256), "float16")
            accum = registers((4,))
            if (enabled[row] != 0) & (block * 256 <= positions[row]):
                for dimension in T.serial(d // 64):
                    for key, channel in T.Parallel(256, 64):
                        keys[channel, key ^ channel] = T.if_then_else(
                            block * 256 + key <= positions[row],
                            kc[
                                row,
                                kv_head,
                                block * 256 + key,
                                dimension * 64 + channel,
                            ],
                            0,
                        )
                    T.sync_threads()
                    for band in bands(4):
                        dot_terms(
                            accum,
                            q,
                            keys,
                            row,
                            kv_head * group + head,
                            lane,
                            dimension,
                            band,
                        )
                    T.sync_threads()
            score = (accum[0,][0] + accum[1,][0]) + (accum[2,][0] + accum[3,][0])
            out[(row * h + kv_head * group + head) * cap + block * 256 + lane] = (
                T.if_then_else(
                    (enabled[row] != 0) & (block * 256 + lane <= positions[row]),
                    score * d**-0.5,
                    -1e30,
                )
            )
        return out

    return compute


def group_reduce(maximum):
    import tilelang.language as T

    @T.macro
    def reduce(value: T.Ref, scratch, tx, lane):
        if T.call_extern("uint32", "tensor_subgroup_size") >= 64:
            for step in T.unroll(6):
                other = T.call_extern(
                    "float32", "subgroupShuffleXor", value, T.uint32(32 >> step)
                )
                if maximum:
                    value = T.max(value, other)
                else:
                    value += other
        else:
            scratch[tx] = value
            T.sync_threads()
            for step in T.unroll(6):
                if lane < (32 >> step):
                    if maximum:
                        scratch[tx] = T.max(scratch[tx], scratch[tx + (32 >> step)])
                    else:
                        scratch[tx] += scratch[tx + (32 >> step)]
                T.sync_threads()
            value = scratch[tx - lane]

    return reduce


def cooperative_values(spec, schedule):
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import registers, vector

    p = {**spec.parameters, **schedule}
    r, h, kh, d, cap, splits = (
        p[n] for n in ("r", "h", "kh", "d", "capacity", "splits")
    )
    group = h // kh
    extent = ((cap + splits * 64 - 1) // (splits * 64)) * 64
    value_tile = p.get("value_tile", 64)
    value_dtype = "float32" if p.get("value_float_shared", False) else "float16"
    if value_tile not in (16, 32, 64):
        raise ValueError("unsupported cooperative value tile")
    if d not in (64, 128) or cap % 64 or h % kh or group not in (1, 2, 4, 8):
        raise ValueError("unsupported cooperative value geometry")
    if (
        group * extent * 4
        + value_tile * d * (4 if value_dtype == "float32" else 2)
        + group * 64 * 4
        > 32768
    ):
        raise ValueError(
            "cooperative value shared storage exceeds 32 KiB; increase splits"
        )
    maximum_reduce, sum_reduce = group_reduce(True), group_reduce(False)

    def probability_vector(probs, head, tile, band, chain):
        return vector(
            [
                probs[head, tile * value_tile + band * 16 + chain * 4 + j]
                for j in range(4)
            ]
        )

    def value_vector(values, lane, band, chain, side):
        return vector(
            [
                T.cast(values[band * 16 + chain * 4 + j, lane + side * 64], "float32")
                for j in range(4)
            ]
        )

    @T.macro
    def dot_term(acc: T.Ref, probs, values, head, lane, tile, band, chain, side):
        acc += T.call_extern(
            "float32",
            "dot",
            probability_vector(probs, head, tile, band, chain),
            value_vector(values, lane, band, chain, side),
        )

    def dot_terms(accum, probs, values, head, lane, tile, band):
        for side in range(d // 64):
            for chain in range(4):
                dot_term(
                    accum[side, chain][0],
                    probs,
                    values,
                    head,
                    lane,
                    tile,
                    band,
                    chain,
                    side,
                )

    @T.macro
    def store_value(out, value, row, kv_head, head, part, lane, side):
        offset = ((row * h + kv_head * group + head) * splits + part) * (d + 2)
        out[offset + lane + side * 64] = value

    def store_values(out, accum, row, kv_head, head, part, lane):
        for side in range(d // 64):
            value = (accum[side, 0][0] + accum[side, 1][0]) + (
                accum[side, 2][0] + accum[side, 3][0]
            )
            store_value(out, value, row, kv_head, head, part, lane, side)

    @kernel
    def compute(
        scores: Tensor((r * h * cap,), "float32"),
        vc: Tensor((r, kh, cap, d), "float16"),
        positions: Tensor((r,), "int32"),
        enabled: Tensor((r,), "int32"),
    ) -> Tensor((r * h * splits * (d + 2),), "float32"):
        out = output((r * h * splits * (d + 2),), "float32")
        with T.Kernel(kh, r, splits, threads=group * 64) as (kv_head, row, part):
            tx = T.get_thread_binding()
            head, lane = tx // 64, tx % 64
            length = T.if_then_else(
                enabled[row] != 0, T.min(positions[row] + 1, cap), 0
            )
            span = T.ceildiv(T.ceildiv(length, 64), splits) * 64
            begin, end = part * span, T.min((part + 1) * span, length)
            probs = T.alloc_shared((group, extent), "float32")
            values = T.alloc_shared((value_tile, d), value_dtype)
            scratch = T.alloc_shared((group * 64,), "float32")
            maximum, norm = T.alloc_var("float32"), T.alloc_var("float32")
            maximum, norm = -1e30, 0
            for tile in T.serial(extent // 64):
                index = tile * 64 + lane
                score = T.if_then_else(
                    begin + index < end,
                    scores[
                        (row * h + kv_head * group + head) * cap
                        + T.min(begin + index, cap - 1)
                    ],
                    -1e30,
                )
                maximum = T.max(maximum, score)
            maximum_reduce(maximum, scratch, tx, lane)
            for tile in T.serial(extent // 64):
                index = tile * 64 + lane
                score = T.if_then_else(
                    begin + index < end,
                    scores[
                        (row * h + kv_head * group + head) * cap
                        + T.min(begin + index, cap - 1)
                    ],
                    -1e30,
                )
                probability = T.if_then_else(
                    begin + index < end, T.exp(score - maximum), 0
                )
                probs[head, index] = probability
                norm += probability
            T.sync_threads()
            sum_reduce(norm, scratch, tx, lane)
            accum = registers((d // 64, 4))
            for tile in T.serial(T.ceildiv(T.max(end - begin, 0), value_tile)):
                for key, channel in T.Parallel(value_tile, d):
                    index = begin + tile * value_tile + key
                    values[key, channel] = T.cast(
                        T.if_then_else(
                            index < end,
                            vc[row, kv_head, T.min(index, cap - 1), channel],
                            0,
                        ),
                        value_dtype,
                    )
                T.sync_threads()
                for band in T.unroll(value_tile // 16):
                    dot_terms(accum, probs, values, head, lane, tile, band)
                T.sync_threads()
            store_values(out, accum, row, kv_head, head, part, lane)
            if lane == 0:
                offset = ((row * h + kv_head * group + head) * splits + part) * (d + 2)
                out[offset + d] = maximum
                out[offset + d + 1] = norm
        return out

    return compute


def cooperative_merge(spec, schedule):
    import tilelang.language as T
    from tensor.compiler.webgpu.schedules.registers import registers

    p = {**spec.parameters, **schedule}
    r, h, d, splits, groups = (p[n] for n in ("r", "h", "d", "splits", "merge_splits"))
    count = splits // groups
    if splits % groups or count not in (16, 32, 64) or d not in (64, 128):
        raise ValueError("unsupported hierarchical attention merge geometry")
    from akas.kernels.webgpu import subgroup_reduce

    reduce_max, reduce_sum = subgroup_reduce(d, "max"), subgroup_reduce(d)

    @T.macro
    def product(acc: T.Ref, parts, weights, index, group, tx, part):
        acc += (
            parts[((index * splits + group * count + part) * (d + 2)) + tx]
            * weights[part]
        )

    def products(accum, parts, weights, index, group, tx, band):
        for chain in range(4):
            product(
                accum[chain,][0], parts, weights, index, group, tx, band * 4 + chain
            )

    @kernel
    def compute(parts: Tensor((r * h * splits * (d + 2),), "float32")) -> Tensor(
        (r * h * groups * (d + 2),), "float32"
    ):
        out = output((r * h * groups * (d + 2),), "float32")
        with T.Kernel(r * h, groups, threads=d) as (index, group):
            tx = T.get_thread_binding()
            weights = T.alloc_shared((count,), "float32")
            scratch = T.alloc_shared((d,), "float32")
            part = tx % count
            base = (index * splits + group * count + part) * (d + 2)
            maximum, norm = T.alloc_var("float32"), T.alloc_var("float32")
            size = T.cast(T.call_extern("uint32", "tensor_subgroup_size"), "int32")
            if T.min(size, d) >= count:
                maximum = T.call_extern("float32", "subgroupMax", parts[base + d])
                factor = T.exp(parts[base + d] - maximum)
                norm = T.call_extern(
                    "float32", "subgroupAdd", parts[base + d + 1] * factor
                )
                norm *= T.cast(count, "float32") / T.cast(T.min(size, d), "float32")
                if tx < count:
                    weights[tx] = factor
                T.sync_threads()
            else:
                maximum = parts[base + d]
                reduce_max(scratch, tx, maximum)
                maximum = scratch[0]
                factor = T.exp(parts[base + d] - maximum)
                if tx < count:
                    weights[tx] = factor
                norm = parts[base + d + 1] * factor * T.cast(count, "float32") / d
                reduce_sum(scratch, tx, norm)
                norm = scratch[0]
            accum = registers((4,))
            for band in T.unroll(count // 4):
                products(accum, parts, weights, index, group, tx, band)
            result = (accum[0,][0] + accum[1,][0]) + (accum[2,][0] + accum[3,][0])
            offset = (index * groups + group) * (d + 2)
            out[offset + tx] = result
            if tx == 0:
                out[offset + d] = maximum
                out[offset + d + 1] = norm
        return out

    return compute


def projected_gate_merge(spec, schedule):
    import tilelang.language as T
    from akas.kernels.webgpu import subgroup_reduce
    from akas.quantization.weights.ggml.definition import weight_storage
    from akas.quantization._shared.packing import packed_loads

    p = {**spec.parameters, **schedule}
    r, h, d, splits, k = (p[n] for n in ("r", "h", "d", "splits", "k"))
    if p["type"] != 2 or d not in (64, 128) or k % d:
        raise ValueError(
            "projected attention gate requires Q4_0 and complete D64/D128 rows"
        )
    count, _ = weight_storage(2, h * k, packed_words=True)
    lanes = min(d, k // 8)
    if k % (lanes * 8):
        raise ValueError(
            "projected attention gate requires complete packed vector tiles"
        )
    planar = p.get("weight_layout", "raw") == "planar"
    _, half_value, word_value = packed_loads()
    sum_reduce, max_reduce = subgroup_reduce(d), subgroup_reduce(d, "max")

    @T.macro
    def vector_product(acc: T.Ref, x, row, index, scale, packed, component):
        left = T.call_extern(
            "float32x4",
            "vec4<f32>",
            x[row * k + index + component * 16],
            x[row * k + index + component * 16 + 1],
            x[row * k + index + component * 16 + 2],
            x[row * k + index + component * 16 + 3],
        )
        right = T.call_extern(
            "float32x4",
            "vec4<f32>",
            T.cast(packed >> (component * 4) & 15, "float32") - 8,
            T.cast(packed >> (8 + component * 4) & 15, "float32") - 8,
            T.cast(packed >> (16 + component * 4) & 15, "float32") - 8,
            T.cast(packed >> (24 + component * 4) & 15, "float32") - 8,
        )
        acc += scale * T.call_extern("float32", "dot", left, right)

    @T.macro
    def packed_product(acc: T.Ref, x, w, row, head, tile, lane):
        block = head * (k // 32) + tile * (lanes // 4) + lane // 4
        if planar:
            scale = T.call_extern(
                "float32",
                "tensor_unpack_f16",
                w[(h * k // 32) * 4 + block // 2] >> (block % 2 * 16) & T.uint32(65535),
            )
            packed = w[block * 4 + lane % 4]
        else:
            scale = half_value(w, block * 18)
            packed = word_value(w, block * 18 + 2 + lane % 4 * 4)
        index = tile * (lanes * 8) + lane // 4 * 32 + lane % 4 * 4
        vector_product(acc, x, row, index, scale, packed, 0)
        vector_product(acc, x, row, index, scale, packed, 1)

    @kernel
    def compute(
        parts: Tensor((r * h * splits * (d + 2),), "float32"),
        x: Tensor((r * k,), "float32"),
        w: Tensor((count,), "uint32"),
        enabled: Tensor((r,), "int32"),
    ) -> Tensor((r * h * d,), "float32"):
        out = output((r * h * d,), "float32", position=3)
        with T.Kernel(r * h, threads=d) as index:
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((d,), "float32")
            weights = T.alloc_shared((splits,), "float32")
            gate, coefficient = T.alloc_var("float32"), T.alloc_var("float32")
            gate = 0
            if tx < lanes:
                for tile in T.serial(k // (lanes * 8)):
                    packed_product(gate, x, w, index // h, index % h, tile, tx)
            sum_reduce(scratch, tx, gate)
            coefficient = 1 / (1 + T.exp(-scratch[0]))
            maximum, norm, result = (
                T.alloc_var("float32"),
                T.alloc_var("float32"),
                T.alloc_var("float32"),
            )
            maximum, norm, result = -1e30, 0, 0
            subgroup = T.cast(T.call_extern("uint32", "tensor_subgroup_size"), "int32")
            if T.min(subgroup, d) >= splits:
                part = tx % splits
                base = (index * splits + part) * (d + 2)
                maximum = T.call_extern("float32", "subgroupMax", parts[base + d])
                factor = T.exp(parts[base + d] - maximum)
                norm = T.call_extern(
                    "float32", "subgroupAdd", parts[base + d + 1] * factor
                )
                norm *= T.cast(splits, "float32") / T.cast(
                    T.min(subgroup, d), "float32"
                )
                if tx < splits:
                    weights[tx] = factor
                T.sync_threads()
            else:
                T.sync_threads()
                for tile in T.serial(T.ceildiv(splits, d)):
                    split = tile * d + tx
                    if split < splits:
                        maximum = T.max(
                            maximum, parts[(index * splits + split) * (d + 2) + d]
                        )
                max_reduce(scratch, tx, maximum)
                maximum = scratch[0]
                for tile in T.serial(T.ceildiv(splits, d)):
                    split = tile * d + tx
                    if split < splits:
                        offset = (index * splits + split) * (d + 2)
                        weight = T.exp(parts[offset + d] - maximum)
                        weights[split] = weight
                        norm += parts[offset + d + 1] * weight
                T.sync_threads()
                sum_reduce(scratch, tx, norm)
                norm = scratch[0]
            for part in T.serial(splits):
                result += parts[(index * splits + part) * (d + 2) + tx] * weights[part]
            out[index * d + tx] = T.if_then_else(
                enabled[index // h] != 0, result / T.max(norm, 1e-30) * coefficient, 0
            )
        return out

    return compute
