"""WebGPU specializations and shared bindings for attention."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


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
        return vector(
            [
                T.cast(keys[channel * dot_width + j, lane], "float32")
                for j in range(dot_width)
            ]
        )

    def probability_vector(scores, i, warp, key):
        return vector([scores[i, warp, key * dot_width + j] for j in range(dot_width)])

    def value_vector(values, key, lane):
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
            keys = T.alloc_shared((d, bk + pad), dtype)
            values = T.alloc_shared((bk, d), dtype)
            scores = T.alloc_shared((bm, group, 32), "float32")
            maxima, norms, factors = (
                registers((bm,)),
                registers((bm,)),
                registers((bm,)),
            )
            accum = registers((bm, 2))
            initialize_maxima(maxima)
            for i, channel in T.Parallel(bm * group, d):
                queries[i // group, i % group, channel] = q[
                    (base + i // group), (kv_head * group + i % group) * d + channel
                ]
            T.sync_threads()
            for tile in T.serial(T.truncdiv(length + bk - 1, bk)):
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
    cache_write,
    chunk_active,
    merge_floating,
    partial_floating,
    prepare_flat,
    rope,
)
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))
