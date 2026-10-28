"""Shared TileLang computations for attention."""


def cache_pad(spec, schedule):
    """Zero only the unused native F16 tail read by four async key tiles."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel

    p = {**spec.parameters, **schedule}
    slots, heads, capacity, depth = (p[n] for n in ("slots", "heads", "capacity", "d"))
    shape = (slots, heads, capacity, depth)

    @kernel
    def compute(
        kc: Tensor(shape, "float16"),
        vc: Tensor(shape, "float16"),
        lengths: Tensor((slots,), "int32"),
    ) -> tuple[Tensor(shape, "float16"), Tensor(shape, "float16")]:
        with T.Kernel(heads, slots, threads=128) as (head, slot):
            live = T.min(T.max(lengths[slot], 0), capacity)
            for index in T.Parallel(256 * depth):
                position = live + index // depth
                if position < T.ceildiv(live, 256) * 256 and position < capacity:
                    kc[slot, head, position, index % depth] = 0
                    vc[slot, head, position, index % depth] = 0
        return kc, vc

    return compute


def cache_extent(spec, schedule):
    """Return each request's live cache prefix from its enabled query rows."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.kernels.reductions import tree_reduce

    p = {**spec.parameters, **schedule}
    slots, chunk, capacity = (p[name] for name in ("slots", "chunk", "capacity"))
    rows = slots * chunk
    maximum = tree_reduce(128, "max")

    @kernel
    def compute(
        positions: Tensor((rows,), "int32"),
        enabled: Tensor((rows,), "int32"),
    ) -> Tensor((slots,), "int32"):
        lengths = output((slots,), "int32")
        with T.Kernel(slots, threads=128) as slot:
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((128,), "int32")
            value = T.alloc_var("int32")
            value = 0
            for tile in T.serial(T.ceildiv(chunk, 128)):
                index = tile * 128 + tx
                row = slot * chunk + T.min(index, chunk - 1)
                if index < chunk and enabled[row] != 0:
                    value = T.max(value, positions[row] + 1)
            scratch[tx] = value
            T.sync_threads()
            maximum(scratch, tx)
            if tx == 0:
                lengths[slot] = T.min(T.max(scratch[0], 0), capacity)
        return lengths

    return compute


from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def cache_reader(p):
    """Bind the cache layout and codec without coupling them to a backend kernel."""
    import tilelang.language as T
    from akas.ops.attention.definition import cache_storage

    storage = cache_storage(p)
    if p["kv_dtype"] not in ("float16", "float32"):
        from akas.quantization.kv import get

        from akas.quantization._shared.access import cache_arguments

        shape = (p["r"] // p.get("chunk", 1), p["kh"], p["capacity"], p["d"])
        codec = get(p["kv_dtype"]).bind(shape, provider=p["provider"])
        read = cache_arguments(codec.read)
    elif p.get("unit_scales", False):

        @T.macro
        def read(cache, metadata, slot, head, position, channel):
            return (
                T.cast(cache[slot, head, position, channel], "float32")
                * metadata[slot, head, 0, 0]
            )
    else:

        @T.macro
        def read(cache, metadata, slot, head, position, channel):
            return (
                T.cast(cache[slot, head, position, channel], "float32")
                * metadata[slot, head, position, 0]
            )

    if len(storage.metadata) != 1:
        raise ValueError("floating attention requires one metadata buffer per cache")
    return (storage, read)


def cache_arguments(p, storage):
    r, h, d = (p["r"], p["h"], p["d"])
    return [
        ("q", (r, h * d), "float32"),
        ("kc", storage.values.shape, storage.values.dtype),
        ("vc", storage.values.shape, storage.values.dtype),
        ("ks", storage.metadata[0].shape, "float32"),
        ("vs", storage.metadata[0].shape, "float32"),
    ]


def merge_floating(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.attention.kernel import cache_arguments, cache_reader

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    if d != 64:
        raise ValueError("floating attention requires 64-channel heads")
    splits = p.get("splits", 16)
    storage, read = cache_reader(p)
    controls = [("positions", r, "int32"), ("enabled", r, "int32")]

    @kernel
    def compute(
        parts: Tensor(r * h * splits * 66, "float32"), enabled: Tensor(r, "int32")
    ) -> Tensor(r * h * d, "float32"):
        out = output(r * h * d, "float32", position=1)
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
        return out

    return compute


def tiled_partial_floating(spec: KernelSpec, schedule: Mapping[str, object]):
    """Reuse a KV tile across grouped heads and normalize once per key tile."""
    import tilelang.language as T
    from akas.ops.attention.kernel import cache_reader

    p = {**spec.parameters, **schedule}
    rows, heads, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    group, splits = heads // kh, p.get("splits", 16)
    if d != 64 or heads % kh or group not in (1, 2, 4, 8):
        raise ValueError("unsupported grouped floating decode geometry")
    storage, read = cache_reader(p)
    webgpu = p["provider"] == "webgpu"
    shared_dtype = p.get("shared_dtype", "float32")
    dot_width = p.get("dot_width", 1)
    if shared_dtype not in ("float16", "float32") or dot_width not in (1, 4):
        raise ValueError("unsupported grouped decode staging")
    if shared_dtype == "float16" and p["kv_dtype"] != "float16":
        raise ValueError("half staging requires native half KV")

    def dot_terms(queries, keys, head, channel, lane):
        from tensor.compiler.webgpu.schedules.registers import vector

        left = vector([queries[head, channel * 4 + j] for j in range(4)])
        right = vector([T.cast(keys[channel * 4 + j, lane], "float32") for j in range(4)])
        return T.call_extern("float32", "dot", left, right)

    @T.macro
    def exchange(value, offset):
        if webgpu:
            return T.call_extern(
                "float32", "subgroupShuffleXor", value, T.uint32(offset)
            )
        else:
            return T.shfl_xor(value, offset)

    @kernel
    def compute(
        q: Tensor((rows, heads * d), "float32"),
        kc: Tensor(storage.values.shape, storage.values.dtype),
        vc: Tensor(storage.values.shape, storage.values.dtype),
        ks: Tensor(storage.metadata[0].shape, "float32"),
        vs: Tensor(storage.metadata[0].shape, "float32"),
        positions: Tensor(rows, "int32"),
        enabled: Tensor(rows, "int32"),
    ) -> Tensor(rows * heads * splits * 66, "float32"):
        out = output(rows * heads * splits * 66, "float32", position=5)
        with T.Kernel(kh, rows, splits, threads=group * 32) as (kv_head, slot, split):
            tx = T.get_thread_binding()
            head, lane = tx // 32, tx % 32
            length = T.if_then_else(
                enabled[slot] != 0, T.min(positions[slot] + 1, cap), 0
            )
            span = T.truncdiv(length + splits - 1, splits)
            begin, end = split * span, T.min((split + 1) * span, length)
            queries = T.alloc_shared((group, d), "float32")
            keys = T.alloc_shared((d, 33), shared_dtype)
            values = T.alloc_shared((32, d), shared_dtype)
            probabilities = T.alloc_shared((group, 32), "float32")
            maximum, normalizer = T.alloc_var("float32"), T.alloc_var("float32")
            acc0, acc1 = T.alloc_var("float32"), T.alloc_var("float32")
            maximum, normalizer, acc0, acc1 = -T.infinity("float32"), 0, 0, 0
            queries[head, lane] = q[slot, (kv_head * group + head) * d + lane]
            queries[head, lane + 32] = q[slot, (kv_head * group + head) * d + lane + 32]
            T.sync_threads()
            for tile in T.serial(T.truncdiv(T.max(end - begin, 0) + 31, 32)):
                base = begin + tile * 32
                for key, channel in T.Parallel(32, d):
                    keys[channel, key] = T.if_then_else(
                        base + key < end,
                        read(kc, ks, slot, kv_head, base + key, channel),
                        0,
                    )
                    values[key, channel] = T.if_then_else(
                        base + key < end,
                        read(vc, vs, slot, kv_head, base + key, channel),
                        0,
                    )
                T.sync_threads()
                score = T.alloc_var("float32")
                score = 0
                for channel in T.serial(d // 4):
                    if dot_width == 4:
                        score += dot_terms(queries, keys, head, channel, lane)
                    else:
                        for j in T.unroll(4):
                            score += (
                                queries[head, channel * 4 + j]
                                * T.cast(keys[channel * 4 + j, lane], "float32")
                            )
                score = T.if_then_else(
                    base + lane < end, score * 0.125, -T.infinity("float32")
                )
                peak = T.alloc_var("float32")
                peak = score
                for step in T.unroll(5):
                    peak = T.max(peak, exchange(peak, 16 >> step))
                next_max = T.max(maximum, peak)
                correction = T.exp(maximum - next_max)
                probability = T.if_then_else(
                    base + lane < end, T.exp(score - next_max), 0
                )
                norm = T.alloc_var("float32")
                norm = probability
                for step in T.unroll(5):
                    norm += exchange(norm, 16 >> step)
                normalizer = normalizer * correction + norm
                maximum = next_max
                probabilities[head, lane] = probability
                acc0, acc1 = acc0 * correction, acc1 * correction
                T.sync_threads()
                for key in T.serial(32):
                    acc0 += probabilities[head, key] * T.cast(values[key, lane], "float32")
                    acc1 += probabilities[head, key] * T.cast(values[key, lane + 32], "float32")
                T.sync_threads()
            offset = ((slot * heads + kv_head * group + head) * splits + split) * 66
            out[offset + lane] = acc0
            out[offset + lane + 32] = acc1
            if lane == 0:
                out[offset + 64] = maximum
                out[offset + 65] = normalizer
        return out

    return compute.with_attr("tirx.noalias", True)


def partial_floating(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.attention.kernel import cache_arguments, cache_reader

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "capacity"))
    if d != 64:
        raise ValueError("floating attention requires 64-channel heads")
    splits = p.get("splits", 16)
    storage, read = cache_reader(p)
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
                T.cast(tx % size // 32 * 32, "uint32"),
            )
    else:

        @T.macro
        def shuffle(value, offset):
            return T.shfl_down(value, offset)

        @T.macro
        def broadcast(value):
            return T.shfl_sync(value, 0)

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
            head, slot = (index % h, index // h)
            tx = T.get_thread_binding()
            lane, warp = (tx % 32, tx // 32)
            length = T.if_then_else(
                enabled[slot] != 0, T.min(positions[slot] + 1, cap), 0
            )
            span = T.ceildiv(length, splits)
            begin, end = (split * span, T.min((split + 1) * span, length))
            maxima = T.alloc_shared((4,), "float32")
            sums = T.alloc_shared((4,), "float32")
            values = T.alloc_shared((4, 64), "float32")
            maximum = T.alloc_var("float32")
            normalizer = T.alloc_var("float32")
            a0 = T.alloc_var("float32")
            a1 = T.alloc_var("float32")
            maximum, normalizer, a0, a1 = (-T.infinity("float32"), 0, 0, 0)
            q0, q1 = (q[slot, head * d + lane], q[slot, head * d + lane + 32])
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
            values[warp, lane], values[warp, lane + 32] = (a0, a1)
            if lane == 0:
                maxima[warp], sums[warp] = (maximum, normalizer)
            T.sync_threads()
            if warp == 0:
                merged_max = T.max(
                    T.max(maxima[0], maxima[1]), T.max(maxima[2], maxima[3])
                )
                total = T.alloc_var("float32")
                result0 = T.alloc_var("float32")
                result1 = T.alloc_var("float32")
                total, result0, result1 = (0, 0, 0)
                for other in T.unroll(4):
                    factor = T.if_then_else(
                        end > begin, T.exp(maxima[other] - merged_max), 0
                    )
                    total = total + sums[other] * factor
                    result0 = result0 + values[other, lane] * factor
                    result1 = result1 + values[other, lane + 32] * factor
                base = (index * splits + split) * 66
                out[base + lane], out[base + lane + 32] = (result0, result1)
                if lane == 0:
                    out[base + 64], out[base + 65] = (merged_max, total)
        return out

    return compute


def online_attention(p, read):
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    r, h, kh, d, cap = (p[k] for k in ("r", "h", "kh", "d", "capacity"))
    chunk = p.get("chunk", 1)
    if p.get("optimized") and p.get("provider") == "webgpu" and d == 64:

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
                T.cast(tx % size // 32 * 32, "uint32"),
            )

        @T.macro
        def attend_fast(q, kc, vc, ks, vs, out, positions, enabled):
            with T.Kernel(h, r, threads=128) as (head, row):
                tx = T.get_thread_binding()
                lane, warp = (tx % 32, tx // 32)
                length = T.if_then_else(
                    enabled[row] != 0, T.min(positions[row] + 1, cap), 0
                )
                maxima = T.alloc_shared((4,), "float32")
                sums = T.alloc_shared((4,), "float32")
                values = T.alloc_shared((4, 64), "float32")
                maximum = T.alloc_var("float32")
                normalizer = T.alloc_var("float32")
                a0 = T.alloc_var("float32")
                a1 = T.alloc_var("float32")
                maximum, normalizer, a0, a1 = (-T.infinity("float32"), 0, 0, 0)
                q0, q1 = (
                    q[row, head * d + lane],
                    q[row, head * d + lane + 32],
                )
                for offset in T.serial(T.ceildiv(T.max(length - warp, 0), 4)):
                    token = warp + offset * 4
                    score = T.alloc_var("float32")
                    score = q0 * read(
                        kc, ks, row // chunk, head // (h // kh), token, lane
                    ) + q1 * read(
                        kc, ks, row // chunk, head // (h // kh), token, lane + 32
                    )
                    for step in T.unroll(5):
                        score = score + shuffle(score, 16 >> step)
                    score = broadcast(score) * 0.125
                    next_max = T.max(maximum, score)
                    correction, probability = (
                        T.exp(maximum - next_max),
                        T.exp(score - next_max),
                    )
                    a0 = a0 * correction + probability * read(
                        vc, vs, row // chunk, head // (h // kh), token, lane
                    )
                    a1 = a1 * correction + probability * read(
                        vc, vs, row // chunk, head // (h // kh), token, lane + 32
                    )
                    normalizer = normalizer * correction + probability
                    maximum = next_max
                values[warp, lane], values[warp, lane + 32] = (a0, a1)
                if lane == 0:
                    maxima[warp], sums[warp] = (maximum, normalizer)
                T.sync_threads()
                if warp == 0:
                    merged_max = T.max(
                        T.max(maxima[0], maxima[1]), T.max(maxima[2], maxima[3])
                    )
                    total = T.alloc_var("float32")
                    result0 = T.alloc_var("float32")
                    result1 = T.alloc_var("float32")
                    total, result0, result1 = (0, 0, 0)
                    for other in T.unroll(4):
                        factor = T.if_then_else(
                            length > 0, T.exp(maxima[other] - merged_max), 0
                        )
                        total = total + sums[other] * factor
                        result0 = result0 + values[other, lane] * factor
                        result1 = result1 + values[other, lane + 32] * factor
                    base = row * h * d + head * d
                    out[base + lane] = T.if_then_else(
                        enabled[row] != 0, result0 / total, 0
                    )
                    out[base + lane + 32] = T.if_then_else(
                        enabled[row] != 0, result1 / total, 0
                    )

        return attend_fast

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


def attention(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.attention.definition import cache_storage

    p = {**spec.parameters, **schedule}
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    r, d = (p["r"], p["d"])
    if d != 64:
        raise ValueError("portable reference attention requires 64-channel heads")
    h, kh, cap = (p["h"], p["kh"], p["capacity"])
    chunk = p.get("chunk", 1)
    storage = cache_storage(p)
    shape, dtype = (storage.values.shape, storage.values.dtype)
    if len(storage.metadata) != 1:
        raise ValueError("reference attention requires one metadata buffer per cache")
    scale_shape = storage.metadata[0].shape
    quantized = p["kv_dtype"] not in ("float16", "float32")
    if quantized:
        from akas.quantization.kv import get

        from akas.quantization._shared.access import cache_arguments, writer_arguments

        codec = get(p["kv_dtype"]).bind(
            (r // chunk, kh, cap, d), provider=p["provider"], access={"chunk": chunk}
        )
        encode = writer_arguments(codec.require("write"))
        read = cache_arguments(codec.read)
    else:
        from akas.quantization._shared.rounding import round_half

        rounded = round_half()

        @T.macro
        def read(cache, metadata, slot, head, position, channel):
            return (
                T.cast(cache[slot, head, position, channel], "float32")
                * metadata[slot, head, position, 0]
            )

    total = tree_reduce(64)

    @T.macro
    def attend(q, kc, vc, ks, vs, out, positions, enabled):
        with T.Kernel(h, r, threads=64) as (head, row):
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((64,), "float32")
            maximum = T.alloc_shared((1,), "float32")
            normalizer = T.alloc_shared((1,), "float32")
            factor = T.alloc_shared((1,), "float32")
            probability = T.alloc_shared((1,), "float32")
            accum = T.alloc_var("float32")
            accum = 0
            if tx == 0:
                maximum[0] = -T.infinity("float32")
                normalizer[0] = 0
            T.sync_threads()
            for position in T.serial(
                cap if p["provider"] == "webgpu" else T.min(positions[row] + 1, cap)
            ):
                scratch[tx] = q[row, head * d + tx] * read(
                    kc, ks, row // chunk, head // (h // kh), position, tx
                )
                T.sync_threads()
                total(scratch, tx)
                if tx == 0:
                    score = T.if_then_else(
                        position <= positions[row],
                        scratch[0] * d ** (-0.5),
                        T.float32(-1e30),
                    )
                    next_max = T.max(maximum[0], score)
                    factor[0] = T.exp(maximum[0] - next_max)
                    probability[0] = T.if_then_else(
                        position <= positions[row], T.exp(score - next_max), 0
                    )
                    normalizer[0] = normalizer[0] * factor[0] + probability[0]
                    maximum[0] = next_max
                T.sync_threads()
                accum = accum * factor[0] + probability[0] * read(
                    vc, vs, row // chunk, head // (h // kh), position, tx
                )
                T.sync_threads()
            if enabled[row] != 0:
                out[row * h * d + head * d + tx] = accum / normalizer[0]
            else:
                out[row * h * d + head * d + tx] = 0

    if p["provider"] == "webgpu":
        from akas.ops.attention.kernel import online_attention as attention

        attend = attention(p, read)

    @kernel
    def compute(
        q: Tensor((r, h * d), "float32"),
        kc: Tensor(shape, dtype),
        vc: Tensor(shape, dtype),
        ks: Tensor(scale_shape, "float32"),
        vs: Tensor(scale_shape, "float32"),
        positions: Tensor(r, "int32"),
        enabled: Tensor(r, "int32"),
    ) -> Tensor(r * h * d, "float32"):
        out = output(r * h * d, "float32", position=5)
        attend(q, kc, vc, ks, vs, out, positions, enabled)
        return out

    compiled = compute
    return compiled.with_attr("tirx.noalias", True) if p.get("optimized") else compiled


def cache_write(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.ops.attention.definition import cache_storage

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, d = (p["r"], p["d"])
    if d != 64:
        raise ValueError("portable reference attention requires 64-channel heads")
    h, kh, cap = (p["h"], p["kh"], p["capacity"])
    chunk = p.get("chunk", 1)
    storage = cache_storage(p)
    shape, dtype = (storage.values.shape, storage.values.dtype)
    if len(storage.metadata) != 1:
        raise ValueError("reference attention requires one metadata buffer per cache")
    scale_shape = storage.metadata[0].shape
    quantized = p["kv_dtype"] not in ("float16", "float32")
    if quantized:
        from akas.quantization.kv import get

        from akas.quantization._shared.access import cache_arguments, writer_arguments

        codec = get(p["kv_dtype"]).bind(
            (r // chunk, kh, cap, d), provider=p["provider"], access={"chunk": chunk}
        )
        encode = writer_arguments(codec.require("write"))
        read = cache_arguments(codec.read)
    else:
        from akas.quantization._shared.rounding import round_half

        rounded = round_half()

    @kernel
    def compute(
        keys: Tensor((r, kh * d), "float32"),
        values: Tensor((r, kh * d), "float32"),
        kc: Tensor(shape, dtype),
        vc: Tensor(shape, dtype),
        ks: Tensor(scale_shape, "float32"),
        vs: Tensor(scale_shape, "float32"),
        positions: Tensor(r, "int32"),
        enabled: Tensor(r, "int32"),
    ) -> tuple[
        Tensor(shape, dtype),
        Tensor(shape, dtype),
        Tensor(scale_shape, "float32"),
        Tensor(scale_shape, "float32"),
    ]:
        with T.Kernel(kh, r, threads=128) as (head, row):
            tx = T.get_thread_binding()
            if quantized:
                encode(
                    keys, values, row, head, positions[row], kc, vc, ks, vs, tx, enabled
                )
            elif enabled[row] != 0:
                if tx == 0:
                    ks[row // chunk, head, positions[row], 0] = 1
                    vs[row // chunk, head, positions[row], 0] = 1
                if tx < d:
                    if p["kv_dtype"] == "float16" and p["provider"] == "webgpu":
                        kc[row // chunk, head, positions[row], tx] = rounded(
                            keys[row, head * d + tx]
                        )
                        vc[row // chunk, head, positions[row], tx] = rounded(
                            values[row, head * d + tx]
                        )
                    else:
                        kc[row // chunk, head, positions[row], tx] = keys[
                            row, head * d + tx
                        ]
                        vc[row // chunk, head, positions[row], tx] = values[
                            row, head * d + tx
                        ]
        return (kc, vc, ks, vs)

    return compute


def rope(spec: KernelSpec, schedule: Mapping[str, object]):
    import math

    p = {**spec.parameters, **schedule}
    if p.get("position_limit", 0) > 8192 or p.get("provider") == "webgpu":
        return precise_rope(spec, schedule)
    import tilelang.language as T

    r, d = (p["r"], p["d"])
    if d != 64:
        raise ValueError("portable reference attention requires 64-channel heads")
    h, theta = (p["h"], p["theta"])

    @kernel
    def compute(
        x: Tensor((r * h * d,), "float32"), positions: Tensor((r,), "int32")
    ) -> Tensor((r, h * d), "float32"):
        out = output((r, h * d), "float32", position=1)
        with T.Kernel(h, r, threads=64) as (head, row):
            tx = T.get_thread_binding()
            half = tx % (d // 2)
            angle = T.cast(positions[row], "float32") * T.exp(
                -(math.log(theta) * 2 / d) * half
            )
            left = x[(row * h + head) * d + half]
            right = x[(row * h + head) * d + half + d // 2]
            out[row, head * d + tx] = T.if_then_else(
                tx < d // 2,
                left * T.cos(angle) - right * T.sin(angle),
                left * T.sin(angle) + right * T.cos(angle),
            )
        return out

    return compute


def precise_rope(spec: KernelSpec, schedule: Mapping[str, object]):
    import math
    import struct

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, d = (p["r"], p["d"])
    if d != 64:
        raise ValueError("portable reference attention requires 64-channel heads")
    h, theta = (p["h"], p["theta"])
    precise = p.get("position_limit", 0) > 8192 or p.get("provider") == "webgpu"
    fma = "fma" if p.get("provider") == "webgpu" else "fmaf"

    def single(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    frequencies = tuple(theta ** (-2 * i / d) for i in range(d // 2))
    high = tuple(single(value) for value in frequencies)
    low = tuple(single(value - rounded) for value, rounded in zip(frequencies, high))
    tau_high = single(2 * math.pi)
    tau_low = single(2 * math.pi - tau_high)
    digits = max(1, ((max(1, p.get("position_limit", 131072)) - 1).bit_length() + 3) // 4)
    phases = tuple(
        single(math.remainder(frequency * (16 ** digit), 2 * math.pi))
        for digit in range(digits) for frequency in frequencies
    )

    def exact(value):
        if p.get("provider") == "webgpu":
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            # Keep bit patterns opaque to TIR constant folding and WGSL printers.
            return T.call_extern("float32", "bitcast<f32>", T.uint32(bits))
        return T.float32(value)

    def constant(index, values):
        def select(begin, end):
            if end - begin == 1:
                return exact(values[begin])
            middle = (begin + end) // 2
            return T.if_then_else(index < middle, select(begin, middle), select(middle, end))

        return select(0, len(values))

    @T.macro
    def product_error(integer, coefficient, rounded):
        high_integer = integer >> 6 << 6
        a_high = T.cast(high_integer, "float32")
        a_low = T.cast(integer - high_integer, "float32")
        b_high = T.reinterpret(
            "float32", T.reinterpret("uint32", coefficient) & T.uint32(0xFFFFF000)
        )
        b_low = coefficient - b_high
        return (
            (a_high * b_high - rounded) + a_high * b_low + a_low * b_high
        ) + a_low * b_low

    @kernel
    def compute(
        x: Tensor((r * h * d,), "float32"), positions: Tensor((r,), "int32")
    ) -> Tensor((r, h * d), "float32"):
        out = output((r, h * d), "float32", position=1)
        with T.Kernel(h, r, threads=64) as (head, row):
            tx = T.get_thread_binding()
            half = tx % (d // 2)
            angle = T.alloc_var("float32")
            if precise:
                position = T.cast(positions[row], "float32")
                frequency = T.alloc_var("float32")
                leading = T.alloc_var("float32")
                trailing = T.alloc_var("float32")
                turns = T.alloc_var("int32")
                if p.get("provider") == "webgpu":
                    # Form bounded angles from base-16 position digits. Large
                    # float products lose their low bits, and Vulkan drivers may
                    # contract the compensated product's cancellation sequence.
                    angle = 0
                    for digit in T.unroll(digits):
                        coefficient = constant(half + digit * (d // 2), phases)
                        angle = angle + T.cast((positions[row] >> (digit * 4)) & 15,
                                               "float32") * coefficient
                        angle = angle - T.floor(angle / (2 * math.pi) + 0.5) * exact(tau_high)
                else:
                    frequency = constant(half, high)
                    leading = position * frequency
                    trailing = T.call_extern(
                        "float32", fma, position, frequency, -leading
                    ) + position * constant(half, low)
                    turns = T.cast(T.floor(leading / (2 * math.pi) + 0.5), "int32")
                    angle = T.call_extern(
                        "float32",
                        fma,
                        -T.cast(turns, "float32"),
                        T.float32(tau_high),
                        leading,
                    ) + (trailing - T.cast(turns, "float32") * tau_low)
            else:
                angle = T.cast(positions[row], "float32") * T.exp(
                    -(math.log(theta) * 2 / d) * half
                )
            left = x[(row * h + head) * d + half]
            right = x[(row * h + head) * d + half + d // 2]
            cosine = T.alloc_var("float32")
            sine = T.alloc_var("float32")
            if p.get("provider") == "webgpu":
                # Vulkan sin/cos accuracy varies by driver. After compensated
                # reduction to [-pi, pi], reflect into [-pi/2, pi/2] and use
                # bounded polynomials rather than the approximate intrinsics.
                reduced = T.alloc_var("float32")
                square = T.alloc_var("float32")
                reduced = T.min(T.abs(angle), T.float32(math.pi) - T.abs(angle))
                square = reduced * reduced
                sine = reduced * (1 + square * (-1 / 6 + square * (1 / 120 + square *
                       (-1 / 5040 + square * (1 / 362880 - square / 39916800)))))
                cosine = 1 + square * (-1 / 2 + square * (1 / 24 + square *
                         (-1 / 720 + square * (1 / 40320 + square *
                         (-1 / 3628800 + square / 479001600)))))
                sine = T.if_then_else(angle < 0, -sine, sine)
                cosine = T.if_then_else(T.abs(angle) > math.pi / 2, -cosine, cosine)
            else:
                cosine = T.cos(angle)
                sine = T.sin(angle)
            out[row, head * d + tx] = T.if_then_else(
                tx < d // 2,
                left * cosine - right * sine,
                left * sine + right * cosine,
            )
        return out

    return compute


def attention_flat(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    h, kh, d, cap = (p["h"], p["kh"], p["d"], p["cap"])
    if r == 1:

        @kernel
        def compute(
            q: Tensor((r * h * d,), "float32"),
            kc: Tensor((cap * kh * d,), "float16"),
            vc: Tensor((cap * kh * d,), "float16"),
            pos: Tensor((2,), "int32"),
        ) -> Tensor((r * h * d,), "float32"):
            out = output((r * h * d,), "float32", position=3)
            with T.Kernel(h, threads=128) as head:
                dots = T.alloc_fragment((64, d), "float32")
                scores = T.alloc_fragment((64,), "float32")
                products = T.alloc_fragment((64, d), "float32")
                partial = T.alloc_fragment((d,), "float32")
                result = T.alloc_fragment((d,), "float32")
                maximum = T.alloc_fragment((1,), "float32")
                previous = T.alloc_fragment((1,), "float32")
                normalizer = T.alloc_fragment((1,), "float32")
                total = T.alloc_fragment((1,), "float32")
                T.fill(maximum, -T.infinity("float32"))
                T.clear(result)
                T.clear(normalizer)
                for tile in T.serial(T.ceildiv(pos[0] + 1, 64)):
                    for i, j in T.Parallel(64, d):
                        dots[i, j] = q[head * d + j] * T.cast(
                            kc[((tile * 64 + i) * kh + head // (h // kh)) * d + j],
                            "float32",
                        )
                    T.reduce_sum(dots, scores, dim=1)
                    T.copy(maximum, previous)
                    for i in T.Parallel(64):
                        scores[i] = T.if_then_else(
                            tile * 64 + i <= pos[0],
                            scores[i] * d ** (-0.5),
                            -T.infinity("float32"),
                        )
                    T.reduce_max(scores, maximum, dim=0, clear=False)
                    for i in T.Parallel(64):
                        scores[i] = T.exp(scores[i] - maximum[0])
                    T.reduce_sum(scores, total, dim=0)
                    normalizer[0] = (
                        normalizer[0] * T.exp(previous[0] - maximum[0]) + total[0]
                    )
                    for i, j in T.Parallel(64, d):
                        products[i, j] = scores[i] * T.cast(
                            vc[((tile * 64 + i) * kh + head // (h // kh)) * d + j],
                            "float32",
                        )
                    T.reduce_sum(products, partial, dim=0)
                    for j in T.Parallel(d):
                        result[j] = (
                            result[j] * T.exp(previous[0] - maximum[0]) + partial[j]
                        )
                for j in T.Parallel(d):
                    out[head * d + j] = result[j] / normalizer[0]
            return out

        return compute

    @kernel
    def compute(
        q: Tensor((r * h * d,), "float32"),
        kc: Tensor((cap * kh * d,), "float16"),
        vc: Tensor((cap * kh * d,), "float16"),
        pos: Tensor((2,), "int32"),
    ) -> Tensor((r * h * d,), "float32"):
        out = output((r * h * d,), "float32", position=3)
        with T.Kernel(T.ceildiv(r, 32), h, threads=128) as (bx, head):
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
                query[i, j] = T.if_then_else(
                    bx * 32 + i < r, q[((bx * 32 + i) * h + head) * d + j], 0
                )
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            for tile in T.serial(T.ceildiv(pos[0] + T.min(r, (bx + 1) * 32), 64)):
                for i, j in T.Parallel(64, d):
                    key[i, j] = kc[((tile * 64 + i) * kh + head // (h // kh)) * d + j]
                    value[i, j] = vc[((tile * 64 + i) * kh + head // (h // kh)) * d + j]
                T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                T.copy(maximum, previous)
                for i, j in T.Parallel(32, 64):
                    scores[i, j] = T.if_then_else(
                        tile * 64 + j <= pos[0] + bx * 32 + i,
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
                if bx * 32 + i < r:
                    out[((bx * 32 + i) * h + head) * d + j] = T.if_then_else(
                        bx * 32 + i < pos[1], result[i, j] / normalizer[i], 0
                    )
        return out

    return compute


def prepare_flat(spec: KernelSpec, schedule: Mapping[str, object]):
    import math

    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p.get("r", 1)
    h, kh, d, cap = (p["h"], p["kh"], p["d"], p["cap"])
    eps, theta = (p["eps"], p["theta"])

    @kernel
    def compute(
        q: Tensor((r * h * d,), "float32"),
        k: Tensor((r * kh * d,), "float32"),
        v: Tensor((r * kh * d,), "float32"),
        qw: Tensor((d,), "float32"),
        kw: Tensor((d,), "float32"),
        kc: Tensor((cap * kh * d,), "float16"),
        vc: Tensor((cap * kh * d,), "float16"),
        pos: Tensor((2,), "int32"),
    ) -> tuple[
        Tensor((r * h * d,), "float32"),
        Tensor((cap * kh * d,), "float16"),
        Tensor((cap * kh * d,), "float16"),
    ]:
        qo = output((r * h * d,), "float32", position=5)
        with T.Kernel(h, r, threads=64) as (head, row):
            square = T.alloc_fragment((d,), "float32")
            total = T.alloc_fragment((1,), "float32")
            norm = T.alloc_shared((d,), "float32")
            for i in T.Parallel(d):
                square[i] = q[(row * h + head) * d + i] * q[(row * h + head) * d + i]
            T.reduce_sum(square, total, dim=0)
            for i in T.Parallel(d):
                norm[i] = (
                    q[(row * h + head) * d + i] * T.rsqrt(total[0] / d + eps) * qw[i]
                )
            for i in T.Parallel(d // 2):
                angle = T.cast(pos[0] + row, "float32") * T.exp(
                    -(math.log(theta) * 2 / d) * i
                )
                qo[(row * h + head) * d + i] = norm[i] * T.cos(angle) - norm[
                    i + d // 2
                ] * T.sin(angle)
                qo[(row * h + head) * d + i + d // 2] = norm[i] * T.sin(angle) + norm[
                    i + d // 2
                ] * T.cos(angle)
            if head < kh:
                for i in T.Parallel(d):
                    square[i] = (
                        k[(row * kh + head) * d + i] * k[(row * kh + head) * d + i]
                    )
                T.reduce_sum(square, total, dim=0)
                for i in T.Parallel(d):
                    norm[i] = (
                        k[(row * kh + head) * d + i]
                        * T.rsqrt(total[0] / d + eps)
                        * kw[i]
                    )
                    if row < pos[1]:
                        vc[((pos[0] + row) * kh + head) * d + i] = v[
                            (row * kh + head) * d + i
                        ]
                for i in T.Parallel(d // 2):
                    angle = T.cast(pos[0] + row, "float32") * T.exp(
                        -(math.log(theta) * 2 / d) * i
                    )
                    if row < pos[1]:
                        kc[((pos[0] + row) * kh + head) * d + i] = norm[i] * T.cos(
                            angle
                        ) - norm[i + d // 2] * T.sin(angle)
                        kc[((pos[0] + row) * kh + head) * d + i + d // 2] = norm[
                            i
                        ] * T.sin(angle) + norm[i + d // 2] * T.cos(angle)
        return (qo, kc, vc)

    return compute


def chunk_active(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    slots, chunk = (p["slots"], p["chunk"])

    @kernel
    def compute(lengths: Tensor((slots,), "int32")) -> Tensor(
        (slots * chunk,), "int32"
    ):
        out = output((slots * chunk,), "int32")
        with T.Kernel(T.ceildiv(slots * chunk, 64), threads=64) as block:
            row = block * 64 + T.get_thread_binding()
            if row < slots * chunk:
                out[row] = T.if_then_else(row % chunk < lengths[row // chunk], 1, 0)
        return out

    return compute


def _normalization(p, query):
    import math
    import tilelang.language as T

    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "cap"))
    heads = h if query else kh
    eps, angle_scale = (p["eps"], math.log(p["theta"]) * 2 / d)

    @T.macro
    def normalize(x, w, out, pos, v=None, vc=None):
        with T.Kernel(heads, r, threads=64) as (head, row):
            square = T.alloc_fragment((d,), "float32")
            total = T.alloc_fragment((1,), "float32")
            norm = T.alloc_shared((d,), "float32")
            for i in T.Parallel(d):
                square[i] = (
                    x[(row * heads + head) * d + i] * x[(row * heads + head) * d + i]
                )
            T.reduce_sum(square, total, dim=0)
            for i in T.Parallel(d):
                norm[i] = (
                    x[(row * heads + head) * d + i] * T.rsqrt(total[0] / d + eps) * w[i]
                )
                if not query:
                    if row < pos[1]:
                        vc[((pos[0] + row) * kh + head) * d + i] = v[
                            (row * kh + head) * d + i
                        ]
            for i in T.Parallel(d // 2):
                angle = T.cast(pos[0] + row, "float32") * T.exp(-angle_scale * i)
                if query:
                    out[(row * heads + head) * d + i] = norm[i] * T.cos(angle) - norm[
                        i + d // 2
                    ] * T.sin(angle)
                    out[(row * heads + head) * d + i + d // 2] = norm[i] * T.sin(
                        angle
                    ) + norm[i + d // 2] * T.cos(angle)
                elif row < pos[1]:
                    out[((pos[0] + row) * heads + head) * d + i] = norm[i] * T.cos(
                        angle
                    ) - norm[i + d // 2] * T.sin(angle)
                    out[((pos[0] + row) * heads + head) * d + i + d // 2] = norm[
                        i
                    ] * T.sin(angle) + norm[i + d // 2] * T.cos(angle)

    if query:

        @kernel
        def compute(
            q: Tensor(r * h * d, "float32"),
            qw: Tensor(d, "float32"),
            pos: Tensor(2, "int32"),
        ) -> Tensor(r * h * d, "float32"):
            qo = output(r * h * d, "float32", position=2)
            normalize(q, qw, qo, pos)
            return qo

        return compute

    @kernel
    def compute(
        k: Tensor(r * kh * d, "float32"),
        v: Tensor(r * kh * d, "float32"),
        kw: Tensor(d, "float32"),
        kc: Tensor(cap * kh * d, "float16"),
        vc: Tensor(cap * kh * d, "float16"),
        pos: Tensor(2, "int32"),
    ) -> tuple[Tensor(cap * kh * d, "float16"), Tensor(cap * kh * d, "float16")]:
        normalize(k, kw, kc, pos, v, vc)
        return (kc, vc)

    return compute


def rotary_partial(spec, schedule):
    """Split-half rotary embedding of a leading subspace, leaving the tail intact."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, h, d, rot, theta = (p[k] for k in ("r", "h", "d", "rotary", "theta"))
    import math

    # CUDA double argument reduction preserves the phase at million-token positions.
    @kernel
    def compute(
        x: Tensor((r * h * d,), "float32"), positions: Tensor((r,), "int32")
    ) -> Tensor((r, h * d), "float32"):
        out = output((r, h * d), "float32", position=1)
        with T.Kernel(h, r, threads=d) as (head, row):
            tx = T.get_thread_binding()
            if tx < rot:
                half = tx % (rot // 2)
                frequency = T.exp(
                    T.cast(half, "float64") * (-2 * math.log(theta) / rot)
                )
                angle = T.cast(positions[row], "float64") * frequency
                cosine = T.cast(T.cos(angle), "float32")
                sine = T.cast(T.sin(angle), "float32")
                left = x[(row * h + head) * d + half]
                right = x[(row * h + head) * d + half + rot // 2]
                out[row, head * d + tx] = T.if_then_else(
                    tx < rot // 2,
                    left * cosine - right * sine,
                    left * sine + right * cosine,
                )
            else:
                out[row, head * d + tx] = x[(row * h + head) * d + tx]
        return out

    return compute


def window_cache(spec, schedule):
    """Write a chunk to a circular cache without dropping keys needed by its first query."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, kh, d, cap, chunk = (p[k] for k in ("r", "kh", "d", "capacity", "chunk"))
    shape = (r // chunk, kh, cap, d)

    @kernel
    def compute(
        keys: Tensor((r, kh * d), "float32"),
        values: Tensor((r, kh * d), "float32"),
        kc: Tensor(shape, "float16"),
        vc: Tensor(shape, "float16"),
        positions: Tensor((r,), "int32"),
        enabled: Tensor((r,), "int32"),
    ) -> tuple[Tensor(shape, "float16"), Tensor(shape, "float16")]:
        with T.Kernel(kh, r, threads=d) as (head, row):
            tx = T.get_thread_binding()
            if enabled[row] != 0:
                kc[row // chunk, head, positions[row] % cap, tx] = keys[
                    row, head * d + tx
                ]
                vc[row // chunk, head, positions[row] % cap, tx] = values[
                    row, head * d + tx
                ]
        return kc, vc

    return compute


def qkv_split(spec, schedule):
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, h, kh, d = (p[k] for k in ("r", "h", "kh", "d"))
    width = (h + 2 * kh) * d

    @kernel
    def compute(
        x: Tensor((r * width,), "float32"),
    ) -> tuple[
        Tensor((r * h * d,), "float32"),
        Tensor((r * kh * d,), "float32"),
        Tensor((r, kh * d), "float32"),
    ]:
        q = output((r * h * d,), "float32")
        k = output((r * kh * d,), "float32")
        v = output((r, kh * d), "float32")
        with T.Kernel(r, T.ceildiv(width, 256), threads=256) as (row, tile):
            for lane in T.Parallel(256):
                col = tile * 256 + lane
                if col < h * d:
                    q[row * h * d + col] = x[row * width + col]
                elif col < (h + kh) * d:
                    k[row * kh * d + col - h * d] = x[row * width + col]
                elif col < width:
                    v[row, col - (h + kh) * d] = x[row * width + col]
        return q, k, v

    return compute


def window_merge(spec, schedule):
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, h, d, splits = (p[k] for k in ("r", "h", "d", "splits"))

    @kernel
    def compute(
        parts: Tensor((r * h * splits * (d + 2),), "float32"),
        enabled: Tensor((r,), "int32"),
    ) -> Tensor((r * h * d,), "float32"):
        out = output((r * h * d,), "float32", position=1)
        with T.Kernel(r * h, threads=128) as index:
            maxima = T.alloc_fragment((splits,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            sums = T.alloc_fragment((splits,), "float32")
            normalizer = T.alloc_fragment((1,), "float32")
            products = T.alloc_fragment((splits, d), "float32")
            result = T.alloc_fragment((d,), "float32")
            for i in T.Parallel(splits):
                maxima[i] = parts[(index * splits + i) * (d + 2) + d]
            T.reduce_max(maxima, maximum, dim=0)
            for i in T.Parallel(splits):
                sums[i] = parts[(index * splits + i) * (d + 2) + d + 1] * T.exp(
                    maxima[i] - maximum[0]
                )
            T.reduce_sum(sums, normalizer, dim=0)
            for i, j in T.Parallel(splits, d):
                products[i, j] = parts[(index * splits + i) * (d + 2) + j] * T.exp(
                    maxima[i] - maximum[0]
                )
            T.reduce_sum(products, result, dim=0)
            for j in T.Parallel(d):
                out[index * d + j] = T.if_then_else(
                    enabled[index // h] != 0, result[j] / T.max(normalizer[0], 1e-30), 0
                )
        return out

    return compute


def rotary_table(spec, schedule):
    """Apply a shared high-precision rotary table using F32 reads and products."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, h, d, rot, cap = (p[n] for n in ("r", "h", "d", "rotary", "capacity"))

    @kernel
    def compute(
        x: Tensor((r * h * d,), "float32"),
        positions: Tensor((r,), "int32"),
        table: Tensor((cap, rot), "float32"),
    ) -> Tensor((r, h * d), "float32"):
        out = output((r, h * d), "float32", position=1)
        with T.Kernel(h, r, threads=d) as (head, row):
            tx = T.get_thread_binding()
            position = T.min(T.max(positions[row], 0), cap - 1)
            if tx < rot:
                half = tx % (rot // 2)
                left = x[(row * h + head) * d + half]
                right = x[(row * h + head) * d + half + rot // 2]
                co = table[position, half]
                si = table[position, half + rot // 2]
                out[row, head * d + tx] = T.if_then_else(
                    tx < rot // 2, left * co - right * si, left * si + right * co
                )
            else:
                out[row, head * d + tx] = x[(row * h + head) * d + tx]
        return out

    return compute


def rotary_compact(spec, schedule):
    """Reconstruct absolute phases from fine and coarse F32 rotations."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, h, d, rot, cap, stride, blocks = (
        p[n] for n in ("r", "h", "d", "rotary", "capacity", "stride", "blocks")
    )
    half = rot // 2
    pair = d // 2 if p.get("full_head_pairs", False) else half

    @kernel
    def compute(
        x: Tensor((r * h * d,), "float32"),
        positions: Tensor((r,), "int32"),
        fine: Tensor((stride, rot), "float32"),
        coarse: Tensor((blocks, rot), "float32"),
    ) -> Tensor((r, h * d), "float32"):
        out = output((r, h * d), "float32", position=1)
        with T.Kernel(T.ceildiv(r * h * d, 256), threads=256) as block:
            for thread in T.Parallel(256):
                index = block * 256 + thread
                if index < r * h * d:
                    row, dim = index // (h * d), index % d
                    if dim % pair < half and dim < 2 * pair:
                        pos = T.min(T.max(positions[row], 0), cap - 1)
                        column = dim % pair
                        c0, s0 = (
                            fine[pos % stride, column],
                            fine[pos % stride, column + half],
                        )
                        c1, s1 = (
                            coarse[pos // stride, column],
                            coarse[pos // stride, column + half],
                        )
                        cosine, sine = c0 * c1 - s0 * s1, s0 * c1 + c0 * s1
                        base = index - dim
                        left, right = x[base + column], x[base + column + pair]
                        out[row, index % (h * d)] = T.if_then_else(
                            dim < pair,
                            left * cosine - right * sine,
                            left * sine + right * cosine,
                        )
                    else:
                        out[row, index % (h * d)] = x[index]
        return out

    return compute


def qkv_rotary_cache(spec, schedule):
    """Split projected QKV, apply compact RoPE and write final F16 storage."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    r, h, kh, d, cap, chunk, rot, context, stride, blocks = (
        p[n]
        for n in (
            "r",
            "h",
            "kh",
            "d",
            "capacity",
            "chunk",
            "rotary",
            "context",
            "stride",
            "blocks",
        )
    )
    width = (h + 2 * kh) * d
    shape = (r // chunk, kh, cap, d)

    @kernel
    def compute(
        x: Tensor((r * width,), "float32"),
        kc: Tensor(shape, "float16"),
        vc: Tensor(shape, "float16"),
        positions: Tensor((r,), "int32"),
        enabled: Tensor((r,), "int32"),
        fine: Tensor((stride, rot), "float32"),
        coarse: Tensor((blocks, rot), "float32"),
    ) -> tuple[
        Tensor((r, h * d), "float16"),
        Tensor(shape, "float16"),
        Tensor(shape, "float16"),
    ]:
        q = output((r, h * d), "float16", position=1)
        with T.Kernel(T.ceildiv(r * width, 256), threads=256) as block:
            for lane in T.Parallel(256):
                index = block * 256 + lane
                if index < r * width:
                    row, column = index // width, index % width
                    head, channel = column // d, column % d
                    if enabled[row] != 0:
                        value = T.alloc_var("float32")
                        value = x[index]
                        if (head < h + kh) & (channel < rot):
                            pos = T.min(T.max(positions[row], 0), context - 1)
                            frequency = channel % (rot // 2)
                            c0, s0 = (
                                fine[pos % stride, frequency],
                                fine[pos % stride, frequency + rot // 2],
                            )
                            c1, s1 = (
                                coarse[pos // stride, frequency],
                                coarse[pos // stride, frequency + rot // 2],
                            )
                            cosine, sine = c0 * c1 - s0 * s1, s0 * c1 + c0 * s1
                            pair = row * width + head * d + frequency
                            value = T.if_then_else(
                                channel < rot // 2,
                                x[pair] * cosine - x[pair + rot // 2] * sine,
                                x[pair] * sine + x[pair + rot // 2] * cosine,
                            )
                        if head < h:
                            q[row, column] = value
                        elif head < h + kh:
                            kc[
                                row // chunk,
                                head - h,
                                (
                                    row % chunk
                                    if p.get("current_chunk", False)
                                    else positions[row] % cap
                                ),
                                channel,
                            ] = value
                        else:
                            vc[
                                row // chunk,
                                head - h - kh,
                                (
                                    row % chunk
                                    if p.get("current_chunk", False)
                                    else positions[row] % cap
                                ),
                                channel,
                            ] = value
                    elif head < h:
                        q[row, column] = 0
        return q, kc, vc

    return compute


def window_cache_tail(spec, schedule):
    """Commit only the live chunk suffix after attention consumes the old ring."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel

    p = {**spec.parameters, **schedule}
    r, kh, d, cap, chunk = (p[n] for n in ("r", "kh", "d", "capacity", "chunk"))
    shape = (r // chunk, kh, cap, d)
    current_shape = (r // chunk, kh, chunk, d)

    @kernel
    def compute(
        current_k: Tensor(current_shape, "float16"),
        current_v: Tensor(current_shape, "float16"),
        kc: Tensor(shape, "float16"),
        vc: Tensor(shape, "float16"),
        positions: Tensor((r,), "int32"),
        enabled: Tensor((r,), "int32"),
    ) -> tuple[Tensor(shape, "float16"), Tensor(shape, "float16")]:
        with T.Kernel(kh, r, threads=d) as (head, row):
            channel = T.get_thread_binding()
            latest = T.alloc_var("bool")
            latest = True
            if row % chunk + cap < chunk:
                latest = enabled[row + cap] == 0
            if (enabled[row] != 0) & latest:
                kc[row // chunk, head, positions[row] % cap, channel] = current_k[
                    row // chunk, head, row % chunk, channel
                ]
                vc[row // chunk, head, positions[row] % cap, channel] = current_v[
                    row // chunk, head, row % chunk, channel
                ]
        return kc, vc

    return compute
