"""WebGPU specializations and shared bindings for attention."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


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
