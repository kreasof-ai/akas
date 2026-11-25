"""Shared TileLang computations for residual."""

from collections.abc import Mapping
from akas.kernels.spec import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def add(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    c = p.get("c")
    r = p.get("r", 1)
    n = r * c
    input_dtype = p.get("input_dtype", "float32")
    other_dtype = p.get("other_dtype", input_dtype)
    output_dtype = p.get("output_dtype", input_dtype)
    if any(
        dtype not in ("float32", "float16", "bfloat16")
        for dtype in (input_dtype, other_dtype, output_dtype)
    ):
        raise ValueError("residual addition requires floating storage")

    @kernel
    def compute(x: Tensor((n,), input_dtype), y: Tensor((n,), other_dtype)) -> Tensor(
        (n,), output_dtype
    ):
        out = output((n,), output_dtype)
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            for lane in T.Parallel(256):
                i = block * 256 + lane
                if i < n:
                    out[i] = T.cast(x[i], "float32") + T.cast(y[i], "float32")
        return out

    return compute


def attention_residual(spec, schedule):
    """Append one source and attend over the valid prefix of a residual bank."""
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    p = {**spec.parameters, **schedule}
    rows, width, capacity, index, epsilon = (
        p[key] for key in ("r", "c", "capacity", "index", "eps")
    )
    threads = 128
    reduce = tree_reduce(threads)

    @kernel
    def compute(
        bank: Tensor((capacity, rows, width), "float32"),
        source: Tensor((rows, width), "float32"),
        query: Tensor((width,), "float32"),
    ) -> tuple[
        Tensor((capacity, rows, width), "float32"), Tensor((rows, width), "float32")
    ]:
        out = output((rows, width), "float32")
        with T.Kernel(rows, threads=threads) as row:
            lane = T.get_thread_binding()
            scores = T.alloc_shared((index + 1,), "float32")
            norm = T.alloc_shared((threads,), "float32")
            score = T.alloc_shared((threads,), "float32")
            peak = T.alloc_shared((1,), "float32")
            denominator = T.alloc_shared((1,), "float32")
            for stripe in T.serial(T.ceildiv(width, threads)):
                column = stripe * threads + lane
                if column < width:
                    bank[index, row, column] = source[row, column]
            T.sync_threads()
            for i in T.serial(index + 1):
                sum_sq = T.alloc_var("float32")
                dot = T.alloc_var("float32")
                sum_sq = 0
                dot = 0
                for stripe in T.serial(T.ceildiv(width, threads)):
                    column = stripe * threads + lane
                    if column < width:
                        entry = bank[i, row, column]
                        sum_sq += entry * entry
                        dot += entry * query[column]
                norm[lane] = sum_sq
                score[lane] = dot
                T.sync_threads()
                reduce(norm, lane)
                reduce(score, lane)
                if lane == 0:
                    scores[i] = score[0] * T.rsqrt(norm[0] / width + epsilon)
                T.sync_threads()
            if lane == 0:
                peak[0] = scores[0]
                for i in T.serial(1, index + 1):
                    peak[0] = T.max(peak[0], scores[i])
                denominator[0] = 0
                for i in T.serial(index + 1):
                    scores[i] = T.exp(scores[i] - peak[0])
                    denominator[0] += scores[i]
            T.sync_threads()
            for stripe in T.serial(T.ceildiv(width, threads)):
                column = stripe * threads + lane
                if column < width:
                    total = T.alloc_var("float32")
                    total = 0
                    for i in T.serial(index + 1):
                        total += bank[i, row, column] * (scores[i] / denominator[0])
                    out[row, column] = total
        return bank, out

    return compute


def scaled_residual(spec, schedule):
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows, width = p["r"], p["c"]

    @kernel
    def compute(
        residual: Tensor((rows, width), "float32"),
        mixed: Tensor((rows, width), "float32"),
        scale: Tensor((width,), "float32"),
    ) -> Tensor((rows, width), "float32"):
        out = output((rows, width), "float32")
        with T.Kernel(rows, T.ceildiv(width, 128), threads=128) as (row, tile):
            for lane in T.Parallel(128):
                column = tile * 128 + lane
                if column < width:
                    out[row, column] = (
                        residual[row, column] + mixed[row, column] * scale[column]
                    )
        return out

    return compute
