"""CUDA specializations and shared bindings for vocabulary."""


def quantized_embedding(spec, schedule):
    from .kernel import quantized_embedding

    return quantized_embedding(spec, schedule, provider="cuda")


from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def head_argmax(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p["r"]
    tiles, vocab = (p["tiles"], p["vocab"])
    padded = 1 << (tiles - 1).bit_length()

    @kernel
    def compute(
        maximum: Tensor((r, tiles), "float32"),
        index: Tensor((r, tiles), "int32"),
        active: Tensor((r,), "int32"),
    ) -> Tensor((r,), "int32"):
        predicted = output((r,), "int32")
        with T.Kernel(r, threads=256) as row:
            values = T.alloc_fragment((padded,), "float32")
            best = T.alloc_fragment((1,), "float32")
            ids = T.alloc_fragment((padded,), "int32")
            first = T.alloc_fragment((1,), "int32")
            for j in T.Parallel(padded):
                values[j] = T.if_then_else(
                    j < tiles, maximum[row, T.min(j, tiles - 1)], -T.infinity("float32")
                )
            T.reduce_max(values, best, dim=0)
            for j in T.Parallel(padded):
                ids[j] = T.if_then_else(
                    (j < tiles) & (values[j] == best[0]),
                    index[row, T.min(j, tiles - 1)],
                    vocab,
                )
            T.reduce_min(ids, first, dim=0)
            for j in T.Parallel(1):
                predicted[row] = T.if_then_else(active[row] != 0, first[0], -1)
        return predicted

    return compute


def head_linear(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r = p["r"]
    k, o, m, n, bk = (p[name] for name in ("k", "o", "m", "n", "bk"))
    tiles = T.ceildiv(o, n)

    @kernel
    def compute(
        x: Tensor((r, k), "bfloat16"), w: Tensor((o, k), "bfloat16")
    ) -> tuple[Tensor((r, tiles), "float32"), Tensor((r, tiles), "int32")]:
        maximum = output((r, tiles), "float32")
        index = output((r, tiles), "int32")
        with T.Kernel(T.ceildiv(r, m), tiles, threads=128) as (by, bx):
            a = T.alloc_shared((m, bk), "bfloat16")
            b = T.alloc_shared((n, bk), "bfloat16")
            acc = T.alloc_fragment((m, n), "float32")
            T.clear(acc)
            best = T.alloc_fragment((m,), "float32")
            ids = T.alloc_fragment((m, n), "int32")
            first = T.alloc_fragment((m,), "int32")
            for tile in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
                for i, j in T.Parallel(m, bk):
                    a[i, j] = T.if_then_else(
                        (by * m + i < r) & (tile * bk + j < k),
                        x[T.min(by * m + i, r - 1), T.min(tile * bk + j, k - 1)],
                        0,
                    )
                for i, j in T.Parallel(n, bk):
                    b[i, j] = T.if_then_else(
                        (bx * n + i < o) & (tile * bk + j < k),
                        w[T.min(bx * n + i, o - 1), T.min(tile * bk + j, k - 1)],
                        0,
                    )
                T.gemm(a, b, acc, transpose_B=True)
            for i, j in T.Parallel(m, n):
                if bx * n + j >= o:
                    acc[i, j] = -T.infinity("float32")
            T.reduce_max(acc, best, dim=1)
            for i, j in T.Parallel(m, n):
                ids[i, j] = T.if_then_else(
                    (acc[i, j] == best[i]) & (bx * n + j < o), bx * n + j, o
                )
            T.reduce_min(ids, first, dim=1)
            for i in T.Parallel(m):
                if by * m + i < r:
                    maximum[by * m + i, bx] = best[i]
                    index[by * m + i, bx] = first[i]
        return (maximum, index)

    return compute


def bf16(spec: KernelSpec, schedule: Mapping[str, object]):
    """Vector-loaded BF16 tensor-core projection for the large vocabulary head."""
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, o = (p["r"], p["k"], p["o"])
    m = 16
    n, depth, threads = (
        p.get("columns", 64),
        p.get("depth", 128),
        p.get("threads", 128),
    )
    parts = p.get("partitions", 1)
    if o % n or k % (depth * parts):
        raise ValueError("invalid BF16 projection partition")
    shape = (r, parts, o) if parts > 1 else (r, o)

    @kernel
    def compute(x: Tensor((r, k), "bfloat16"), w: Tensor((o, k), "bfloat16")) -> Tensor(
        shape, "float32"
    ):
        out = output(shape, "float32")
        with T.Kernel(o // n, T.ceildiv(r, m), parts, threads=threads) as (
            bx,
            by,
            part,
        ):
            lhs = T.alloc_shared((m, depth), "bfloat16")
            rhs = T.alloc_shared((n, depth), "bfloat16")
            accum = T.alloc_fragment((m, n), "float32")
            T.clear(accum)
            for tile in T.Pipelined(k // (depth * parts), num_stages=2):
                start = (part * (k // (depth * parts)) + tile) * depth
                for i, j in T.Parallel(m, depth):
                    lhs[i, j] = T.if_then_else(
                        by * m + i < r, x[T.min(by * m + i, r - 1), start + j], 0
                    )
                T.copy(w[bx * n : bx * n + n, start : start + depth], rhs)
                T.gemm(lhs, rhs, accum, transpose_B=True)
            for i, j in T.Parallel(m, n):
                if by * m + i < r:
                    if parts > 1:
                        out[by * m + i, part, bx * n + j] = accum[i, j]
                    else:
                        out[by * m + i, bx * n + j] = accum[i, j]
        return out

    return compute


def speculative(spec: KernelSpec, schedule: Mapping[str, object]):
    """Vector-loaded BF16 tensor-core projection for the large vocabulary head."""
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, o = (p["r"], p["k"], p["o"])
    m = p.get("block_m", 32)
    n, depth, threads = (
        p.get("columns", 64),
        p.get("depth", 128),
        p.get("threads", 128),
    )
    parts = p.get("partitions", 1)
    if o % n or k % (depth * parts):
        raise ValueError("invalid BF16 projection partition")
    shape = (r, parts, o) if parts > 1 else (r, o)

    @kernel
    def compute(x: Tensor((r, k), "bfloat16"), w: Tensor((o, k), "bfloat16")) -> Tensor(
        shape, "float32"
    ):
        out = output(shape, "float32")
        with T.Kernel(o // n, T.ceildiv(r, m), parts, threads=threads) as (
            bx,
            by,
            part,
        ):
            lhs = T.alloc_shared((m, depth), "bfloat16")
            rhs = T.alloc_shared((n, depth), "bfloat16")
            accum = T.alloc_fragment((m, n), "float32")
            T.clear(accum)
            for tile in T.Pipelined(k // (depth * parts), num_stages=2):
                start = (part * (k // (depth * parts)) + tile) * depth
                for i, j in T.Parallel(m, depth):
                    lhs[i, j] = T.if_then_else(
                        by * m + i < r, x[T.min(by * m + i, r - 1), start + j], 0
                    )
                T.copy(w[bx * n : bx * n + n, start : start + depth], rhs)
                T.gemm(lhs, rhs, accum, transpose_B=True)
            for i, j in T.Parallel(m, n):
                if by * m + i < r:
                    if parts > 1:
                        out[by * m + i, part, bx * n + j] = accum[i, j]
                    else:
                        out[by * m + i, bx * n + j] = accum[i, j]
        return out

    return compute


from .kernel import (
    argmax,
    argmax_packed,
    embedding,
    embedding_packed,
    native_argmax,
    native_embedding,
    packed_word_argmax,
    packed_word_embedding,
)
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "cuda"))


from .kernel import argmax_partial as argmax_partial, argmax_merge as argmax_merge
