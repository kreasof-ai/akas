"""Shared TileLang computations for mtp."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def mtp_cast(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, c = (p["r"], p["c"])

    @kernel
    def compute(x: Tensor((r, c), "float32")) -> Tensor((r, c), "bfloat16"):
        out = output((r, c), "bfloat16")
        with T.Kernel(r, T.ceildiv(c, 256), threads=256) as (b, tile):
            for j in T.Parallel(256):
                if tile * 256 + j < c:
                    out[b, tile * 256 + j] = x[b, tile * 256 + j]
        return out

    return compute


def mtp_join(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, c = (p["r"], p["c"])
    eps = p["eps"]

    @kernel
    def compute(
        embedding: Tensor((r, c), "bfloat16"),
        hidden: Tensor((r, c), "bfloat16"),
        embedding_weight: Tensor((c,), "bfloat16"),
        hidden_weight: Tensor((c,), "bfloat16"),
    ) -> Tensor((r, 2 * c), "bfloat16"):
        out = output((r, 2 * c), "bfloat16")
        with T.Kernel(r, threads=256) as b:
            values = T.alloc_fragment((2, c), "float32")
            squares = T.alloc_fragment((2, c), "float32")
            sums = T.alloc_fragment((2,), "float32")
            for i, j in T.Parallel(2, c):
                if i == 0:
                    values[i, j] = embedding[b, j]
                else:
                    values[i, j] = hidden[b, j]
                squares[i, j] = values[i, j] * values[i, j]
            T.reduce_sum(squares, sums, dim=1)
            for i, j in T.Parallel(2, c):
                weight = T.alloc_var("float32")
                if i == 0:
                    weight = embedding_weight[j]
                else:
                    weight = hidden_weight[j]
                out[b, i * c + j] = (
                    values[i, j] * T.rsqrt(sums[i] / c + eps) * (1 + weight)
                )
        return out

    return compute
