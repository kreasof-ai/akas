"""Shared TileLang computations for indexing."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def clear_rows(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    rows, pool, width = (p["rows"], p["pool"], p["width"])
    dtype, threads = (p["dtype"], p.get("threads", 256))
    if any((type(n) is not int or n < 1 for n in (rows, pool, width, threads))):
        raise ValueError("row geometry must contain positive integers")

    @kernel
    def compute(
        mapping: Tensor((rows,), "int32"), state: Tensor((pool, width), dtype)
    ) -> Tensor((pool, width), dtype):
        with T.Kernel(rows, T.ceildiv(width, threads), threads=threads) as (row, tile):
            request = mapping[row]
            if (request >= 0) & (request < pool):
                for i in T.Parallel(threads):
                    col = tile * threads + i
                    if col < width:
                        state[request, col] = 0
        return state

    return compute
