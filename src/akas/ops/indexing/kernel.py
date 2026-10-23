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


def select_token(spec, schedule):
    """Gather one token per request from a flattened request/token matrix."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    rows, chunk, width, index = (p[n] for n in ("r", "chunk", "c", "index"))
    if not 0 <= index < chunk:
        raise ValueError("token index exceeds the block")

    @kernel
    def compute(x: Tensor((rows * chunk * width,), "float32")) -> Tensor(
        (rows * width,), "float32"
    ):
        out = output((rows * width,), "float32")
        with T.Kernel(rows, T.ceildiv(width, 256), threads=256) as (row, tile):
            for lane in T.Parallel(256):
                column = tile * 256 + lane
                if column < width:
                    out[row * width + column] = x[
                        (row * chunk + index) * width + column
                    ]
        return out

    return compute


def concatenate(spec, schedule):
    """Concatenate two row-major feature matrices into a fresh result."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    rows, left_width, right_width = (p[n] for n in ("r", "left", "right"))
    width = left_width + right_width

    @kernel
    def compute(
        left: Tensor((rows * left_width,), "float32"),
        right: Tensor((rows * right_width,), "float32"),
    ) -> Tensor((rows * width,), "float32"):
        out = output((rows * width,), "float32")
        with T.Kernel(rows, T.ceildiv(width, 256), threads=256) as (row, tile):
            for lane in T.Parallel(256):
                column = tile * 256 + lane
                if column < width:
                    if column < left_width:
                        out[row * width + column] = left[row * left_width + column]
                    else:
                        out[row * width + column] = right[
                            row * right_width + column - left_width
                        ]
        return out

    return compute


def replace_token(spec, schedule):
    """Return a token block with one column replaced, preserving the source."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    rows, chunk, index = (p[n] for n in ("r", "chunk", "index"))
    if not 0 <= index < chunk:
        raise ValueError("token index exceeds the block")

    @kernel
    def compute(
        x: Tensor((rows * chunk,), "int32"), token: Tensor((rows,), "int32")
    ) -> Tensor((rows * chunk,), "int32"):
        out = output((rows * chunk,), "int32")
        with T.Kernel(T.ceildiv(rows * chunk, 128), threads=128) as block:
            for lane in T.Parallel(128):
                i = block * 128 + lane
                if i < rows * chunk:
                    out[i] = T.if_then_else(i % chunk == index, token[i // chunk], x[i])
        return out

    return compute
