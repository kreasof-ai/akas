"""Shared TileLang computations for convolution."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def convolution(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, c, length = (p["r"], p["c"], p.get("length", 3))
    chunk = p.get("chunk", 1)

    @kernel
    def compute(
        x: Tensor((r * chunk * 3 * c,), "float32"),
        w: Tensor((c * length,), "float32"),
        state: Tensor((r, length - 1, c), "float32"),
        enabled: Tensor((r,), "int32"),
    ) -> tuple[
        Tensor((r, length - 1, c), "float32"), Tensor((r * chunk * c,), "float32")
    ]:
        out = output((r * chunk * c,), "float32", position=3)
        with T.Kernel(T.ceildiv(c, 64), r, threads=64) as (block, row):
            tx = T.get_thread_binding()
            col = block * 64 + tx
            if col < c:
                for token in T.serial(chunk):
                    index = row * chunk + token
                    if token < enabled[row]:
                        current = T.alloc_var("float32")
                        acc = T.alloc_var("float32")
                        current = (
                            x[index * 3 * c + col] * x[index * 3 * c + 2 * c + col]
                        )
                        acc = current * w[col * length + length - 1]
                        for i in T.serial(length - 1):
                            acc += state[row, i, col] * w[col * length + i]
                        out[index * c + col] = x[index * 3 * c + c + col] * acc
                        for i in T.serial(length - 2):
                            state[row, i, col] = state[row, i + 1, col]
                        state[row, length - 2, col] = current
                    else:
                        out[index * c + col] = 0
        return (state, out)

    return compute


def convolution_flat(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    c = p.get("c")
    r = p.get("r", 1)

    @kernel
    def compute(
        x: Tensor((r * 3 * c,), "float32"),
        w: Tensor((c * 3,), "float32"),
        state: Tensor((2 * c,), "float32"),
        pos: Tensor((2,), "int32"),
    ) -> tuple[Tensor((2 * c,), "float32"), Tensor((r * c,), "float32")]:
        out = output((r * c,), "float32", position=3)
        with T.Kernel(T.ceildiv(c, 256), threads=256) as block:
            for lane in T.Parallel(256):
                col = block * 256 + lane
                if col < c:
                    left = T.alloc_var("float32")
                    right = T.alloc_var("float32")
                    current = T.alloc_var("float32")
                    left = state[col]
                    right = state[c + col]
                    for row in T.serial(r):
                        if row < pos[1]:
                            current = (
                                x[row * (3 * c) + col] * x[row * (3 * c) + 2 * c + col]
                            )
                            out[row * c + col] = x[row * (3 * c) + c + col] * (
                                left * w[col * 3]
                                + right * w[col * 3 + 1]
                                + current * w[col * 3 + 2]
                            )
                            left = right
                            right = current
                        else:
                            out[row * c + col] = 0
                    state[col] = left
                    state[c + col] = right
        return (state, out)

    return compute
