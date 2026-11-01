"""Shared TileLang computations for residual."""

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def add(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    c = p.get("c")
    r = p.get("r", 1)
    n = r * c

    @kernel
    def compute(x: Tensor((n,), "float32"), y: Tensor((n,), "float32")) -> Tensor(
        (n,), "float32"
    ):
        out = output((n,), "float32")
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            for lane in T.Parallel(256):
                i = block * 256 + lane
                if i < n:
                    out[i] = x[i] + y[i]
        return out

    return compute
