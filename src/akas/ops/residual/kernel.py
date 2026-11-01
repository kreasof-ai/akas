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
