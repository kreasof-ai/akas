"""CUDA specializations and shared bindings for normalization."""

from .kernel import add_rms, kvnorm, qnorm, rms
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "cuda"))


def post_add_rms(spec, schedule):
    """Sandwich normalization, residual addition and the next normalization."""
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    r, c, eps = p["r"], p["c"], p["eps"]
    dtype = p.get("output_dtype", "float32")
    threads = min(256, c)

    @kernel
    def compute(
        x: Tensor(r * c, "float32"),
        post_weight: Tensor(c, "float32"),
        residual: Tensor(r * c, "float32"),
        w: Tensor(c, "float32"),
    ) -> tuple[Tensor(r * c, "float32"), Tensor(r * c, dtype)]:
        hidden = output(r * c, "float32")
        out = output(r * c, dtype)
        with T.Kernel(r, threads=threads) as row:
            values = T.alloc_fragment((c,), "float32")
            squares = T.alloc_fragment((c,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(c):
                values[j] = x[row * c + j]
                squares[j] = values[j] * values[j]
            T.reduce_sum(squares, total, dim=0)
            for j in T.Parallel(c):
                values[j] = values[j] * T.rsqrt(total[0] / c + eps) * post_weight[j]
                values[j] = residual[row * c + j] + values[j]
                hidden[row * c + j] = values[j]
                squares[j] = values[j] * values[j]
            T.reduce_sum(squares, total, dim=0)
            for j in T.Parallel(c):
                out[row * c + j] = values[j] * T.rsqrt(total[0] / c + eps) * w[j]
        return hidden, out

    return compute
