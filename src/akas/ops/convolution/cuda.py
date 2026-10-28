"""CUDA specializations and shared bindings for convolution."""

from .kernel import convolution, convolution_flat, verify, commit
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "cuda"))


from akas.kernels.kernel import Tensor, kernel, output


def parallel_prefill(spec, schedule):
    """Parallel causal taps with one block owning each channel history.

    Snapshot history before any write; only this block reads or updates these
    channels, so an inactive or ragged request retains the correct prefix.
    """
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    r, c, chunk, length = (p[n] for n in ("r", "c", "chunk", "length"))
    columns = p.get("columns", 8 if chunk >= 128 else 16)
    if columns not in (8, 16, 32) or chunk * columns * 4 > 90000:
        raise ValueError("parallel convolution tile exceeds shared-memory capacity")
    threads = p.get("threads", 256)

    @kernel
    def compute(
        x: Tensor(r * chunk * 3 * c, "float32"),
        w: Tensor(c * length, "float32"),
        state: Tensor((r, length - 1, c), "float32"),
        enabled: Tensor(r, "int32"),
    ) -> tuple[Tensor((r, length - 1, c), "float32"), Tensor(r * chunk * c, "float32")]:
        out = output(r * chunk * c, "float32", position=3)
        with T.Kernel(T.ceildiv(c, columns), r, threads=threads) as (bx, row):
            products = T.alloc_shared((chunk, columns), "float32")
            history = T.alloc_shared((length - 1, columns), "float32")
            coeff = T.alloc_shared((length, columns), "float32")
            for i, j in T.Parallel(chunk, columns):
                col = bx * columns + j
                if col < c and i < enabled[row]:
                    products[i, j] = (
                        x[(row * chunk + i) * 3 * c + col]
                        * x[(row * chunk + i) * 3 * c + 2 * c + col]
                    )
                else:
                    products[i, j] = 0
            for i, j in T.Parallel(length - 1, columns):
                if bx * columns + j < c:
                    history[i, j] = state[row, i, bx * columns + j]
            for i, j in T.Parallel(length, columns):
                if bx * columns + j < c:
                    coeff[i, j] = w[(bx * columns + j) * length + i]
            T.sync_threads()
            for i, j in T.Parallel(chunk, columns):
                col = bx * columns + j
                acc = T.alloc_var("float32")
                if col < c:
                    if i < enabled[row]:
                        acc = products[i, j] * coeff[length - 1, j]
                        for tap in T.unroll(length - 1):
                            if i + tap >= length - 1:
                                acc += (
                                    products[i + tap - (length - 1), j] * coeff[tap, j]
                                )
                            else:
                                acc += history[i + tap, j] * coeff[tap, j]
                        out[(row * chunk + i) * c + col] = (
                            x[(row * chunk + i) * 3 * c + c + col] * acc
                        )
                    else:
                        out[(row * chunk + i) * c + col] = 0
            for i, j in T.Parallel(length - 1, columns):
                col = bx * columns + j
                if col < c:
                    if enabled[row] + i >= length - 1:
                        state[row, i, col] = products[
                            enabled[row] + i - (length - 1), j
                        ]
                    else:
                        state[row, i, col] = history[enabled[row] + i, j]
        return state, out

    return compute
