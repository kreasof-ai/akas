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
    time_tile = p.get("time_tile", chunk)
    if (
        type(time_tile) is not int
        or time_tile < 1
        or time_tile > chunk
        or columns not in (8, 16, 32, 64, 128)
        or (time_tile + 3 * length - 2) * columns * 4 > 90000
    ):
        raise ValueError("parallel convolution tile exceeds shared-memory capacity")
    threads = p.get("threads", 256)
    output_dtype = p.get("output_dtype", "float32")
    if output_dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("convolution output requires floating storage")

    @kernel
    def compute(
        x: Tensor(r * chunk * 3 * c, "float32"),
        w: Tensor(c * length, "float32"),
        state: Tensor((r, length - 1, c), "float32"),
        enabled: Tensor(r, "int32"),
    ) -> tuple[
        Tensor((r, length - 1, c), "float32"), Tensor(r * chunk * c, output_dtype)
    ]:
        out = output(r * chunk * c, output_dtype, position=3)
        with T.Kernel(T.ceildiv(c, columns), r, threads=threads) as (bx, row):
            T.annotate_min_blocks_per_sm(p.get("min_blocks", 1))
            products = T.alloc_shared((time_tile, columns), "float32")
            history = T.alloc_shared((length - 1, columns), "float32")
            coeff = T.alloc_shared((length, columns), "float32")
            for i, j in T.Parallel(length - 1, columns):
                if bx * columns + j < c:
                    history[i, j] = state[row, i, bx * columns + j]
            for i, j in T.Parallel(length, columns):
                if bx * columns + j < c:
                    coeff[i, j] = w[(bx * columns + j) * length + i]
            T.sync_threads()
            for tile in T.serial(T.ceildiv(chunk, time_tile)):
                start = tile * time_tile
                count = T.min(T.max(enabled[row] - start, 0), time_tile)
                for i, j in T.Parallel(time_tile, columns):
                    col = bx * columns + j
                    if col < c and start + i < chunk and i < count:
                        products[i, j] = (
                            x[(row * chunk + start + i) * 3 * c + col]
                            * x[(row * chunk + start + i) * 3 * c + 2 * c + col]
                        )
                    else:
                        products[i, j] = 0
                T.sync_threads()
                for i, j in T.Parallel(time_tile, columns):
                    col = bx * columns + j
                    acc = T.alloc_var("float32")
                    if col < c and start + i < chunk:
                        if i < count:
                            acc = products[i, j] * coeff[length - 1, j]
                            for tap in T.unroll(length - 1):
                                if i + tap >= length - 1:
                                    acc += (
                                        products[i + tap - (length - 1), j]
                                        * coeff[tap, j]
                                    )
                                else:
                                    acc += history[i + tap, j] * coeff[tap, j]
                            out[(row * chunk + start + i) * c + col] = (
                                x[(row * chunk + start + i) * 3 * c + c + col] * acc
                            )
                        else:
                            out[(row * chunk + start + i) * c + col] = 0
                T.sync_threads()
                # Read the complete prior history before publishing any update.
                # Short or empty tiles can retain entries from that history.
                next_history = T.alloc_shared((length - 1, columns), "float32")
                for i, j in T.Parallel(length - 1, columns):
                    if bx * columns + j < c:
                        if count + i >= length - 1:
                            next_history[i, j] = products[count + i - (length - 1), j]
                        else:
                            next_history[i, j] = history[count + i, j]
                T.sync_threads()
                T.copy(next_history, history)
                T.sync_threads()
            for i, j in T.Parallel(length - 1, columns):
                if bx * columns + j < c:
                    state[row, i, bx * columns + j] = history[i, j]
        return state, out

    return compute
