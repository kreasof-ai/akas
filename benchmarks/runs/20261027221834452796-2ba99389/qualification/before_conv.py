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


def verify(spec: KernelSpec, schedule: Mapping[str, object]):
    """Evaluate a token block and retain each history prefix without committing it."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows, channels, length, chunk = (p[n] for n in ("r", "c", "length", "chunk"))
    if length < 2 or chunk < 1:
        raise ValueError(
            "Convolution verification requires a positive block and history"
        )

    @kernel
    def compute(
        x: Tensor((rows * chunk * 3 * channels,), "float32"),
        w: Tensor((channels * length,), "float32"),
        state: Tensor((rows, length - 1, channels), "float32"),
        enabled: Tensor((rows,), "int32"),
    ) -> tuple[
        Tensor((rows, chunk, length - 1, channels), "float32"),
        Tensor((rows * chunk * channels,), "float32"),
    ]:
        journal = output((rows, chunk, length - 1, channels), "float32", position=3)
        out = output((rows * chunk * channels,), "float32", position=4)
        with T.Kernel(T.ceildiv(channels, 64), rows, threads=64) as (block, row):
            tx = T.get_thread_binding()
            col = block * 64 + tx
            history = T.alloc_local((length - 1,), "float32")
            if col < channels:
                for tap in T.serial(length - 1):
                    history[tap] = state[row, tap, col]
                for token in T.serial(chunk):
                    index = row * chunk + token
                    if token < enabled[row]:
                        current = T.alloc_var("float32")
                        acc = T.alloc_var("float32")
                        current = (
                            x[index * 3 * channels + col]
                            * x[index * 3 * channels + 2 * channels + col]
                        )
                        acc = current * w[col * length + length - 1]
                        for tap in T.serial(length - 1):
                            acc += history[tap] * w[col * length + tap]
                        out[index * channels + col] = (
                            x[index * 3 * channels + channels + col] * acc
                        )
                        for tap in T.serial(length - 2):
                            history[tap] = history[tap + 1]
                        history[length - 2] = current
                    else:
                        out[index * channels + col] = 0
                    for tap in T.serial(length - 1):
                        journal[row, token, tap, col] = history[tap]
        return journal, out

    return compute


def commit(spec: KernelSpec, schedule: Mapping[str, object]):
    """Commit one verified history prefix; a rejected block leaves history intact."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows, channels, length, chunk = (p[n] for n in ("r", "c", "length", "chunk"))

    @kernel
    def compute(
        state: Tensor((rows, length - 1, channels), "float32"),
        journal: Tensor((rows, chunk, length - 1, channels), "float32"),
        accepted: Tensor((rows,), "int32"),
    ) -> Tensor((rows, length - 1, channels), "float32"):
        with T.Kernel(T.ceildiv((length - 1) * channels, 128), rows, threads=128) as (
            block,
            row,
        ):
            for lane in T.Parallel(128):
                index = block * 128 + lane
                if index < (length - 1) * channels and 0 < accepted[row] <= chunk:
                    state[row, index // channels, index % channels] = journal[
                        row, accepted[row] - 1, index // channels, index % channels
                    ]
        return state

    return compute
