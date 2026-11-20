"""Shared selective-scan computations with F32 recurrent state."""

from akas.kernels.kernel import Tensor, kernel, output


def selective_scan(spec, schedule):
    """Run one channel's sixteen-state recurrence in registers per work item."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    sequences, frames, channels, states = (
        p[key] for key in ("sequences", "frames", "c", "s")
    )
    rows = sequences * frames
    group = p.get("rows_per_block", 1)
    if sequences % group or states != 16 or frames < 1:
        raise ValueError("Mamba scan requires whole sequence groups and state width 16")

    @kernel
    def compute(
        u: Tensor((rows, channels), "float32"),
        delta: Tensor((rows, channels), "float32"),
        decay: Tensor((channels, states), "float32"),
        b: Tensor((rows, states), "float32"),
        c: Tensor((rows, states), "float32"),
        skip: Tensor((channels,), "float32"),
        gate: Tensor((rows, channels), "float32"),
    ) -> Tensor((rows, channels), "float32"):
        out = output((rows, channels), "float32")
        with T.Kernel(sequences // group, T.ceildiv(channels, 128), threads=128) as (
            block,
            tile,
        ):
            tx = T.get_thread_binding()
            channel = tile * 128 + tx
            if channel < channels:
                for item in T.serial(group):
                    sequence = block * group + item
                    state = T.alloc_local((states,), "float32")
                    rate = T.alloc_local((states,), "float32")
                    for n in T.unroll(states):
                        state[n] = 0
                        rate[n] = decay[channel, n]
                    for frame in T.serial(frames):
                        row = sequence * frames + frame
                        step = delta[row, channel]
                        x = u[row, channel]
                        value = T.alloc_var("float32")
                        value = 0
                        for n in T.unroll(states):
                            state[n] = (
                                T.exp(step * rate[n]) * state[n] + step * x * b[row, n]
                            )
                            value += state[n] * c[row, n]
                        value += x * skip[channel]
                        z = gate[row, channel]
                        out[row, channel] = value * z / (1 + T.exp(-z))
        return out

    return compute


def reverse(spec, schedule):
    """Reverse sequence order while preserving channel layout."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    sequences, frames, channels = (p[key] for key in ("sequences", "frames", "c"))
    rows = sequences * frames

    @kernel
    def compute(x: Tensor((sequences, frames, channels), "float32")) -> Tensor(
        (sequences, frames, channels), "float32"
    ):
        out = output((sequences, frames, channels), "float32")
        with T.Kernel(T.ceildiv(rows * channels, 256), threads=256) as tile:
            index = tile * 256 + T.get_thread_binding()
            if index < rows * channels:
                sequence = index // (frames * channels)
                offset = index % (frames * channels)
                frame, channel = offset // channels, offset % channels
                out[sequence, frame, channel] = x[sequence, frames - 1 - frame, channel]
        return out

    return compute


def bidirectional_join(spec, schedule):
    """Concatenate forward/backward branches, each with its residual input."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows, channels = (p[key] for key in ("r", "c"))

    @kernel
    def compute(
        original: Tensor((rows, channels), "float32"),
        forward: Tensor((rows, channels), "float32"),
        backward: Tensor((rows, channels), "float32"),
    ) -> Tensor((rows, 2 * channels), "float32"):
        out = output((rows, 2 * channels), "float32")
        with T.Kernel(T.ceildiv(rows * channels, 256), threads=256) as tile:
            index = tile * 256 + T.get_thread_binding()
            if index < rows * channels:
                row, channel = index // channels, index % channels
                out[row, channel] = forward[row, channel] + original[row, channel]
                out[row, channels + channel] = (
                    backward[row, channel] + original[row, channel]
                )
        return out

    return compute
