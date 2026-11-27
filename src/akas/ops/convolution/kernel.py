"""Shared TileLang computations for convolution."""

from collections.abc import Mapping
from akas.kernels.spec import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def image_patches(spec, schedule):
    """Gather strided, zero-padded 2D patches for a matrix projection."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    width, height, channels = (p[k] for k in ("width", "height", "channels"))
    kernel_size, stride, padding = (p[k] for k in ("kernel", "stride", "padding"))
    ow = (width + 2 * padding - kernel_size) // stride + 1
    oh = (height + 2 * padding - kernel_size) // stride + 1
    taps = kernel_size * kernel_size
    depth = channels * taps
    size = ow * oh * depth

    @kernel
    def compute(
        image: Tensor((width, height, channels), "float32"),
    ) -> Tensor((ow * oh, depth), "float32"):
        out = output((ow * oh, depth), "float32")
        with T.Kernel(T.ceildiv(size, 128), threads=128) as tile:
            for lane in T.Parallel(128):
                i = tile * 128 + lane
                if i < size:
                    row, col = i // depth, i % depth
                    x = row // oh * stride + col % kernel_size - padding
                    y = row % oh * stride + col % taps // kernel_size - padding
                    out[row, col] = T.if_then_else(
                        x >= 0 and x < width and y >= 0 and y < height,
                        image[x, y, col // taps],
                        0,
                    )
        return out

    return compute


def convolution(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, c, length = (p["r"], p["c"], p.get("length", 3))
    chunk = p.get("chunk", 1)
    output_dtype = p.get("output_dtype", "float32")
    if output_dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("convolution output requires floating storage")

    @kernel
    def compute(
        x: Tensor((r * chunk * 3 * c,), "float32"),
        w: Tensor((c * length,), "float32"),
        state: Tensor((r, length - 1, c), "float32"),
        enabled: Tensor((r,), "int32"),
    ) -> tuple[
        Tensor((r, length - 1, c), "float32"), Tensor((r * chunk * c,), output_dtype)
    ]:
        out = output((r * chunk * c,), output_dtype, position=3)
        with T.Kernel(T.ceildiv(c, 64), r, threads=64) as (block, row):
            tx = T.get_thread_binding()
            col = block * 64 + tx
            if col < c:
                history = T.alloc_local((length - 1,), "float32")
                for i in T.unroll(length - 1):
                    history[i] = state[row, i, col]
                for token in T.serial(chunk):
                    index = row * chunk + token
                    if token < enabled[row]:
                        current = T.alloc_var("float32")
                        acc = T.alloc_var("float32")
                        current = (
                            x[index * 3 * c + col] * x[index * 3 * c + 2 * c + col]
                        )
                        acc = current * w[col * length + length - 1]
                        for i in T.unroll(length - 1):
                            acc += history[i] * w[col * length + i]
                        out[index * c + col] = x[index * 3 * c + c + col] * acc
                        for i in T.unroll(length - 2):
                            history[i] = history[i + 1]
                        history[length - 2] = current
                    else:
                        out[index * c + col] = 0
                for i in T.unroll(length - 1):
                    state[row, i, col] = history[i]
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


def causal_depthwise(spec, schedule):
    """Finite channelwise FIR over one left-padded sequence."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows, width, taps = (p[key] for key in ("r", "c", "taps"))

    @kernel
    def compute(
        x: Tensor((rows, width), "float32"),
        w: Tensor((width, taps), "float32"),
        bias: Tensor((width,), "float32"),
    ) -> Tensor((rows, width), "float32"):
        out = output((rows, width), "float32")
        with T.Kernel(rows, T.ceildiv(width, 128), threads=128) as (row, tile):
            for lane in T.Parallel(128):
                column = tile * 128 + lane
                if column < width:
                    value = T.alloc_var("float32")
                    value = bias[column]
                    for j in T.unroll(taps):
                        step = row + j - taps + 1
                        if step >= 0:
                            value += x[step, column] * w[column, j]
                    out[row, column] = value
        return out

    return compute


def causal_depthwise_sequences(spec, schedule):
    """Finite channelwise FIR with independent sequence-local left padding."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    sequences, frames, width, taps = (
        p[key] for key in ("sequences", "frames", "c", "taps")
    )

    @kernel
    def compute(
        x: Tensor((sequences, frames, width), "float32"),
        w: Tensor((width, taps), "float32"),
        bias: Tensor((width,), "float32"),
    ) -> Tensor((sequences, frames, width), "float32"):
        out = output((sequences, frames, width), "float32")
        with T.Kernel(sequences, T.ceildiv(width, 128), threads=128) as (
            sequence,
            tile,
        ):
            for lane in T.Parallel(128):
                column = tile * 128 + lane
                if column < width:
                    value = T.alloc_var("float32")
                    for frame in T.serial(frames):
                        value = bias[column]
                        for j in T.unroll(taps):
                            step = frame + j - taps + 1
                            if step >= 0:
                                value += x[sequence, step, column] * w[column, j]
                        out[sequence, frame, column] = value
        return out

    return compute


def causal_depthwise_state(spec, schedule):
    """Stateful channelwise causal FIR across multi-slot streaming chunks."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    slots, chunk, width, taps = (p[k] for k in ("slots", "chunk", "c", "taps"))
    rows = slots * chunk
    hist_len = taps - 1

    @kernel
    def compute(
        x: Tensor((rows, width), "float32"),
        w: Tensor((width, taps), "float32"),
        bias: Tensor((width,), "float32"),
        state: Tensor((slots, hist_len, width), "float32"),
        lengths: Tensor((slots,), "int32"),
    ) -> tuple[
        Tensor((slots, hist_len, width), "float32"),
        Tensor((rows, width), "float32"),
    ]:
        out = output((rows, width), "float32")
        with T.Kernel(slots, T.ceildiv(width, 128), threads=128) as (slot, tile):
            live = lengths[slot]
            for lane in T.Parallel(128):
                col = tile * 128 + lane
                if col < width:
                    hist = T.alloc_local((hist_len,), "float32")
                    taps_w = T.alloc_local((taps,), "float32")
                    b = T.alloc_var("float32")
                    b = bias[col]
                    for k in T.unroll(hist_len):
                        hist[k] = state[slot, k, col]
                    for k in T.unroll(taps):
                        taps_w[k] = w[col, k]
                    for step in T.serial(chunk):
                        row = slot * chunk + step
                        if step < live:
                            cur = T.alloc_var("float32")
                            acc = T.alloc_var("float32")
                            cur = x[row, col]
                            acc = b + cur * taps_w[taps - 1]
                            for k in T.unroll(hist_len):
                                acc += hist[k] * taps_w[k]
                            out[row, col] = acc
                            for k in T.unroll(hist_len - 1):
                                hist[k] = hist[k + 1]
                            hist[hist_len - 1] = cur
                        else:
                            out[row, col] = 0
                    for k in T.unroll(hist_len):
                        state[slot, k, col] = hist[k]
        return state, out

    return compute


def istft_polar(spec, schedule):
    """Convert two-branch log-magnitude/phase spectra into real/imag vectors."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows = p["r"]
    n0, n1 = p.get("n0", 736), p.get("n1", 2208)
    b0, b1 = n0 // 2 + 1, n1 // 2 + 1
    w0, w1 = 2 * b0, 2 * b1
    total = w0 + w1
    max_log = p.get("max_log_mag", 4.605170185988092)

    @kernel
    def compute(
        spectrum: Tensor((rows, total), "float32"),
    ) -> tuple[Tensor((rows, w0), "float32"), Tensor((rows, w1), "float32")]:
        spec0 = output((rows, w0), "float32")
        spec1 = output((rows, w1), "float32")
        with T.Kernel(rows, T.ceildiv(b1, 128), threads=128) as (row, tile):
            for lane in T.Parallel(128):
                j = tile * 128 + lane
                if j < b0:
                    mag0 = T.exp(T.min(spectrum[row, j], T.float32(max_log)))
                    ph0 = spectrum[row, b0 + j]
                    spec0[row, j] = mag0 * T.cos(ph0)
                    spec0[row, b0 + j] = T.if_then_else(
                        (j == 0) | (j == b0 - 1), 0.0, mag0 * T.sin(ph0)
                    )
                if j < b1:
                    mag1 = T.exp(T.min(spectrum[row, w0 + j], T.float32(max_log)))
                    ph1 = spectrum[row, w0 + b1 + j]
                    spec1[row, j] = mag1 * T.cos(ph1)
                    spec1[row, b1 + j] = T.if_then_else(
                        (j == 0) | (j == b1 - 1), 0.0, mag1 * T.sin(ph1)
                    )
        return spec0, spec1

    return compute


def istft_ola(spec, schedule):
    """Materialize continuous overlap-added mixer inputs without mutating state."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    slots, frames = p["slots"], p["frames"]
    cols = frames * 16
    hop, n0, n1 = 184, 736, 2208
    t0, t1, m0_len, m1_len = n0 - hop, n1 - hop, 862, 126
    state_width = t0 + t1 + m0_len + m1_len
    span = cols * hop

    @kernel
    def compute(
        fr0: Tensor((slots * cols, n0), "float32"),
        fr1: Tensor((slots * cols, n1), "float32"),
        state: Tensor((slots, state_width), "float32"),
        lengths: Tensor((slots,), "int32"),
    ) -> tuple[
        Tensor((slots, m0_len + span + t0), "float32"),
        Tensor((slots, m1_len + span + t1), "float32"),
    ]:
        cat0 = output((slots, m0_len + span + t0), "float32")
        cat1 = output((slots, m1_len + span + t1), "float32")
        with T.Kernel(slots, T.ceildiv(m0_len + span + t1, 256), threads=256) as (
            slot,
            blk,
        ):
            live_cols = lengths[slot] * 16
            for lane in T.Parallel(256):
                idx = blk * 256 + lane
                if idx < m0_len + span + t0:
                    if idx < m0_len:
                        cat0[slot, idx] = state[slot, t0 + t1 + idx]
                    else:
                        s0 = idx - m0_len
                        v0 = T.alloc_var("float32")
                        v0 = T.if_then_else(s0 < t0, state[slot, s0], 0.0)
                        c0 = T.max(0, (s0 - n0 + hop) // hop)
                        for m in T.unroll(4):
                            c = c0 + m
                            off = s0 - c * hop
                            if (c >= 0) & (c < live_cols) & (off >= 0) & (off < n0):
                                v0 += fr0[slot * cols + c, off]
                        cat0[slot, idx] = v0
                if idx < m1_len + span + t1:
                    if idx < m1_len:
                        cat1[slot, idx] = state[slot, t0 + t1 + m0_len + idx]
                    else:
                        s1 = idx - m1_len
                        v1 = T.alloc_var("float32")
                        v1 = T.if_then_else(s1 < t1, state[slot, t0 + s1], 0.0)
                        c1 = T.max(0, (s1 - n1 + hop) // hop)
                        for m in T.unroll(12):
                            c = c1 + m
                            off = s1 - c * hop
                            if (c >= 0) & (c < live_cols) & (off >= 0) & (off < n1):
                                v1 += fr1[slot * cols + c, off]
                        cat1[slot, idx] = v1
        return cat0, cat1

    return compute


def istft_mix_commit(spec, schedule):
    """Apply the 127-tap two-branch FIR mixer and commit updated overlap/mixer state."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    slots, frames = p["slots"], p["frames"]
    cols = frames * 16
    hop, n0, n1 = 184, 736, 2208
    t0, t1, m0_len, m1_len, fir_k = n0 - hop, n1 - hop, 862, 126, 127
    state_width = t0 + t1 + m0_len + m1_len
    span = cols * hop

    @kernel
    def compute(
        cat0: Tensor((slots, m0_len + span + t0), "float32"),
        cat1: Tensor((slots, m1_len + span + t1), "float32"),
        mixer: Tensor((2, fir_k), "float32"),
        state: Tensor((slots, state_width), "float32"),
        lengths: Tensor((slots,), "int32"),
    ) -> tuple[
        Tensor((slots, state_width), "float32"),
        Tensor((slots, span), "float32"),
    ]:
        mid = output((slots, span), "float32")
        with T.Kernel(slots, T.ceildiv(T.max(span, state_width), 256), threads=256) as (
            slot,
            blk,
        ):
            live_span = lengths[slot] * 16 * hop
            sh0 = T.alloc_shared((256 + fir_k,), "float32")
            sh1 = T.alloc_shared((256 + fir_k,), "float32")
            mw0 = T.alloc_shared((fir_k,), "float32")
            mw1 = T.alloc_shared((fir_k,), "float32")
            tx = T.get_thread_binding()
            if tx < fir_k:
                mw0[tx] = mixer[0, tx]
                mw1[tx] = mixer[1, tx]
            base = blk * 256
            for step in T.unroll(2):
                pos = tx + step * 256
                if pos < 256 + fir_k - 1:
                    g = base + pos
                    sh0[pos] = T.if_then_else(g < m0_len + span, cat0[slot, g], 0.0)
                    sh1[pos] = T.if_then_else(g < m1_len + span, cat1[slot, g], 0.0)
            T.sync_threads()
            idx = base + tx
            if idx < span:
                if idx < live_span:
                    acc = T.alloc_var("float32")
                    acc = 0.0
                    for k in T.unroll(fir_k):
                        acc += mw0[k] * sh0[tx + k] + mw1[k] * sh1[tx + k]
                    mid[slot, idx] = acc
                else:
                    mid[slot, idx] = 0.0
            if (lengths[slot] > 0) & (idx < state_width):
                if idx < t0:
                    state[slot, idx] = cat0[slot, m0_len + live_span + idx]
                elif idx < t0 + t1:
                    state[slot, idx] = cat1[slot, m1_len + live_span + (idx - t0)]
                elif idx < t0 + t1 + m0_len:
                    state[slot, idx] = cat0[slot, live_span + (idx - t0 - t1)]
                else:
                    state[slot, idx] = cat1[slot, live_span + (idx - t0 - t1 - m0_len)]
        return state, mid

    return compute


def sinc_resample_limit(spec, schedule):
    """Resample 44160->44100 Hz (736->735), apply headroom gain + soft limiter, and update buffer."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    slots, frames = p["slots"], p["frames"]
    blocks = frames * 4
    orig, new, kl, buf_len = 736, 735, 770, 322
    in_span = blocks * orig
    out_span = blocks * new
    gain = p.get("gain", 0.70794578)
    knee = p.get("knee", 0.8912509381337456)

    @kernel
    def compute(
        mid: Tensor((slots, in_span), "float32"),
        filt: Tensor((new, kl), "float32"),
        rs_state: Tensor((slots, buf_len), "float32"),
        lengths: Tensor((slots,), "int32"),
    ) -> Tensor((slots, out_span), "float32"):
        pcm = output((slots, out_span), "float32")
        with T.Kernel(T.ceildiv(new, 64), slots * blocks, threads=64) as (
            out_tile,
            sb,
        ):
            slot = sb // blocks
            blk = sb % blocks
            live_blocks = lengths[slot] * 4
            tx = T.get_thread_binding()
            win = T.alloc_shared((kl,), "float32")
            for step in T.serial(T.ceildiv(kl, 64)):
                j = step * 64 + tx
                if j < kl:
                    src = blk * orig + j
                    win[j] = T.if_then_else(
                        src < buf_len,
                        rs_state[slot, src],
                        mid[slot, T.min(src - buf_len, in_span - 1)],
                    )
            T.sync_threads()
            o_idx = out_tile * 64 + tx
            if o_idx < new:
                if blk < live_blocks:
                    acc = T.alloc_var("float32")
                    acc = 0.0
                    for j in T.serial(kl):
                        acc += win[j] * filt[o_idx, j]
                    val = T.alloc_var("float32")
                    val = acc * T.float32(gain)
                    mag = T.abs(val)
                    if mag > T.float32(knee):
                        width = T.float32(1.0 - knee)
                        lim = T.float32(knee) + width * T.tanh(
                            (mag - T.float32(knee)) / width
                        )
                        val = T.if_then_else(val >= 0.0, lim, -lim)
                    pcm[slot, blk * new + o_idx] = val
                else:
                    pcm[slot, blk * new + o_idx] = 0.0
        return pcm

    return compute


def sinc_resample_commit(spec, schedule):
    """Commit the trailing 322 samples of each active slot into rs_state."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    slots, frames = p["slots"], p["frames"]
    in_span = frames * 4 * 736
    buf_len = 322

    @kernel
    def compute(
        mid: Tensor((slots, in_span), "float32"),
        rs_state: Tensor((slots, buf_len), "float32"),
        lengths: Tensor((slots,), "int32"),
    ) -> Tensor((slots, buf_len), "float32"):
        with T.Kernel(slots, T.ceildiv(buf_len, 128), threads=128) as (slot, blk):
            live_span = lengths[slot] * 4 * 736
            for lane in T.Parallel(128):
                idx = blk * 128 + lane
                if (lengths[slot] > 0) & (idx < buf_len):
                    rs_state[slot, idx] = mid[slot, live_span - buf_len + idx]
        return rs_state

    return compute


def residual_short_conv(spec, schedule):
    """Static or input-dependent depthwise causal convolution with residual gain."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    slots, chunk, width, taps = (p[n] for n in ("r", "chunk", "c", "taps"))
    groups, dynamic = p["groups"], p["dynamic"]
    x_shape = (slots * chunk, width)
    history_shape = (slots, taps - 1, width)
    weight_shape = (taps, width) if dynamic else (width, taps)
    dynamic_shape = (slots * chunk, groups * taps) if dynamic else x_shape
    scale = ((2 if dynamic else 1) * taps + 1) ** -0.5

    @T.macro
    def convolve(x, w, dynamic_w, history, lengths, out):
        with T.Kernel(slots, T.ceildiv(width, 128), threads=128) as (slot, tile):
            for lane in T.Parallel(128):
                col = tile * 128 + lane
                if col < width:
                    previous = T.alloc_local((taps - 1,), "float32")
                    for j in T.unroll(taps - 1):
                        previous[j] = history[slot, j, col]
                    for i in T.serial(chunk):
                        row = slot * chunk + i
                        if i < lengths[slot]:
                            current = T.cast(T.cast(x[row, col], "bfloat16"), "float32")
                            total = T.alloc_var("float32")
                            total = current
                            for lag in T.unroll(taps):
                                value = current if lag == 0 else previous[lag - 1]
                                if dynamic:
                                    group = col // (width // groups)
                                    coefficient = T.cast(w[lag, col], "float32")
                                    coefficient += T.cast(
                                        T.cast(
                                            dynamic_w[row, group * taps + lag],
                                            "bfloat16",
                                        ),
                                        "float32",
                                    )
                                else:
                                    coefficient = T.cast(
                                        w[col, taps - 1 - lag], "float32"
                                    )
                                total += value * coefficient
                            out[row, col] = T.cast(
                                T.cast(total * scale, "bfloat16"), "float32"
                            )
                            for lag in T.unroll(taps - 2):
                                previous[taps - 2 - lag] = previous[taps - 3 - lag]
                            previous[0] = current
                        else:
                            out[row, col] = 0
                    for j in T.unroll(taps - 1):
                        history[slot, j, col] = previous[j]

    if dynamic:

        @kernel
        def compute(
            x: Tensor(x_shape, "float32"),
            w: Tensor(weight_shape, "bfloat16"),
            dynamic_w: Tensor(dynamic_shape, "float32"),
            history: Tensor(history_shape, "float32"),
            lengths: Tensor((slots,), "int32"),
        ) -> tuple[Tensor(history_shape, "float32"), Tensor(x_shape, "float32")]:
            out = output(x_shape, "float32")
            convolve(x, w, dynamic_w, history, lengths, out)
            return history, out
    else:

        @kernel
        def compute(
            x: Tensor(x_shape, "float32"),
            w: Tensor(weight_shape, "bfloat16"),
            history: Tensor(history_shape, "float32"),
            lengths: Tensor((slots,), "int32"),
        ) -> tuple[Tensor(history_shape, "float32"), Tensor(x_shape, "float32")]:
            out = output(x_shape, "float32")
            convolve(x, w, x, history, lengths, out)
            return history, out

    return compute
