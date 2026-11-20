"""Shared fixed-size STFT, iSTFT and band-layout computations."""

from akas.kernels.kernel import Tensor, kernel, output


def stft(spec, schedule):
    """Apply centered real FFT analysis with periodic window coefficients."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    algorithm = p.get("algorithm", "direct")
    if algorithm != "direct":
        raise ValueError("unknown spectral transform schedule")
    samples, size, hop, frames, bins = (
        p[key] for key in ("samples", "fft", "hop", "frames", "bins")
    )
    center, reflect = size // 2, p.get("reflect", True)

    @kernel
    def compute(
        audio: Tensor((samples,), "float32"), window: Tensor((size,), "float32")
    ) -> Tensor((frames, bins, 2), "float32"):
        out = output((frames, bins, 2), "float32")
        with T.Kernel(T.ceildiv(frames * bins, 128), threads=128) as tile:
            tx = T.get_thread_binding()
            index = tile * 128 + tx
            if index < frames * bins:
                frame, frequency = index // bins, index % bins
                real = T.alloc_var("float32")
                imag = T.alloc_var("float32")
                real, imag = 0, 0
                for tap in T.serial(size):
                    phase = (
                        -T.float32(6.28318530717958647692)
                        * (frequency * tap % size)
                        / size
                    )
                    cosine = T.cos(phase)
                    sine = T.sin(phase)
                    source = frame * hop + tap - center
                    reflected = T.if_then_else(
                        source < 0,
                        -source,
                        T.if_then_else(
                            source >= samples, 2 * samples - 2 - source, source
                        ),
                    )
                    position = T.max(
                        0, T.min(samples - 1, reflected if reflect else source)
                    )
                    value = T.if_then_else(
                        reflect or (0 <= source and source < samples),
                        audio[position] * window[tap],
                        0,
                    )
                    real += value * cosine
                    imag += value * sine
                out[frame, frequency, 0] = real
                out[frame, frequency, 1] = imag
        return out

    return compute


def istft_matrix(spec, schedule):
    """Apply a precomputed real inverse-DFT matrix to every spectral frame."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    frames, bins, size = (p[key] for key in ("frames", "bins", "fft"))

    @kernel
    def compute(
        spectrum: Tensor((frames, bins, 2), "float32"),
        matrix: Tensor((size, 2 * bins), "float32"),
    ) -> Tensor((frames, size), "float32"):
        out = output((frames, size), "float32")
        with T.Kernel(frames, T.ceildiv(size, 64), threads=64) as (frame, block):
            tap = block * 64 + T.get_thread_binding()
            value = T.alloc_var("float32")
            value = 0
            for frequency in T.serial(bins):
                value += (
                    spectrum[frame, frequency, 0] * matrix[tap, 2 * frequency]
                    + spectrum[frame, frequency, 1] * matrix[tap, 2 * frequency + 1]
                )
            out[frame, tap] = value
        return out

    return compute


def istft_overlap(spec, schedule):
    """Overlap-add inverse frames, normalize window envelopes, and add input."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    samples, size, hop, frames = (p[key] for key in ("samples", "fft", "hop", "frames"))
    center = size // 2

    @kernel
    def compute(
        frames_input: Tensor((frames, size), "float32"),
        original: Tensor((samples,), "float32"),
        window: Tensor((size,), "float32"),
    ) -> Tensor((samples,), "float32"):
        out = output((samples,), "float32")
        with T.Kernel(T.ceildiv(samples, 128), threads=128) as tile:
            index = tile * 128 + T.get_thread_binding()
            if index < samples:
                total = T.alloc_var("float32")
                normalization = T.alloc_var("float32")
                total, normalization = 0, 0
                first = T.max(0, (index + center - size + hop) // hop)
                last = T.min(frames, (index + center) // hop + 1)
                for frame in T.serial(first, last):
                    tap = index + center - frame * hop
                    coefficient = window[tap]
                    total += frames_input[frame, tap] * coefficient
                    normalization += coefficient * coefficient
                out[index] = original[index] + total / T.max(normalization, 1e-8)
        return out

    return compute


def complex_correction(spec, schedule):
    """Apply a bounded complex mask delta and bounded per-bin residual."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    frames, bins, bound = (p[key] for key in ("frames", "bins", "bound"))

    @kernel
    def compute(
        raw: Tensor((frames, bins, 4), "float32"),
        spectrum: Tensor((frames, bins, 2), "float32"),
    ) -> Tensor((frames, bins, 2), "float32"):
        out = output((frames, bins, 2), "float32")
        with T.Kernel(T.ceildiv(frames * bins, 128), threads=128) as tile:
            index = tile * 128 + T.get_thread_binding()
            if index < frames * bins:
                frame, frequency = index // bins, index % bins
                mr = 2 * T.tanh(raw[frame, frequency, 0])
                mi = 2 * T.tanh(raw[frame, frequency, 1])
                rr = T.tanh(raw[frame, frequency, 2])
                ri = T.tanh(raw[frame, frequency, 3])
                norm = T.max(T.sqrt(rr * rr + ri * ri), 1)
                sr, si = spectrum[frame, frequency, 0], spectrum[frame, frequency, 1]
                magnitude = bound * T.sqrt(sr * sr + si * si)
                rr, ri = rr / norm * magnitude, ri / norm * magnitude
                out[frame, frequency, 0] = sr * mr - si * mi + rr
                out[frame, frequency, 1] = sr * mi + si * mr + ri
        return out

    return compute


def exclusive_attention(spec, schedule):
    """Remove each attention result's component along its own value vector."""
    import tilelang.language as T
    from akas.kernels.reductions import tree_reduce

    p = {**spec.parameters, **schedule}
    rows, heads, width = (p[key] for key in ("r", "h", "d"))
    reduce = tree_reduce(width)

    @kernel
    def compute(
        attended: Tensor((rows, heads, width), "float32"),
        values: Tensor((rows, heads, width), "float32"),
    ) -> Tensor((rows, heads, width), "float32"):
        out = output((rows, heads, width), "float32")
        with T.Kernel(rows * heads, threads=width) as index:
            tx = T.get_thread_binding()
            numerator = T.alloc_shared((width,), "float32")
            denominator = T.alloc_shared((width,), "float32")
            value = values[index // heads, index % heads, tx]
            numerator[tx] = attended[index // heads, index % heads, tx] * value
            denominator[tx] = value * value
            T.sync_threads()
            reduce(numerator, tx)
            reduce(denominator, tx)
            coefficient = numerator[0] / T.max(denominator[0], 1e-6)
            out[index // heads, index % heads, tx] = (
                attended[index // heads, index % heads, tx] - coefficient * value
            )
        return out

    return compute


def token_transpose(spec, schedule):
    """Exchange the band and frame axes of a width-preserving token tensor."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    bands, frames, width = (p[key] for key in ("bands", "frames", "width"))

    @kernel
    def compute(
        x: Tensor((bands, frames, width), "float32"),
    ) -> Tensor((frames, bands, width), "float32"):
        out = output((frames, bands, width), "float32")
        with T.Kernel(T.ceildiv(bands * frames * width, 256), threads=256) as tile:
            index = tile * 256 + T.get_thread_binding()
            if index < bands * frames * width:
                band = index // (frames * width)
                offset = index % (frames * width)
                frame, channel = offset // width, offset % width
                out[frame, band, channel] = x[band, frame, channel]
        return out

    return compute


def band_gather(spec, schedule):
    """Extract one band's frame-major tokens without changing their width."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    bands, frames, width, selected = (
        p[key] for key in ("bands", "frames", "width", "band")
    )

    @kernel
    def compute(
        x: Tensor((bands, frames, width), "float32"),
    ) -> Tensor((frames, width), "float32"):
        out = output((frames, width), "float32")
        with T.Kernel(frames, T.ceildiv(width, 128), threads=128) as (frame, tile):
            channel = tile * 128 + T.get_thread_binding()
            if channel < width:
                out[frame, channel] = x[selected, frame, channel]
        return out

    return compute


def band_scatter(spec, schedule):
    """Place a frame-major band head result into the linear-frequency layout."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    frames, bins, start, end = (p[key] for key in ("frames", "bins", "start", "end"))
    width = end - start

    @kernel
    def compute(
        x: Tensor((frames, width, 4), "float32"),
        spectrum: Tensor((frames, bins, 4), "float32"),
    ) -> Tensor((frames, bins, 4), "float32"):
        with T.Kernel(T.ceildiv(frames * width * 4, 128), threads=128) as tile:
            index = tile * 128 + T.get_thread_binding()
            if index < frames * width * 4:
                frame = index // (width * 4)
                offset = index % (width * 4)
                frequency, component = offset // 4, offset % 4
                spectrum[frame, start + frequency, component] = x[
                    frame, frequency, component
                ]
        return spectrum

    return compute


def spectrum_band(spec, schedule):
    """Flatten one FFT bin interval into interleaved real/imag components."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    frames, bins, start, end = (p[key] for key in ("frames", "bins", "start", "end"))
    width = end - start

    @kernel
    def compute(
        x: Tensor((frames, bins, 2), "float32"),
    ) -> Tensor((frames, 2 * width), "float32"):
        out = output((frames, 2 * width), "float32")
        with T.Kernel(frames, T.ceildiv(2 * width, 128), threads=128) as (frame, tile):
            index = tile * 128 + T.get_thread_binding()
            if index < 2 * width:
                frequency, component = index // 2, index % 2
                out[frame, index] = x[frame, start + frequency, component]
        return out

    return compute


def token_scatter(spec, schedule):
    """Store one projected band and add its learned embedding during the write."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    bands, frames, width, selected = (
        p[key] for key in ("bands", "frames", "width", "band")
    )

    @kernel
    def compute(
        x: Tensor((frames, width), "float32"),
        embedding: Tensor((width,), "float32"),
        features: Tensor((bands, frames, width), "float32"),
    ) -> Tensor((bands, frames, width), "float32"):
        with T.Kernel(frames, T.ceildiv(width, 128), threads=128) as (frame, tile):
            channel = tile * 128 + T.get_thread_binding()
            if channel < width:
                features[selected, frame, channel] = (
                    x[frame, channel] + embedding[channel]
                )
        return features

    return compute


def channel_slice(spec, schedule):
    """Extract a contiguous feature interval without an implicit host copy."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows, channels, start, width = (p[key] for key in ("r", "c", "start", "width"))

    @kernel
    def compute(x: Tensor((rows, channels), "float32")) -> Tensor(
        (rows, width), "float32"
    ):
        out = output((rows, width), "float32")
        with T.Kernel(rows, T.ceildiv(width, 128), threads=128) as (row, tile):
            channel = tile * 128 + T.get_thread_binding()
            if channel < width:
                out[row, channel] = x[row, start + channel]
        return out

    return compute


def channel_affine(spec, schedule):
    """Apply one featurewise affine transform with a unit-plus scale."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows, channels = (p[key] for key in ("r", "c"))

    @kernel
    def compute(
        x: Tensor((rows, channels), "float32"),
        scale: Tensor((channels,), "float32"),
        shift: Tensor((channels,), "float32"),
    ) -> Tensor((rows, channels), "float32"):
        out = output((rows, channels), "float32")
        with T.Kernel(T.ceildiv(rows * channels, 256), threads=256) as tile:
            index = tile * 256 + T.get_thread_binding()
            if index < rows * channels:
                channel = index % channels
                out[index // channels, channel] = (
                    x[index // channels, channel] * (1 + scale[channel])
                    + shift[channel]
                )
        return out

    return compute


def elementwise_binary(spec, schedule):
    """Multiply, divide or subtract matching tensors at a fixed layout."""
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    count, operation = p["n"], p["operation"]

    @kernel
    def compute(
        x: Tensor((count,), "float32"), y: Tensor((count,), "float32")
    ) -> Tensor((count,), "float32"):
        out = output((count,), "float32")
        with T.Kernel(T.ceildiv(count, 256), threads=256) as tile:
            index = tile * 256 + T.get_thread_binding()
            if index < count:
                if operation == "multiply":
                    out[index] = x[index] * y[index]
                elif operation == "divide":
                    out[index] = x[index] / T.max(y[index], 1e-6)
                else:
                    out[index] = x[index] - y[index]
        return out

    return compute
