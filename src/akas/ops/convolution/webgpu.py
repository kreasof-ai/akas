"""WebGPU specializations and shared bindings for convolution."""

from .definition import IMPLEMENTATIONS as _CHOICES
from akas.kernels.kernel import Tensor, kernel, output

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))


def implicit_half_conv2d(spec, schedule):
    """Read padded convolution taps directly into the half GEMM workgroup tile."""
    import tilelang.language as T
    from akas.ops.linear.webgpu import HALF_TILE_LAYOUTS, native_half_matmul

    p = spec.parameters
    width, height, channels, columns = (
        p[k] for k in ("width", "height", "channels", "columns")
    )
    size, stride, pad = (p[k] for k in ("kernel", "stride", "padding"))
    ow, oh = ((n + 2 * pad - size) // stride + 1 for n in (width, height))
    rows, depth = ow * oh, size * size * channels
    if min(ow, oh) < 1 or depth % 16:
        raise ValueError(
            "implicit half convolution requires a nonempty output and 16-wide reduction groups"
        )

    parameters = dict(
        r=rows,
        k=depth,
        o=columns,
        type=1,
        half_tile=HALF_TILE_LAYOUTS[schedule.get("layout", 0)],
        half_pad=1,
        half_k=schedule.get("half_k", 16),
    )
    m, _, _, _, threads = parameters["half_tile"]
    bk = parameters["half_k"]
    if channels % bk:
        raise ValueError("each input tile must fit within one convolution tap")
    pairs, blocks = bk // 2, channels // bk

    @T.macro
    def stage_image(image, lhs, by, tile, tx):
        tap = tile // blocks
        for load in T.serial(T.ceildiv(m * pairs, threads)):
            flat = load * threads + tx
            row, pair = flat // pairs, flat % pairs
            if row < m:
                site = by * m + row
                x = site // oh * stride + tap % size - pad
                y = site % oh * stride + tap // size - pad
                channel = tile % blocks * bk + pair * 2
                if site < rows and 0 <= x and x < width and 0 <= y and y < height:
                    for j in T.unroll(2):
                        lhs[pair, row, j] = image[x, y, channel + j]
                else:
                    for j in T.unroll(2):
                        lhs[pair, row, j] = 0

    matmul = native_half_matmul(parameters, stage_input=stage_image)

    @kernel
    def compute(
        image: Tensor((width, height, channels), "float16"),
        weights: Tensor((columns * depth,), "float16"),
    ) -> Tensor((rows * columns,), "float32"):
        out = output((rows * columns,), "float32")
        matmul(image, weights, out)
        return out

    return compute


def image_patches_half(spec, schedule):
    """Gather contiguous channel groups and round each patch element once."""
    import tilelang.language as T
    from akas.quantization.storage.rounding import round_values

    p = spec.parameters
    width, height, channels = (p[k] for k in ("width", "height", "channels"))
    size, stride, pad = (p[k] for k in ("kernel", "stride", "padding"))
    ow, oh = ((n + 2 * pad - size) // stride + 1 for n in (width, height))
    groups = channels // 4
    depth = size * size * channels
    rounded = round_values("float16")

    @kernel
    def compute(image: Tensor((width, height, groups, 4), "float32")) -> Tensor(
        (ow * oh, depth), "float16"
    ):
        out = output((ow * oh, depth), "float16")
        with T.Kernel(T.ceildiv(ow * oh * depth // 4, 128), threads=128) as tile:
            for lane in T.Parallel(128):
                i = tile * 128 + lane
                if i < ow * oh * depth // 4:
                    row, col = i // (depth // 4), i % (depth // 4)
                    tap, channel = col // groups, col % groups
                    x = row // oh * stride + tap % size - pad
                    y = row % oh * stride + tap // size - pad
                    if 0 <= x and x < width and 0 <= y and y < height:
                        for j in T.unroll(4):
                            out[row, col * 4 + j] = rounded(image[x, y, channel, j])
                    else:
                        for j in T.unroll(4):
                            out[row, col * 4 + j] = 0
        return out

    return compute


def single_channel_conv2d(spec, schedule):
    """Share a short single-channel image patch across all output filters."""
    import tilelang.language as T
    from akas.quantization.storage.rounding import round_values

    p = {**spec.parameters, **schedule}
    width, height, columns = (p[k] for k in ("width", "height", "columns"))
    size, stride, pad = (p[k] for k in ("kernel", "stride", "padding"))
    ow, oh = ((n + 2 * pad - size) // stride + 1 for n in (width, height))
    taps = size * size
    sites = p.get("sites", 4)
    if sites <= 0 or sites * taps > 128:
        raise ValueError("single-channel convolution patch tile exceeds 128 lanes")
    rounded = round_values("float16")

    @kernel
    def compute(
        image: Tensor((width, height, 1), "float32"),
        weights: Tensor((columns * taps,), "float16"),
    ) -> Tensor((ow * oh, columns), "float32"):
        out = output((ow * oh, columns), "float32")
        with T.Kernel(
            T.ceildiv(ow * oh, sites), T.ceildiv(columns, 128), threads=128
        ) as (
            strip,
            tile,
        ):
            samples = T.alloc_shared((sites, taps), "float32")
            tx = T.get_thread_binding()
            if tx < sites * taps:
                site, tap = tx // taps, tx % taps
                row = strip * sites + site
                x = row // oh * stride + tap % size - pad
                y = row % oh * stride + tap // size - pad
                samples[site, tap] = 0
                if row < ow * oh and 0 <= x and x < width and 0 <= y and y < height:
                    samples[site, tap] = rounded(image[x, y, 0])
            T.sync_threads()
            column = tile * 128 + tx
            if column < columns:
                filter_values = T.alloc_local((taps,), "float32")
                for i in T.unroll(taps):
                    filter_values[i] = T.cast(weights[column * taps + i], "float32")
                for row_offset in T.unroll(sites):
                    row_id = strip * sites + row_offset
                    if row_id < ow * oh:
                        value = T.alloc_var("float32")
                        value = 0
                        for i in T.unroll(taps):
                            value += samples[row_offset, i] * filter_values[i]
                        out[row_id, column] = value
        return out

    return compute
