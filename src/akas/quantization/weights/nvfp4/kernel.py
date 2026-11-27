"""Unpack low-nibble-first E2M1 and apply the two weight scales."""


def bind(shape):
    import tilelang.language as T

    from akas.quantization.formats import Codec
    from akas.quantization.storage.access import bind_reader
    from akas.quantization.storage.e4m3 import helpers
    from akas.quantization.weights.nvfp4.definition import FORMAT

    storage = FORMAT.storage(shape)
    _, rows, _ = shape
    _, decode = helpers()

    @T.macro
    def read(encoded, row, column):
        packed = encoded[0][row // rows, row % rows, column // 2]
        nibble = (packed >> ((column % 2) * 4)) & 15
        magnitude = T.cast(nibble & 7, "float32")
        magnitude = T.if_then_else(
            magnitude < 4,
            magnitude * 0.5,
            T.if_then_else(magnitude < 6, magnitude - 2.0, 2.0 * magnitude - 8.0),
        )
        value = T.if_then_else(nibble < 8, magnitude, -magnitude)
        scale = decode(encoded[1][row // rows, row % rows, column // 16])
        return value * scale * (encoded[2][row // rows] / encoded[3][0])

    return Codec(storage, read, bind_reader(read, shape)[1])


def dequantize(spec, schedule):
    """Decode selected rows of an NVFP4 bank without expanding the whole bank."""
    import tilelang.language as T

    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    shape = (p["experts"], p["rows"], p["depth"])
    experts, rows, depth = shape
    start, count = p["start"], p["count"]
    if not 0 <= start < start + count <= experts * rows:
        raise ValueError("invalid NVFP4 row range")
    read = bind(shape).read

    @kernel
    def compute(
        w: Tensor((experts, rows, depth // 2), "uint8"),
        scale: Tensor((experts, rows, depth // 16), "uint8"),
        amax: Tensor((experts,), "float32"),
        denominator: Tensor((1,), "float32"),
    ) -> Tensor((count, depth), "float32"):
        out = output((count, depth), "float32")
        with T.Kernel(count, T.ceildiv(depth, 128), threads=128) as (row, block):
            for lane in T.Parallel(128):
                col = block * 128 + lane
                if col < depth:
                    out[row, col] = read(
                        (w, scale, amax, denominator), start + row, col
                    )
        return out

    return compute
