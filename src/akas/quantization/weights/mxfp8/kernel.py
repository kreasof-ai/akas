"""Reconstruct E4M3 coefficients using a power-of-two microscale."""


def bind(shape):
    import tilelang.language as T

    from akas.quantization.formats import Codec
    from akas.quantization.storage.access import bind_reader
    from akas.quantization.storage.e4m3 import helpers
    from akas.quantization.weights.mxfp8.definition import FORMAT

    storage = FORMAT.storage(shape)
    _, rows, _ = shape
    _, decode = helpers()

    @T.macro
    def read(encoded, row, column):
        exponent = T.cast(encoded[1][row // rows, row % rows, column // 32], "int32")
        weight = decode(encoded[0][row // rows, row % rows, column])
        return weight * T.exp2(T.cast(exponent - 127, "float32"))

    return Codec(storage, read, bind_reader(read, shape)[1])


def dequantize(spec, schedule):
    """Decode selected rows of an MXFP8 bank without expanding the whole bank."""
    import tilelang.language as T

    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    shape = (p["experts"], p["rows"], p["depth"])
    experts, rows, depth = shape
    start, count = p["start"], p["count"]
    if not 0 <= start < start + count <= experts * rows:
        raise ValueError("invalid MXFP8 row range")
    read = bind(shape).read

    @kernel
    def compute(
        w: Tensor(shape, "uint8"),
        scale: Tensor((experts, rows, depth // 32), "uint8"),
    ) -> Tensor((count, depth), "float32"):
        out = output((count, depth), "float32")
        with T.Kernel(count, T.ceildiv(depth, 128), threads=128) as (row, block):
            for lane in T.Parallel(128):
                col = block * 128 + lane
                if col < depth:
                    out[row, col] = read((w, scale), start + row, col)
        return out

    return compute
