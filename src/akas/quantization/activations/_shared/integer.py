"""One/two signed-byte planes with per-block scales and correction sums."""


def reader(shape, *, components):
    import tilelang.language as T
    from math import prod

    depth, count = shape[-1], prod(shape)
    words, blocks = count // 4, count // 32

    @T.macro
    def component(encoded, row, column, plane):
        index = row * depth + column
        bits = encoded[0][plane * words + index // 4] >> (index % 4 * 8) & 255
        signed = T.cast(bits, "int32") - T.if_then_else(
            T.cast(bits, "int32") >= 128, 256, 0
        )
        return T.cast(signed, "float32") * encoded[1][plane * blocks + index // 32]

    @T.macro
    def read(encoded, row, column):
        if components == 1:
            return component(encoded, row, column, 0)
        else:
            return component(encoded, row, column, 0) + component(
                encoded, row, column, 1
            )

    return read


def quantize(spec, schedule, *, name, provider):
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.quantization import get
    from .packing import packing

    p = {**spec.parameters, **schedule}
    rows, depth = p.get("r", 1), p["k"]
    storage = get("activations", name).storage((rows, depth), provider=provider)
    scale, total = storage.metadata
    encode = packing(p, components=1 if name == "q8" else 2, provider=provider)

    if name == "q8":

        @kernel
        def compute(
            x: Tensor((rows * depth,), "float32"),
        ) -> tuple[
            Tensor(storage.values.shape, storage.values.dtype),
            Tensor(scale.shape, scale.dtype),
            Tensor(total.shape, total.dtype),
        ]:
            out = output(storage.values.shape, storage.values.dtype)
            scales = output(scale.shape, scale.dtype)
            sums = output(total.shape, total.dtype)
            encode(x, out, scales, sums)
            return out, scales, sums

        return compute

    @kernel
    def compute(
        x: Tensor((rows * depth,), "float32"),
    ) -> tuple[
        Tensor(storage.values.shape, storage.values.dtype),
        Tensor(scale.shape, scale.dtype),
        Tensor(total.shape, total.dtype),
    ]:
        packed = output(storage.values.shape, storage.values.dtype)
        scales = output(scale.shape, scale.dtype)
        sums = output(total.shape, total.dtype)
        encode(x, packed, scales, sums)
        return packed, scales, sums

    return compute


def dequantize(spec, schedule, *, name, provider):
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.quantization import get

    p = {**spec.parameters, **schedule}
    rows, depth = p.get("r", 1), p["k"]
    binding = get("activations", name).bind((rows, depth), provider=provider)
    values, scale = binding.read_buffers
    read = binding.read

    @kernel
    def compute(
        packed: Tensor(values.shape, values.dtype),
        scales: Tensor(scale.shape, scale.dtype),
    ) -> Tensor((rows, depth), "float32"):
        out = output((rows, depth), "float32")
        with T.Kernel(T.ceildiv(rows * depth, 128), threads=128) as block:
            for lane in T.Parallel(128):
                index = block * 128 + lane
                if index < rows * depth:
                    out[index // depth, index % depth] = read(
                        (packed, scales), index // depth, index % depth
                    )
        return out

    return compute
