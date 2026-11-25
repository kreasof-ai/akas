"""FP8 matrix access; storage precision is independent of staging precision."""


def reader(shape, *, role, provider):
    import tilelang.language as T
    from .e4m3 import helpers

    if len(shape) not in (2, 3):
        raise ValueError("FP8 matrix access requires rank two or three")
    _, decode = helpers()
    packed = provider == "webgpu"
    width = shape[-2]

    @T.macro
    def read(encoded, row, column):
        index = column // 4 if packed else column
        if len(shape) == 3:
            bits = T.cast(encoded[0][row // width, row % width, index], "uint32")
        else:
            bits = T.cast(encoded[0][row, index], "uint32")
        decoded_bits = bits >> (column % 4 * 8) & 255 if packed else bits
        scale_row = row if role == "activations" else row // 128
        if len(shape) == 3:
            scale_row = row % width if role == "activations" else row % width // 128
            factor = encoded[1][row // width, scale_row, column // 128]
        else:
            factor = encoded[1][scale_row, column // 128]
        if provider == "cuda":
            return T.call_extern(
                "float32", "tensor_decode_e4m3", decoded_bits
            ) * T.cast(factor, "float32")
        else:
            return decode(decoded_bits) * T.cast(factor, "float32")

    return read


def dequantize(spec, schedule, *, role, provider):
    import tilelang.language as T
    from math import prod
    from akas.kernels.kernel import Tensor, kernel, output
    from akas.quantization import get

    p = {**spec.parameters, **schedule}
    shape = tuple(p.get("shape", (p.get("r", 1), p.get("k", 128))))
    binding = get(role, "fp8").bind(shape, provider=provider)
    values, scale = binding.storage.buffers
    rows, depth = prod(shape[:-1]), shape[-1]
    read = binding.read

    @kernel
    def compute(
        bits: Tensor(values.shape, values.dtype),
        scales: Tensor(scale.shape, scale.dtype),
    ) -> Tensor((rows, depth), "float32"):
        out = output((rows, depth), "float32")
        with T.Kernel(T.ceildiv(rows * depth, 128), threads=128) as block:
            for lane in T.Parallel(128):
                index = block * 128 + lane
                if index < rows * depth:
                    out[index // depth, index % depth] = read(
                        (bits, scales), index // depth, index % depth
                    )
        return out

    return compute
