"""CUDA Q8 compatibility conversions; reads fuse into consumers."""

from .definition import FORMAT, IMPLEMENTATIONS as _CHOICES


def quantize(spec, schedule):
    import tilelang.language as T
    from akas.kernels.kernel import Tensor, kernel, output

    p = {**spec.parameters, **schedule}
    rows, depth = p.get("r", 1), p["k"]
    dtype, threads = p.get("input_dtype", "float32"), p.get("threads", 256)
    if depth % 32 or dtype not in ("float16", "float32") or threads not in (128, 256):
        raise ValueError("invalid CUDA Q8 packing geometry or precision")
    blocks, warps = rows * depth // 32, threads // 32

    @kernel
    def compute(
        x: Tensor(rows * depth, dtype),
    ) -> tuple[
        Tensor(rows * depth // 4, "uint32"),
        Tensor(blocks, "float32"),
        Tensor(blocks, "int32"),
    ]:
        out = output(rows * depth // 4, "uint32")
        scales = output(blocks, "float32")
        sums = output(blocks, "int32")
        with T.Kernel(T.ceildiv(blocks, warps), threads=threads) as bx:
            tx = T.get_thread_binding()
            lane, block = tx % 32, bx * warps + tx // 32
            if block < blocks:
                value = T.cast(x[block * 32 + lane], "float32")
                maximum = T.alloc_var("float32")
                maximum = T.abs(value)
                for step in T.unroll(5):
                    maximum = T.max(maximum, T.shfl_xor(maximum, 1 << step))
                scale = T.max(maximum / T.float32(127), T.float32(1e-20))
                quant = T.cast(T.round(value / scale), "int32")
                bits = T.alloc_var("uint32")
                bits = 0
                for part in T.unroll(4):
                    bits |= T.cast(
                        T.shfl_sync(quant, lane // 4 * 4 + part) & 255, "uint32"
                    ) << (part * 8)
                if lane % 4 == 0:
                    out[block * 8 + lane // 4] = bits
                total = T.alloc_var("int32")
                total = quant
                for step in T.unroll(5):
                    total += T.shfl_xor(total, 1 << step)
                if lane == 0:
                    scales[block] = scale
                    sums[block] = total
        return out, scales, sums

    return compute


def dequantize(spec, schedule):
    from .kernel import dequantize

    return dequantize(spec, schedule, provider="cuda")


def bind(shape):
    from akas.quantization.formats import Codec
    from akas.quantization._shared.access import matrix_tile
    from .kernel import reader

    storage = FORMAT.storage(shape, provider="cuda")
    read = reader(shape)
    return Codec(
        storage, read, matrix_tile(read, shape), read_fields=("values", "scales")
    )


AVAILABLE = "cuda" in dict(FORMAT.providers)
CAPABILITIES = ("quantize", "dequantize", "read", "read_tile")
IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "cuda")
