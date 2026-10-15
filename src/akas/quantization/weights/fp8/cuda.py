"""CUDA specializations and shared bindings for fp8."""

from .definition import FORMAT, IMPLEMENTATIONS as _CHOICES

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def decode(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    @kernel
    def compute(bits: Tensor((256,), "uint8")) -> Tensor((256,), "float32"):
        out = output((256,), "float32")
        with T.Kernel(1, threads=256):
            for i in T.Parallel(256):
                out[i] = T.call_extern(
                    "float32", "tensor_decode_e4m3", T.cast(bits[i], "uint32")
                )
        return out

    return compute
    raise ValueError("unsupported operation: e4m3_decode")


def widen(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    shape = tuple(p["shape"])
    if len(shape) != 3:
        raise ValueError("expert operand cache needs [experts,output,input]")
    experts, rows, k = shape
    count = rows * k
    if k % 32:
        raise ValueError("operand pairing requires K32")

    @kernel
    def compute(source: Tensor(shape, "uint8")) -> Tensor(shape, "float16"):
        out = output(shape, "float16")
        with T.Kernel(experts, T.ceildiv(count, 1024), threads=256) as (expert, block):
            for index in T.Parallel(1024):
                pos = block * 1024 + index
                if pos < count:
                    row = pos // k
                    j = pos % 32
                    original = (
                        pos % k // 32 * 32 + j % 16 // 2 * 4 + j % 2 + j // 16 * 2
                    )
                    out[expert, row, pos % k] = T.reinterpret(
                        "float8_e4m3fn", source[expert, row, original]
                    )
        return out

    return compute


CUDA_SOURCE = '\n__device__ __forceinline__ unsigned int tensor_widen_e4m3x2(unsigned short bits) {\n    unsigned int packed;\n    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(packed) : "h"(bits));\n    return packed;\n}\n'


def dequantize(spec, schedule):
    from .kernel import dequantize

    return dequantize(spec, schedule, provider="cuda")


def bind(shape):
    from akas.quantization.formats import Codec
    from akas.quantization._shared.access import matrix_tile
    from .kernel import reader

    storage = FORMAT.storage(shape, provider="cuda")
    read = reader(shape, provider="cuda")
    return Codec(storage, read, matrix_tile(read, shape))


AVAILABLE = "cuda" in dict(FORMAT.providers)
CAPABILITIES = ("dequantize", "read", "read_tile")
IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "cuda")
