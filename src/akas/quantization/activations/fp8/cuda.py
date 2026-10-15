"""CUDA specializations and shared bindings for fp8."""

from .definition import FORMAT, IMPLEMENTATIONS as _CHOICES

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def quantize(spec: KernelSpec, schedule: Mapping[str, object]):
    p = {**spec.parameters, **schedule}
    import tilelang.language as T

    r, k, top = (p["r"], p["k"], p.get("top"))
    rows = r * (top or 1)
    shape = (r, top, k) if top else (r, k)
    storage = FORMAT.storage(shape, provider="cuda")
    scales_shape = storage.named["scales"].shape

    @kernel
    def compute(
        x: Tensor(shape, "bfloat16"),
    ) -> tuple[Tensor(shape, "uint8"), Tensor(scales_shape, "float32")]:
        out = output(shape, "uint8")
        scales = output(scales_shape, "float32")
        with T.Kernel(rows, k // 128, threads=128) as (b, tile):
            values = T.alloc_fragment((128,), "float32")
            absolute = T.alloc_fragment((128,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(128):
                if top:
                    values[j] = T.cast(x[b // top, b % top, tile * 128 + j], "float32")
                else:
                    values[j] = T.cast(x[b, tile * 128 + j], "float32")
                absolute[j] = T.abs(values[j])
            T.reduce_max(absolute, maximum, dim=0)
            for j in T.Parallel(128):
                scale = T.max(maximum[0], T.float32(1e-12)) / T.float32(448)
                bits = T.cast(
                    T.call_extern("uint32", "tensor_encode_e4m3", values[j] / scale),
                    "uint8",
                )
                if top:
                    out[b // top, b % top, tile * 128 + j] = bits
                else:
                    out[b, tile * 128 + j] = bits
            for j in T.Parallel(1):
                scale = T.max(maximum[0], T.float32(1e-12)) / T.float32(448)
                if top:
                    scales[b // top, b % top, tile] = scale
                else:
                    scales[b, tile] = scale
        return (out, scales)

    return compute


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
CAPABILITIES = ("dequantize", "read", "read_tile", "quantize")
IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "cuda")
