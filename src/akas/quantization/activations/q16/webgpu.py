"""WebGPU specializations and shared bindings for q16."""

from .definition import FORMAT, IMPLEMENTATIONS as _CHOICES

from collections.abc import Mapping
from akas.kernels.definition import KernelSpec
from akas.kernels.kernel import Tensor, kernel, output


def quantize(spec: KernelSpec, schedule: Mapping[str, object]):
    from akas.quantization.activations._shared.packing import packing

    p = {**spec.parameters, **schedule}
    rows = p.get("r", 1)
    depth = p["k"]
    storage = FORMAT.storage((rows, depth), provider="webgpu")
    scale, total = storage.metadata
    encode = packing(p, components=2)

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
        return (packed, scales, sums)

    return compute


def dequantize(spec, schedule):
    from .kernel import dequantize

    return dequantize(spec, schedule, provider="webgpu")


def bind(shape):
    from akas.quantization.formats import Codec
    from akas.quantization._shared.access import matrix_tile
    from .kernel import reader

    storage = FORMAT.storage(shape, provider="webgpu")
    read = reader(shape)
    return Codec(
        storage, read, matrix_tile(read, shape), read_fields=("values", "scales")
    )


AVAILABLE = "webgpu" in dict(FORMAT.providers)
CAPABILITIES = ("quantize", "dequantize", "read", "read_tile")
IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "webgpu")
