"""WebGPU FP8 weights: packed storage with fused software decoding."""

from .definition import FORMAT, IMPLEMENTATIONS as _CHOICES


def dequantize(spec, schedule):
    from .kernel import dequantize

    return dequantize(spec, schedule, provider="webgpu")


def bind(shape):
    from akas.quantization.formats import Codec
    from akas.quantization._shared.access import matrix_tile
    from .kernel import reader

    storage = FORMAT.storage(shape, provider="webgpu")
    read = reader(shape, provider="webgpu")
    return Codec(storage, read, matrix_tile(read, shape))


AVAILABLE = "webgpu" in dict(FORMAT.providers)
CAPABILITIES = ("dequantize", "read", "read_tile")
IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "webgpu")
