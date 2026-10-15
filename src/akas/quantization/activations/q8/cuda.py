"""CUDA Q8 compatibility conversions; reads fuse into consumers."""

from .definition import FORMAT, IMPLEMENTATIONS as _CHOICES


def quantize(spec, schedule):
    from .kernel import quantize

    return quantize(spec, schedule, provider="cuda")


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
