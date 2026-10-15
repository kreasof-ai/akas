"""WebGPU specializations and shared bindings for fp8."""

from .definition import FORMAT, IMPLEMENTATIONS as _CHOICES

from functools import partial
from akas.quantization.kv.fp8.kernel import reference_decoder as _decoder
from akas.quantization.kv.fp8.kernel import reference_writer as _writer

reference_writer = partial(_writer, provider="webgpu")
reference_decoder = partial(_decoder, provider="webgpu")


def bind(shape, *, chunk=1, value_offset=0):
    from .kernel import binding

    return binding(shape, provider="webgpu", chunk=chunk, value_offset=value_offset)


def quantize(spec, schedule):
    from .kernel import quantize

    return quantize(spec, schedule)


def dequantize(spec, schedule):
    from .kernel import dequantize

    return dequantize(spec, schedule)


AVAILABLE = "webgpu" in dict(FORMAT.providers)
CAPABILITIES = ("read", "read_tile", "write", "quantize", "dequantize")
IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "webgpu")
