"""Shared FP8 activations access and standalone dequantization."""


def reader(shape, *, provider):
    from akas.quantization._shared.fp8 import reader

    return reader(shape, role="activations", provider=provider)


def dequantize(spec, schedule, *, provider):
    from akas.quantization._shared.fp8 import dequantize

    return dequantize(spec, schedule, role="activations", provider=provider)
