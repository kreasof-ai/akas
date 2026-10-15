"""Shared callable Q16 conversions and inline matrix access."""


def reader(shape):
    from akas.quantization.activations._shared.integer import reader

    return reader(shape, components=2)


def quantize(spec, schedule, *, provider):
    from akas.quantization.activations._shared.integer import quantize

    return quantize(spec, schedule, name="q16", provider=provider)


def dequantize(spec, schedule, *, provider):
    from akas.quantization.activations._shared.integer import dequantize

    return dequantize(spec, schedule, name="q16", provider=provider)
