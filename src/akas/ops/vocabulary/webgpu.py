"""WebGPU specializations and shared bindings for vocabulary."""


def quantized_embedding(spec, schedule):
    from .kernel import quantized_embedding

    return quantized_embedding(spec, schedule, provider="webgpu")


from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))
