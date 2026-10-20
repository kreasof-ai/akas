"""WebGPU specializations and shared bindings for vocabulary."""


def quantized_embedding(spec, schedule):
    from .kernel import quantized_embedding

    return quantized_embedding(spec, schedule, provider="webgpu")


from .kernel import (
    argmax,
    argmax_packed,
    embedding,
    embedding_packed,
    native_argmax,
    native_embedding,
    packed_word_argmax,
    packed_word_embedding,
)
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))


from .kernel import argmax_partial as argmax_partial, argmax_merge as argmax_merge
