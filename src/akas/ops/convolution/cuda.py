"""CUDA specializations and shared bindings for convolution."""

from .kernel import convolution, convolution_flat
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "cuda"))
