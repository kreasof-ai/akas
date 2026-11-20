"""CUDA bindings reserved for qualified spectral specializations."""

from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "cuda")
