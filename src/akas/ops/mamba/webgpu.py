"""WebGPU bindings for shared Mamba computations."""

from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "webgpu")
