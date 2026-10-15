"""WebGPU specializations and shared bindings for residual."""

from .kernel import add
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))
