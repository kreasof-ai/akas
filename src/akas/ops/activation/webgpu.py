"""WebGPU specializations and shared bindings for activation."""

from .kernel import swiglu, swiglu_experts
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))
