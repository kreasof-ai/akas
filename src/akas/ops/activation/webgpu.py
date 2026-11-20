"""WebGPU specializations and shared bindings for activation."""

from .kernel import column_activation, swiglu, swiglu_experts, sigmoid_bias
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))
