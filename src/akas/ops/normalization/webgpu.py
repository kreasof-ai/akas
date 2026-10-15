"""WebGPU specializations and shared bindings for normalization."""

from .kernel import add_rms, kvnorm, qnorm, rms
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))
