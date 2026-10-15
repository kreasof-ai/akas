"""WebGPU specializations and shared bindings for mtp."""

from .kernel import mtp_cast, mtp_join
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))
