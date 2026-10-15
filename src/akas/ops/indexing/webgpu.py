"""WebGPU specializations and shared bindings for indexing."""

from .kernel import clear_rows
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))
