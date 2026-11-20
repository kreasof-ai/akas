"""WebGPU bindings for shared Mamba computations."""

from .kernel import bidirectional_join, reverse, selective_scan
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == "webgpu")
