"""WebGPU specializations and shared bindings for indexing."""

from .kernel import clear_rows, select_token, concatenate, replace_token
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))
