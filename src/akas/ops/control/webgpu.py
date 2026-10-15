"""WebGPU specializations and shared bindings for control."""

from .kernel import (
    copy_buffer,
    spec_accept,
    spec_draft,
    spec_proposal,
    spec_repair,
    spec_result,
    spec_setup,
    token_feedback,
)
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))
