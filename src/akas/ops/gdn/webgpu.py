"""WebGPU specializations and shared bindings for gdn."""

from .kernel import (
    conv_commit,
    decode_conv,
    decode_norm,
    decode_prepare,
    decode_recurrent,
    defer_accept,
    defer_clear,
    gdn_deferred,
    gdn_materialize,
    journal_conv,
    journal_restore,
    journal_scan,
    mapped_conv,
    mapped_norm,
    mapped_prepare,
    mapped_scan,
    prefill_conv,
    prefill_scan,
    restore,
)
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "webgpu"))
