"""CUDA specializations and shared bindings for sequence."""

from .kernel import (
    block_limits,
    advance,
    baseline_advance,
    baseline_last,
    chunk_advance,
    chunk_controls,
    chunk_last,
    chunk_last_rows,
    chunk_tail,
    controls_chunk_controls,
    flat_prefill_tail,
    mapped_controls,
    mapped_gather_hidden,
    mapped_last_rows,
    mapped_store_hidden,
    masked_copy,
    next_tokens,
    prefill_tail,
    tail_controls,
)
from .definition import IMPLEMENTATIONS as _CHOICES

IMPLEMENTATIONS = tuple((choice for choice in _CHOICES if choice.provider == "cuda"))
