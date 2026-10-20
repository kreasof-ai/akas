"""Bounded host readers for independent encoded-weight operation validation."""

from importlib import import_module
from akas.quantization.formats import get


def sample_parts(parts, depth, storage, *, seed=31):
    """Sample a single embedded piece and return its host row reader.

    A format supplies ``validation.reference(options, shape, storage, seed=...)``.
    Sampling and decoding stay with the codec; callers own the operation math.
    """
    if len(parts) != 1 or parts[0].get("row_order") or parts[0].get("column_order"):
        raise ValueError("host reference requires one unpermuted encoded piece")
    part = parts[0]
    get("weights", part["format"])
    module = import_module(f"akas.quantization.weights.{part['format']}.validation")
    if not hasattr(module, "reference"):
        raise ValueError(f"weight format {part['format']} has no host reference")
    return module.reference(part["options"], (part["rows"], depth), storage, seed=seed)
