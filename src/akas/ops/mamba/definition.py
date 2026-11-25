"""Contracts for Mamba preparation, recurrence and bidirectional joins."""

from akas.kernels.spec import Implementation, OutputABI
from akas.ops.mamba.validation import VALIDATIONS as VALIDATIONS


OUTPUT_ARGUMENTS = {
    "reference": {
        "selective_scan": OutputABI((-1,), ("out",)),
        "mamba_reverse": OutputABI((-1,), ("out",)),
        "mamba_join": OutputABI((-1,), ("out",)),
    }
}
IMPLEMENTATIONS = tuple(
    Implementation(
        kind,
        "akas.ops.mamba.kernel",
        "reference",
        provider="webgpu",
        factory=factory,
        name=f"mamba.webgpu.{factory}",
        space=space,
        semantic_keys=("sequences", "frames", "c", "s")
        if kind == "selective_scan"
        else None,
    )
    for kind, factory, space in (
        ("selective_scan", "selective_scan", (("rows_per_block", (1, 4, 8)),)),
        ("mamba_reverse", "reverse", ()),
        ("mamba_join", "bidirectional_join", ()),
    )
)
