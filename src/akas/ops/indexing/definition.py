"""Contracts and implementation choices for indexing."""

from akas.kernels.contracts import Implementation
from akas.ops.indexing.validation import validation

OUTPUT_ARGUMENTS = {}
VALIDATIONS = {"clear_rows": validation}
IMPLEMENTATIONS = (
    Implementation(
        "clear_rows",
        "akas.ops.indexing.kernel",
        "state",
        space=(("threads", (64, 128, 256)),),
        factory="clear_rows",
        name="indexing.rows",
    ),
    Implementation(
        "clear_rows",
        "akas.ops.indexing.kernel",
        "state",
        provider="webgpu",
        space=(("threads", (64, 128, 256)),),
        factory="clear_rows",
        name="indexing.rows",
    ),
)
