"""Contracts and implementation choices for indexing."""

from akas.kernels.contracts import Implementation
from akas.ops.indexing.validation import validation, feature_validation

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


OUTPUT_ARGUMENTS["reference"] = {
    kind: (-1,) for kind in ("select_token", "concatenate", "replace_token")
}
VALIDATIONS.update(
    {
        kind: feature_validation
        for kind in ("select_token", "concatenate", "replace_token")
    }
)
IMPLEMENTATIONS += tuple(
    Implementation(
        kind,
        "akas.ops.indexing.kernel",
        "reference",
        provider=provider,
        factory=kind,
        name="indexing." + kind,
    )
    for provider in ("cuda", "webgpu")
    for kind in ("select_token", "concatenate", "replace_token")
)
