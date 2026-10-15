"""Contracts and implementation choices for residual."""

from akas.kernels.contracts import OutputABI
from akas.kernels.contracts import Implementation
from akas.ops.residual.validation import validation

OUTPUT_ARGUMENTS = {"reference": {"add": (-1,)}}
for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update(
        {"add": OutputABI((-1,), ("out",))}
    )
VALIDATIONS = {"add": validation}
IMPLEMENTATIONS = (
    Implementation(
        "add",
        "akas.ops.residual.kernel",
        "packed-baseline",
        factory="add",
        name="residual.kernel",
    ),
    Implementation(
        "add",
        "akas.ops.residual.kernel",
        "packed",
        priority=0,
        factory="add",
        name="residual.kernel",
    ),
    Implementation(
        "add",
        "akas.ops.residual.kernel",
        "packed",
        provider="webgpu",
        priority=0,
        factory="add",
        name="residual.kernel",
    ),
    Implementation(
        "add",
        "akas.ops.residual.kernel",
        "reference",
        factory="add",
        name="residual.kernel",
    ),
    Implementation(
        "add",
        "akas.ops.residual.kernel",
        "reference",
        provider="webgpu",
        factory="add",
        name="residual.kernel",
    ),
)
