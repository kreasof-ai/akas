"""Contracts and implementation choices for residual."""

from akas.kernels.spec import OutputABI
from akas.kernels.spec import Implementation
from akas.ops.residual.validation import validation
from akas.ops.residual.validation import bank_validation
from akas.ops.residual.validation import block_mix_validation
from akas.ops.residual.validation import scaled_residual_validation


OUTPUT_ARGUMENTS = {"reference": {"add": (-1,)}}
for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update(
        {"add": OutputABI((-1,), ("out",))}
    )
VALIDATIONS = {"add": validation}
OUTPUT_ARGUMENTS["*"] = {"attention_residual": OutputABI((0, -1), ("bank", "out"))}
VALIDATIONS["attention_residual"] = bank_validation
OUTPUT_ARGUMENTS["*"]["scaled_residual"] = OutputABI((-1,), ("out",))
VALIDATIONS["scaled_residual"] = scaled_residual_validation
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

IMPLEMENTATIONS += tuple(
    Implementation(
        kind,
        "akas.ops.residual.kernel",
        "*",
        provider=provider,
        factory=kind,
    )
    for provider in ("cuda", "webgpu")
    for kind in ("attention_residual", "scaled_residual")
)

for _kind in ("normalized_block_mix", "message_blend"):
    OUTPUT_ARGUMENTS["reference"][_kind] = OutputABI((-1,), ("out",))
    VALIDATIONS[_kind] = block_mix_validation
    IMPLEMENTATIONS += (
        Implementation(_kind, "akas.ops.residual.kernel", "reference", factory=_kind),
    )
