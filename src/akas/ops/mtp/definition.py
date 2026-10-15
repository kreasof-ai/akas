"""Contracts and implementation choices for mtp."""

from akas.kernels.contracts import OutputABI
from akas.kernels.contracts import Implementation
from akas.ops.mtp.validation import validation

OUTPUT_ARGUMENTS = {}
for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update(
        {"mtp_join": OutputABI((-1,), ("out",)), "mtp_cast": OutputABI((-1,), ("out",))}
    )
VALIDATIONS = {name: validation for name in ("mtp_join", "mtp_cast")}
IMPLEMENTATIONS = (
    Implementation(
        "mtp_join", "akas.ops.mtp.kernel", "bf16", factory="mtp_join", name="mtp.cuda"
    ),
    Implementation(
        "mtp_cast", "akas.ops.mtp.kernel", "bf16", factory="mtp_cast", name="mtp.cuda"
    ),
    Implementation(
        "mtp_join",
        "akas.ops.mtp.kernel",
        "block-fp8",
        factory="mtp_join",
        name="mtp.cuda",
    ),
    Implementation(
        "mtp_cast",
        "akas.ops.mtp.kernel",
        "block-fp8",
        factory="mtp_cast",
        name="mtp.cuda",
    ),
)
