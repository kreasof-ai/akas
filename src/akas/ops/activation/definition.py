"""Contracts and implementation choices for activation."""

from akas.kernels.contracts import OutputABI
from akas.ops.activation.validation import validation, sigmoid_bias_validation
from akas.kernels.contracts import Implementation

ROW_INDEPENDENT = ("swiglu",)
OUTPUT_ARGUMENTS = {"reference": {"swiglu": (-1,)}}
for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update(
        {
            "swiglu": OutputABI((-1,), ("out",)),
            "swiglu_experts": OutputABI((-1,), ("out",)),
        }
    )
VALIDATIONS = {name: validation for name in ("swiglu", "swiglu_experts")}
IMPLEMENTATIONS = (
    Implementation(
        "swiglu",
        "akas.ops.activation.kernel",
        "packed-baseline",
        defaults=(("flat", True), ("input_names", ("x", "y"))),
        factory="swiglu",
        name="activation.kernel",
    ),
    Implementation(
        "swiglu",
        "akas.ops.activation.kernel",
        "packed",
        defaults=(("flat", True), ("input_names", ("x", "y"))),
        factory="swiglu",
        name="activation.kernel",
    ),
    Implementation(
        "swiglu",
        "akas.ops.activation.kernel",
        "reference",
        defaults=(("flat", True), ("input_names", ("x", "y"))),
        factory="swiglu",
        name="activation.kernel",
    ),
    Implementation(
        "swiglu",
        "akas.ops.activation.kernel",
        "bf16",
        defaults=(
            ("packed", True),
            ("output_dtype", "bfloat16"),
            ("round_inputs", True),
        ),
        factory="swiglu",
        name="activation.kernel",
    ),
    Implementation(
        "swiglu",
        "akas.ops.activation.kernel",
        "block-fp8",
        defaults=(("output_dtype", "bfloat16"), ("round_inputs", True)),
        factory="swiglu",
        name="activation.kernel",
    ),
    Implementation(
        "swiglu",
        "akas.ops.activation.kernel",
        "packed-baseline",
        provider="webgpu",
        defaults=(("flat", True), ("input_names", ("x", "y"))),
        factory="swiglu",
        name="activation.kernel",
    ),
    Implementation(
        "swiglu",
        "akas.ops.activation.kernel",
        "packed",
        provider="webgpu",
        defaults=(("flat", True), ("input_names", ("x", "y"))),
        factory="swiglu",
        name="activation.kernel",
    ),
    Implementation(
        "swiglu",
        "akas.ops.activation.kernel",
        "reference",
        provider="webgpu",
        defaults=(("flat", True), ("input_names", ("x", "y"))),
        factory="swiglu",
        name="activation.kernel",
    ),
    Implementation(
        "swiglu_experts",
        "akas.ops.activation.kernel",
        "block-fp8",
        defaults=(
            ("expert_rows", True),
            ("output_dtype", "bfloat16"),
            ("round_inputs", True),
        ),
        factory="swiglu_experts",
        name="activation.kernel",
    ),
    Implementation(
        "swiglu_experts",
        "akas.ops.activation.kernel",
        "bf16",
        defaults=(
            ("expert_rows", True),
            ("output_dtype", "bfloat16"),
            ("round_inputs", True),
        ),
        factory="swiglu_experts",
        name="activation.kernel:encoded-experts",
    ),
    Implementation(
        "swiglu",
        "akas.ops.activation.kernel",
        "bf16",
        provider="webgpu",
        defaults=(
            ("packed", True),
            ("output_dtype", "float16"),
            ("round_dtype", "float16"),
            ("round_inputs", True),
        ),
        factory="swiglu",
        name="activation.kernel",
    ),
)

IMPLEMENTATIONS += tuple(
    Implementation(
        "swiglu_experts",
        "akas.ops.activation.kernel",
        "reference",
        provider=provider,
        factory="swiglu_experts",
        defaults=(("expert_rows", True),),
    )
    for provider in ("cuda", "webgpu")
)


OUTPUT_ARGUMENTS["reference"]["sigmoid_bias"] = (-1,)
VALIDATIONS["sigmoid_bias"] = sigmoid_bias_validation
IMPLEMENTATIONS += tuple(
    Implementation(
        "sigmoid_bias",
        "akas.ops.activation.kernel",
        "reference",
        provider=provider,
        factory="sigmoid_bias",
        name="activation.sigmoid_bias",
    )
    for provider in ("cuda", "webgpu")
)


from akas.ops.activation.validation import gated_validation

ROW_INDEPENDENT += ("geglu", "head_gate")
for _kind in ("geglu", "head_gate"):
    OUTPUT_ARGUMENTS["reference"][_kind] = OutputABI((-1,), ("out",))
    VALIDATIONS[_kind] = gated_validation
    IMPLEMENTATIONS += (
        Implementation(
            _kind,
            "akas.ops.activation.kernel",
            "reference",
            factory=_kind,
            name="activation." + _kind,
        ),
    )

OUTPUT_ARGUMENTS["reference"]["head_gate_half"] = OutputABI((-1,), ("out",))
VALIDATIONS["head_gate_half"] = gated_validation
ROW_INDEPENDENT += ("head_gate_half",)
IMPLEMENTATIONS += (
    Implementation(
        "head_gate_half",
        "akas.ops.activation.kernel",
        "reference",
        factory="head_gate_half",
        name="activation.head_gate_half",
    ),
)

IMPLEMENTATIONS += (
    Implementation(
        "head_gate",
        "akas.ops.activation.kernel",
        "reference",
        provider="webgpu",
        factory="head_gate",
        name="activation.webgpu.head_gate",
    ),
)
