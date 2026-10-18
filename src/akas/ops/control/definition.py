"""Contracts and implementation choices for control."""

from akas.kernels.contracts import Implementation
from akas.ops.control.validation import feedback_validation as feedback_validation
from akas.ops.control.validation import validation
from akas.ops.control.validation import copy_validation as copy_validation

OUTPUT_ARGUMENTS = {}
KINDS = (
    "spec_setup",
    "spec_draft",
    "spec_proposal",
    "spec_accept",
    "spec_repair",
    "spec_result",
)
VALIDATIONS = {name: validation for name in KINDS}
VALIDATIONS["token_feedback"] = feedback_validation
VALIDATIONS["copy_buffer"] = copy_validation
IMPLEMENTATIONS = (
    Implementation(
        "spec_setup",
        "akas.ops.control.kernel",
        "speculation",
        factory="spec_setup",
        name="control.speculative",
    ),
    Implementation(
        "spec_draft",
        "akas.ops.control.kernel",
        "speculation",
        factory="spec_draft",
        name="control.speculative",
    ),
    Implementation(
        "spec_proposal",
        "akas.ops.control.kernel",
        "speculation",
        factory="spec_proposal",
        name="control.speculative",
    ),
    Implementation(
        "spec_accept",
        "akas.ops.control.kernel",
        "speculation",
        factory="spec_accept",
        name="control.speculative",
    ),
    Implementation(
        "spec_repair",
        "akas.ops.control.kernel",
        "speculation",
        factory="spec_repair",
        name="control.speculative",
    ),
    Implementation(
        "spec_result",
        "akas.ops.control.kernel",
        "speculation",
        factory="spec_result",
        name="control.speculative",
    ),
    Implementation(
        "token_feedback",
        "akas.ops.control.kernel",
        "*",
        factory="token_feedback",
        name="control.feedback",
    ),
    Implementation(
        "token_feedback",
        "akas.ops.control.kernel",
        "*",
        provider="webgpu",
        factory="token_feedback",
        name="control.feedback",
    ),
    Implementation(
        "copy_buffer",
        "akas.ops.control.kernel",
        "*",
        factory="copy_buffer",
        name="control.copy",
    ),
    Implementation(
        "copy_buffer",
        "akas.ops.control.kernel",
        "*",
        provider="webgpu",
        factory="copy_buffer",
        name="control.copy",
    ),
    Implementation(
        "spec_setup",
        "akas.ops.control.kernel",
        "speculation",
        provider="webgpu",
        factory="spec_setup",
        name="control.speculative",
    ),
    Implementation(
        "spec_draft",
        "akas.ops.control.kernel",
        "speculation",
        provider="webgpu",
        factory="spec_draft",
        name="control.speculative",
    ),
    Implementation(
        "spec_proposal",
        "akas.ops.control.kernel",
        "speculation",
        provider="webgpu",
        factory="spec_proposal",
        name="control.speculative",
    ),
    Implementation(
        "spec_accept",
        "akas.ops.control.kernel",
        "speculation",
        provider="webgpu",
        factory="spec_accept",
        name="control.speculative",
    ),
    Implementation(
        "spec_repair",
        "akas.ops.control.kernel",
        "speculation",
        provider="webgpu",
        defaults=(("dtype", "float16"),),
        factory="spec_repair",
        name="control.speculative",
    ),
    Implementation(
        "spec_result",
        "akas.ops.control.kernel",
        "speculation",
        provider="webgpu",
        factory="spec_result",
        name="control.speculative",
    ),
)
