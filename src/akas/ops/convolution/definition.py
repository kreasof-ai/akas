"""Contracts and implementation choices for convolution."""

from akas.kernels.contracts import Implementation
from akas.ops.convolution.validation import validation

OUTPUT_ARGUMENTS = {"reference": {"conv": (2, 3)}}
VALIDATIONS = {"conv": validation}
IMPLEMENTATIONS = (
    Implementation(
        "conv",
        "akas.ops.convolution.kernel",
        "packed-baseline",
        factory="convolution_flat",
        name="convolution.cuda.packed_baseline",
    ),
    Implementation(
        "conv",
        "akas.ops.convolution.kernel",
        "packed",
        priority=0,
        factory="convolution_flat",
        name="convolution.cuda.packed_baseline",
    ),
    Implementation(
        "conv",
        "akas.ops.convolution.kernel",
        "packed",
        provider="webgpu",
        priority=0,
        factory="convolution_flat",
        name="convolution.cuda.packed_baseline",
    ),
    Implementation(
        "conv",
        "akas.ops.convolution.kernel",
        "reference",
        factory="convolution",
        name="convolution.reference",
    ),
    Implementation(
        "conv",
        "akas.ops.convolution.kernel",
        "reference",
        provider="webgpu",
        factory="convolution",
        name="convolution.reference",
    ),
)
