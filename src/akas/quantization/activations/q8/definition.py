"""Contracts and implementation choices for q8."""

from math import prod
from akas.memory.state import BufferSpec
from akas.quantization.formats import Format, Storage
from akas.kernels.contracts import OutputABI
from akas.kernels.contracts import Implementation
from akas.quantization.activations.q8.validation import validation

BLOCK = 32


def layout(shape):
    depth = shape[-1]
    if depth % BLOCK:
        raise ValueError("Q8 channels must be divisible by {}".format(BLOCK))
    rows = prod(shape[:-1])
    blocks = 1 * rows * depth // BLOCK
    return Storage(
        BufferSpec((1 * rows * depth // 4,), "uint32"),
        (BufferSpec((blocks,), "float32"), BufferSpec((blocks,), "int32")),
        ("scales", "sums"),
    )


FORMAT = Format(
    "q8",
    layout,
    (
        ("cuda", "akas.quantization.activations.q8.cuda"),
        ("webgpu", "akas.quantization.activations.q8.webgpu"),
    ),
)


OUTPUT_ARGUMENTS = {
    "*": {
        "quantize_q8": OutputABI((1, 2, 3), ("out", "scales", "sums")),
        "dequantize_q8": OutputABI((-1,), ("out",)),
    },
}

VALIDATIONS = {kind: validation for kind in ("quantize_q8", "dequantize_q8")}
IMPLEMENTATIONS = (
    (
        Implementation(
            "quantize_q8",
            "akas.quantization.activations.q8.webgpu",
            "packed",
            provider="webgpu",
            priority=20,
            factory="quantize",
            name="akas.quantization.activations.packing",
        ),
        Implementation(
            "quantize_q8",
            "akas.quantization.activations.q8.webgpu",
            "reference",
            provider="webgpu",
            priority=20,
            factory="quantize",
            name="akas.quantization.activations.packing",
        ),
    )
    + tuple(
        Implementation(
            "quantize_q8",
            "akas.quantization.activations.q8.cuda",
            contract,
            factory="quantize",
            provider="cuda",
        )
        for contract in ("reference", "packed")
    )
    + tuple(
        Implementation(
            "dequantize_q8",
            f"akas.quantization.activations.q8.{provider}",
            "*",
            factory="dequantize",
            provider=provider,
        )
        for provider in ("cuda", "webgpu")
    )
)
