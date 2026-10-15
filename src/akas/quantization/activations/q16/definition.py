"""Contracts and implementation choices for q16."""

from math import prod
from akas.memory.state import BufferSpec
from akas.quantization.formats import Format, Storage
from akas.kernels.contracts import OutputABI
from akas.kernels.contracts import Implementation
from akas.quantization.activations.q16.validation import validation

BLOCK = 32


def layout(shape):
    depth = shape[-1]
    if depth % BLOCK:
        raise ValueError("Q16 channels must be divisible by {}".format(BLOCK))
    rows = prod(shape[:-1])
    blocks = 2 * rows * depth // BLOCK
    return Storage(
        BufferSpec((2 * rows * depth // 4,), "uint32"),
        (BufferSpec((blocks,), "float32"), BufferSpec((blocks,), "int32")),
        ("scales", "sums"),
    )


FORMAT = Format(
    "q16",
    layout,
    (
        ("cuda", "akas.quantization.activations.q16.cuda"),
        ("webgpu", "akas.quantization.activations.q16.webgpu"),
    ),
)


OUTPUT_ARGUMENTS = {
    "*": {
        "quantize_q16": OutputABI((1, 2, 3), ("packed", "scales", "sums")),
        "dequantize_q16": OutputABI((-1,), ("out",)),
    },
}

VALIDATIONS = {kind: validation for kind in ("quantize_q16", "dequantize_q16")}
IMPLEMENTATIONS = (
    (
        Implementation(
            "quantize_q16",
            "akas.quantization.activations.q16.webgpu",
            "reference",
            provider="webgpu",
            priority=20,
            space=(("sg", (False, True)),),
            semantic_keys=("r", "k"),
            factory="quantize",
            name="akas.quantization.activations.packing",
        ),
        Implementation(
            "quantize_q16",
            "akas.quantization.activations.q16.webgpu",
            "packed",
            provider="webgpu",
            priority=20,
            space=(("sg", (False, True)),),
            semantic_keys=("r", "k"),
            factory="quantize",
            name="akas.quantization.activations.packing",
        ),
    )
    + tuple(
        Implementation(
            "quantize_q16",
            "akas.quantization.activations.q16.cuda",
            contract,
            factory="quantize",
            provider="cuda",
        )
        for contract in ("reference", "packed")
    )
    + tuple(
        Implementation(
            "dequantize_q16",
            f"akas.quantization.activations.q16.{provider}",
            "*",
            factory="dequantize",
            provider=provider,
        )
        for provider in ("cuda", "webgpu")
    )
)
