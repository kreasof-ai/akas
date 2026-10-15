"""Contracts and implementation choices for fp8."""

from akas.memory.state import BufferSpec
from akas.quantization.formats import Format, Storage
from akas.kernels.contracts import Implementation
from akas.quantization.activations.fp8.validation import validation
from akas.kernels.contracts import OutputABI

BLOCK = 128


def layout(shape):
    depth = shape[-1]
    if depth % BLOCK:
        raise ValueError("FP8 channels must be divisible by {}".format(BLOCK))
    return Storage(
        BufferSpec(shape, "uint8"),
        (BufferSpec((*shape[:-1], depth // BLOCK), "float32"),),
        ("scales",),
    )


def webgpu_layout(shape):
    original = layout(shape)
    return Storage(
        BufferSpec((*shape[:-1], shape[-1] // 4), "uint32"),
        original.metadata,
        original.metadata_names,
    )


FORMAT = Format(
    "fp8",
    layout,
    (
        ("cuda", "akas.quantization.activations.fp8.cuda"),
        ("webgpu", "akas.quantization.activations.fp8.webgpu"),
    ),
    layouts=(("webgpu", webgpu_layout),),
)


OUTPUT_ARGUMENTS = {
    "*": {
        "dequantize_fp8": OutputABI((-1,), ("out",)),
        "quantize": OutputABI((1, 2), ("out", "scales")),
    },
}

VALIDATIONS = {kind: validation for kind in ("quantize", "dequantize_fp8")}
IMPLEMENTATIONS = (
    (
        Implementation(
            "quantize",
            "akas.quantization.activations.fp8.cuda",
            "block-fp8",
            factory="quantize",
            name="akas.quantization.activations.fp8",
        ),
    )
    + tuple(
        Implementation(
            "dequantize_fp8",
            f"akas.quantization.activations.fp8.{provider}",
            "*",
            factory="dequantize",
            provider=provider,
        )
        for provider in ("cuda", "webgpu")
    )
    + (
        Implementation(
            "quantize",
            "akas.quantization.activations.fp8.webgpu",
            "block-fp8",
            factory="quantize",
            provider="webgpu",
        ),
    )
)
