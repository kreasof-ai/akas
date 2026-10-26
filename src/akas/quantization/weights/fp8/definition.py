"""Contracts and implementation choices for fp8."""

from akas.memory.state import BufferSpec
from akas.quantization.formats import Format, Storage
from akas.kernels.contracts import Implementation
from akas.quantization.weights.fp8.validation import validation
from akas.kernels.contracts import OutputABI

BLOCK = 128


def layout(shape, *, scale_layout="block"):
    rows, depth = shape[-2:]
    if scale_layout == "channel":
        return Storage(
            BufferSpec(shape, "uint8"),
            (BufferSpec((*shape[:-2], rows, 1), "bfloat16"),),
            ("scales",),
        )
    if scale_layout != "block":
        raise ValueError("FP8 weight scale layout must be block or channel")
    if depth % BLOCK:
        raise ValueError("FP8 weight channels must be divisible by 128")
    scales = (*shape[:-2], (rows + BLOCK - 1) // BLOCK, depth // BLOCK)
    return Storage(
        BufferSpec(shape, "uint8"), (BufferSpec(scales, "bfloat16"),), ("scales",)
    )


def webgpu_layout(shape):
    original = layout(shape)
    return Storage(
        BufferSpec((*shape[:-1], shape[-1] // 4), "uint32"),
        (BufferSpec(original.metadata[0].shape, "float32"),),
        ("scales",),
    )


FORMAT = Format(
    "fp8",
    layout,
    (
        ("cuda", "akas.quantization.weights.fp8.cuda"),
        ("webgpu", "akas.quantization.weights.fp8.webgpu"),
    ),
    layouts=(("webgpu", webgpu_layout),),
)


OUTPUT_ARGUMENTS = {
    "*": {
        "dequantize_weight_fp8": OutputABI((-1,), ("out",)),
    },
}

VALIDATIONS = {kind: validation for kind in ("e4m3_decode", "dequantize_weight_fp8")}
IMPLEMENTATIONS = (
    Implementation(
        "e4m3_decode",
        "akas.quantization.weights.fp8.cuda",
        "block-fp8",
        factory="decode",
        name="akas.quantization.weights.block_fp8",
    ),
) + tuple(
    Implementation(
        "dequantize_weight_fp8",
        f"akas.quantization.weights.fp8.{provider}",
        "*",
        factory="dequantize",
        provider=provider,
    )
    for provider in ("cuda", "webgpu")
)
