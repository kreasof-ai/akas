"""Contracts and implementation choices for fp8."""

from akas.memory.state import BufferSpec
from akas.quantization.formats import Format, Storage
from akas.kernels.contracts import Implementation
from .validation import validation
from akas.kernels.contracts import OutputABI

BLOCK = 128

NAME = "fp8"


def layout(shape):
    scales = shape[:-1] + ((shape[-1] + BLOCK - 1) // BLOCK,)
    return Storage(
        BufferSpec(shape, "uint8"), (BufferSpec(scales, "float32"),), ("scales",)
    )


def webgpu_layout(shape):
    original = layout(shape)
    return Storage(
        BufferSpec((*shape[:-1], (shape[-1] + 3) // 4), "uint32"),
        original.metadata,
        original.metadata_names,
    )


FORMAT = Format(
    "fp8",
    layout,
    (
        ("cuda", "akas.quantization.kv.fp8.cuda"),
        ("webgpu", "akas.quantization.kv.fp8.webgpu"),
    ),
    layouts=(("webgpu", webgpu_layout),),
)


OUTPUT_ARGUMENTS = {
    "*": {
        "kv_encode": OutputABI((4, 5, 6, 7), ("kc", "vc", "ks", "vs")),
        "kv_decode": OutputABI((-2, -1), ("keys", "values")),
        "kv_stage": OutputABI((-3, -2, -1), ("keys", "values", "scales")),
    },
}

VALIDATIONS = {kind: validation for kind in ("kv_encode", "kv_decode", "kv_stage")}
IMPLEMENTATIONS = (
    Implementation(
        "kv_encode",
        "akas.quantization.kv.fp8.cuda",
        "kv-pair",
        factory="kv_encode",
        name="akas.quantization.kv.fp8",
    ),
    Implementation(
        "kv_decode",
        "akas.quantization.kv.fp8.cuda",
        "kv-pair",
        factory="kv_decode",
        name="akas.quantization.kv.fp8",
    ),
) + tuple(
    Implementation(
        kind,
        "akas.quantization.kv.fp8.webgpu",
        "kv-pair",
        factory=factory,
        provider="webgpu",
    )
    for kind, factory in (("kv_encode", "quantize"), ("kv_decode", "dequantize"))
)

IMPLEMENTATIONS += tuple(
    Implementation(
        "kv_stage",
        f"akas.quantization.kv.fp8.{provider}",
        "*",
        factory="stage",
        provider=provider,
        name="akas.quantization.kv.fp8.stage",
        accepts=lambda p: p.get("kv_dtype", "fp8") == "fp8",
    )
    for provider in ("cuda", "webgpu")
)
