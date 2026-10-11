"""Existing block-128 E4M3 KV format with independent FP32 scales for K and V."""

from akas.memory.state import BufferSpec
from akas.ops.kv_quantization.format import Format, Storage

BLOCK = 128


def layout(shape):
    scales = shape[:-1] + ((shape[-1] + BLOCK - 1) // BLOCK,)
    return Storage(BufferSpec(shape, "uint8"), (BufferSpec(scales, "float32"),))


def webgpu_layout(shape):
    original = layout(shape)
    return Storage(
        BufferSpec((*shape[:-1], (shape[-1] + 3) // 4), "uint32"), original.metadata
    )


FORMAT = Format(
    "fp8",
    layout,
    (
        ("cuda", "akas.ops.kv_quantization.cuda.fp8"),
        ("webgpu", "akas.ops.kv_quantization.webgpu.fp8"),
    ),
    layouts=(("webgpu", webgpu_layout),),
)
