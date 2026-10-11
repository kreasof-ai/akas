"""Standalone cache operations; representation is selected by the codec module."""

from akas.ops.registry import variants

IMPLEMENTATIONS = variants(
    ("kv_encode", "kv_decode"), "kv_quantization.kernels", "kv-pair"
)
