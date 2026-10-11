"""Operation kernels for quantization.webgpu.packed."""

from akas.ops.quantization.webgpu.storage import quantize_kernel


def make_kernel(kind, p):
    if kind in ("quantize_q8", "quantize_q16"):
        return quantize_kernel(kind, p)
    raise ValueError("unsupported operation: " + kind)
