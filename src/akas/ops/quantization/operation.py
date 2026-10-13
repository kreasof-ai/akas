"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


IMPLEMENTATIONS = (
    *variants(
        ("quantize_q8", "quantize_q16"),
        "quantization.webgpu.packed",
        "packed",
        provider="webgpu",
        priority=20,
    ),

)
