"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


IMPLEMENTATIONS = (
    *variants(("conv",), "convolution.cuda.packed_baseline", "packed-baseline"),
    *variants(("conv",), "convolution.cuda.packed_baseline", "packed", priority=0),
    *variants(
        ("conv",),
        "convolution.cuda.packed_baseline",
        "packed",
        provider="webgpu",
        priority=0,
    ),
)
