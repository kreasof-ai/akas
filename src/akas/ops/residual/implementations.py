"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


IMPLEMENTATIONS = (
    *variants(("add",), "residual.cuda.packed_baseline", "packed-baseline"),
    *variants(("add",), "residual.cuda.packed_baseline", "packed", priority=0),
    *variants(
        ("add",),
        "residual.cuda.packed_baseline",
        "packed",
        provider="webgpu",
        priority=0,
    ),
)
