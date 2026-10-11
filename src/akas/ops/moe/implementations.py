"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


IMPLEMENTATIONS = (
    *variants(
        ("moe_groups", "fp8_experts", "router", "moe_combine"),
        "moe.cuda.decode",
        "block-fp8",
        phase="decode",
    ),
    *variants(
        ("expert_routes",),
        "moe.cuda.prefill",
        "block-fp8",
        phase="prefill",
        required=("slots", "chunk"),
    ),
)
