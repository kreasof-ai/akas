"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


def large_batch(p):
    return p.get("r", 1) > 8


IMPLEMENTATIONS = (
    *variants(
        ("fp8_linear",),
        "linear.cuda.block_fp8",
        "block-fp8",
        phase="decode",
        priority=20,
        accepts=large_batch,
        kernel_kind="fp8_linear_mma",
        defaults=(("block_m", 16), ("columns", 64), ("threads", 128)),
    ),
    *variants(
        ("linear",),
        "linear.cuda.packed",
        "packed",
        priority=20,
        space=(
            ("threads", (64, 128, 256)),
            ("unroll", (1, 2, 4, 8)),
            ("f16_values", (4, 8, 16)),
        ),
    ),
    *variants(("linear_add", "ffn"), "linear.cuda.packed", "packed", priority=20),
    *variants(("linear",), "linear.cuda.packed_baseline", "packed-baseline"),
    *variants(("linear",), "linear.cuda.packed_baseline", "packed", priority=0),
    *variants(
        ("linear",),
        "linear.cuda.packed_baseline",
        "packed",
        provider="webgpu",
        priority=0,
    ),
    *variants(
        ("linear", "ffn", "linear_q16", "ffn_q16", "linear_add", "linear_q8"),
        "linear.webgpu.packed",
        "packed",
        provider="webgpu",
        priority=20,
    ),
    *variants(("linear", "gemv"), "linear.cuda.bf16_batch", "bf16"),
    *variants(
        ("bf16_linear", "fp8_linear"), "linear.cuda.decode", "block-fp8", phase="decode"
    ),
    *variants(("split_linear", "split_merge"), "linear.cuda.split", "bf16"),
)

IMPLEMENTATIONS += tuple(
    item
    for provider in ("cuda", "webgpu")
    for item in variants(
        ("linear",),
        "linear.reference",
        "reference",
        provider=provider,
        space=(("threads", (32, 64, 128, 256)),),
    )
)

IMPLEMENTATIONS += variants(
    ("linear",),
    "linear.webgpu.reference",
    "reference",
    provider="webgpu",
    priority=20,
    space=(("threads", (32, 64, 128, 256)),),
)
