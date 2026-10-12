"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


def large_batch(p):
    return p.get("r", 1) > 8


def floating_optimized(p):
    return p.get("optimized", False) and p["k"] % 32 == 0


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

IMPLEMENTATIONS += variants(
    ("linear", "ffn"),
    "linear.cuda.floating",
    "reference",
    priority=30,
    accepts=floating_optimized,
    space=(
        ("minimum_mma_groups", (32, 0, 64, 128, 256)),
        ("maximum_gemv_rows", (31, 24, 8, 16, 25, 28, 32, 64)),
        ("block_m", (32, 16, 64)),
        ("block_n", (64, 32, 128)),
        ("block_k", (32, 64, 128)),
    ),
)

IMPLEMENTATIONS += variants(
    ("linear", "ffn"),
    "linear.webgpu.floating",
    "reference",
    provider="webgpu",
    priority=30,
    accepts=floating_optimized,
    space=(
        ("minimum_mma_groups", (0, 64, 128, 256, 512, 1024)),
        ("schedule", ("register", "outer")),
        ("outer_shared_dtype", ("float16", "float32")),
    ),
)
