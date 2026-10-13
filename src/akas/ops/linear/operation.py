"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants


def large_batch(p):
    return p.get("r", 1) > 8


def floating_optimized(p):
    return not p.get("ggml", False) and p.get("optimized", False) and p["k"] % 32 == 0


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
    *variants(
        ("linear",),
        "linear.cuda.bf16_batch",
        "bf16",
        semantic_keys=("r", "k", "o"),
        space=(
            ("m", (16, 32, 64)),
            ("n", (64, 128)),
            ("bk", (128, 64, 256)),
            ("stages", (2, 1, 3)),
        ),
    ),
    *variants(
        ("gemv",),
        "linear.cuda.bf16_batch",
        "bf16",
        semantic_keys=("r", "k", "o"),
        space=(("columns", (16, 8, 32)),),
    ),
    *variants(("bf16_linear",), "linear.cuda.decode", "block-fp8"),
    *variants(("fp8_linear",), "linear.cuda.decode", "block-fp8", phase="decode"),
    *variants(
        ("bf16_linear",),
        "linear.cuda.block_fp8",
        "block-fp8",
        phase="prefill",
        priority=30,
        accepts=lambda p: not p.get("decode_head"),
    ),
    *variants(
        ("fp8_linear",),
        "linear.cuda.block_fp8",
        "block-fp8",
        phase="prefill",
        kernel_kind="fp8_linear_mma_prequantized",
        defaults=(
            ("columns", 64),
            ("block_m", 16),
            ("threads", 128),
            ("packed_copy", True),
            ("stages", 2),
        ),
        space=(
            ("block_m", (16, 32, 64)),
            ("columns", (64, 32, 128)),
            ("threads", (128, 256)),
            ("stages", (2, 1, 3)),
            ("packed_copy", (True, False)),
        ),
    ),
    *variants(
        ("split_linear",),
        "linear.cuda.split",
        "bf16",
        semantic_keys=("r", "k", "o", "parts"),
        space=(
            ("m", (16, 32, 64)),
            ("n", (64, 128)),
            ("bk", (128, 256)),
            ("threads", (128, 256)),
        ),
    ),
    *variants(("split_merge",), "linear.cuda.split", "bf16"),
)

IMPLEMENTATIONS += tuple(
    item
    for provider in ("cuda", "webgpu")
    for item in variants(
        ("linear",),
        "linear.reference",
        "reference",
        provider=provider,
        accepts=lambda p: not p.get("ggml", False),
        space=(("threads", (32, 64, 128, 256)),),
    )
)

IMPLEMENTATIONS += variants(
    ("linear",),
    "linear.webgpu.reference",
    "reference",
    provider="webgpu",
    priority=20,
    accepts=lambda p: not p.get("ggml", False),
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


# Storage format is an operand choice in a common model graph. Prefill stages
# FP16 operands; decode preserves the packed GEMV arithmetic and fused pairs.
IMPLEMENTATIONS += variants(
    ("linear", "ffn", "linear_add"),
    "linear.cuda.packed",
    "reference",
    priority=40,
    accepts=lambda p: p.get("ggml", False) and p["r"] == 1,
    semantic_keys=("r", "k", "o", "type", "ggml"),
    space=(
        ("threads", (64, 128, 256)),
        ("unroll", (1, 2, 4, 8)),
        ("f16_values", (4, 8, 16)),
    ),
)
IMPLEMENTATIONS += variants(
    ("linear", "ffn", "linear_add"),
    "linear.cuda.packed",
    "reference",
    priority=40,
    accepts=lambda p: p.get("ggml", False) and p["r"] > 1,
    variant="prefill",
    semantic_keys=("r", "k", "o", "type", "ggml"),
    defaults=(("block_m", 32), ("block_n", 64), ("block_k", 32), ("stages", 2)),
    space=(
        ("block_m", (32, 16, 64)),
        ("block_n", (64, 32, 128)),
        ("block_k", (32, 64, 128)),
        ("stages", (2, 1, 3)),
        ("threads", (128, 256)),
    ),
)
IMPLEMENTATIONS += variants(
    ("linear", "ffn", "linear_add"),
    "linear.webgpu.packed",
    "reference",
    provider="webgpu",
    priority=40,
    accepts=lambda p: (
        p.get("ggml", False) and (p["r"] != 1 or p["type"] not in (2, 14))
    ),
)
IMPLEMENTATIONS += variants(
    ("linear", "ffn", "linear_add"),
    "linear.webgpu.packed",
    "reference",
    provider="webgpu",
    priority=40,
    variant="quantized-decode",
    accepts=lambda p: p.get("ggml", False) and p["r"] == 1 and p["type"] in (2, 14),
    semantic_keys=("r", "k", "o", "type", "ggml"),
    space=(
        ("sg", (False, True)),
        ("gemv_threads", (128, 64, 256)),
        ("gemv_lanes", (16, 32)),
        ("gemv_unroll", (1, 2, 4, 8)),
        ("gemv_chains", (1, 2, 4)),
    ),
)


from akas.ops.linear.validation import VALIDATIONS


from akas.ops.linear.rewrites import fuse_projections, fuse_residual_projection

REWRITES = (fuse_projections, fuse_residual_projection)
ROW_INDEPENDENT = ("linear",)


def project(graph, x, weight, *, width, name, role="scratch"):
    """Declare dense projection scratch from the selected operation geometry."""
    from akas.ops.linear.schedule import schedule

    rows, depth = x.shape
    columns, expected_depth = weight.shape
    if depth != expected_depth:
        raise ValueError("projection input depth differs from its weight")
    out = graph.buffer(name, (rows, columns), "float32", role=role)
    parameters = schedule(rows, depth, columns, width)
    partial = (
        out.view((rows, 1, columns))
        if parameters["parts"] == 1
        else graph.buffer(
            name + ".parts", (rows, parameters["parts"], columns), "float32"
        )
    )
    graph("split_linear", parameters, x, weight, partial)
    if parameters["parts"] > 1:
        graph("split_merge", parameters, partial, out)
    return out


from akas.ops.linear.lowering import lower
from akas.ops.linear.pipeline import validation as pipeline_validation

IMPLEMENTATIONS += variants(
    ("projection_partial",),
    "linear.pipeline",
    "block-fp8",
    semantic_keys=("r", "k", "o", "partitions"),
    space=(
        ("columns", (64, 32, 128)),
        ("block_m", (16, 32, 64)),
        ("threads", (128, 256)),
        ("stages", (1, 2, 3)),
    ),
)
IMPLEMENTATIONS += variants(("projection_merge",), "linear.pipeline", "block-fp8")
VALIDATIONS.update(
    {name: pipeline_validation for name in ("projection_partial", "projection_merge")}
)


# Output locations are part of the operation ABI, independent of compiler passes.
OUTPUT_ARGUMENTS = {"reference": {"linear": (-1,), "ffn": (-1,), "linear_add": (-1,)}}

IMPLEMENTATIONS += variants(
    ("projection_partial",),
    "linear.hardware.hopper.projection",
    "block-fp8",
    priority=20,
    targets=("sm_90a",),
    accepts=lambda p: p.get("hopper", False),
    parameters_only=True,
    semantic_keys=("r", "k", "o", "partitions"),
    space=(
        ("columns", (64, 128)),
        ("block_m", (16, 32, 64)),
        ("threads", (128, 256)),
        ("stages", (1, 2, 3)),
    ),
)


from akas.ops.contracts import OutputABI

for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update(
        {
            "linear": OutputABI((-1,), ("out",)),
            "gemv": OutputABI((-1,), ("out",)),
            "ffn": OutputABI((-1,), ("out",)),
            "linear_add": OutputABI((-1,), ("out",)),
            "bf16_linear": OutputABI((-1,), ("out",)),
            "fp8_linear": OutputABI((-1,), ("out",)),
            "split_linear": OutputABI((-1,), ("out",)),
            "split_merge": OutputABI((-1,), ("out",)),
            "projection_partial": OutputABI((-1,), ("out",)),
            "projection_merge": OutputABI((-1,), ("out",)),
        }
    )

# Keep per-CTA work bounded as request rows grow; beam search compares this
# tiled path with the scalar reduction while preserving its buffer contract.
IMPLEMENTATIONS += variants(
    ("bf16_linear",),
    "linear.cuda.block_fp8",
    "block-fp8",
    phase="decode",
    priority=20,
    accepts=lambda p: p.get("r", 1) > 8 and p["k"] % 64 == 0,
    variant="large-batch",
)
IMPLEMENTATIONS += variants(
    ("bf16_linear",),
    "linear.cuda.block_fp8",
    "block-fp8",
    phase="prefill",
    priority=20,
    required=("decode_head",),
    accepts=lambda p: p.get("decode_head") and p.get("r", 1) > 8 and p["k"] % 64 == 0,
    variant="large-head",
)
