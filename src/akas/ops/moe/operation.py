"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants

IMPLEMENTATIONS = (
    *variants(
        ("moe_groups", "fp8_experts"),
        "moe.cuda.decode",
        "block-fp8",
        phase="decode",
    ),
    *variants(("router", "moe_combine"), "moe.cuda.decode", "block-fp8"),
    *variants(
        ("fp8_experts",),
        "moe.cuda.prefill",
        "block-fp8",
        phase="prefill",
        defaults=(("block_m", 16), ("threads", 128)),
        space=(
            ("block_m", (16, 32, 64)),
            ("columns", (64, 32, 128)),
            ("threads", (128, 256)),
        ),
    ),
    *variants(
        ("expert_routes",),
        "moe.cuda.prefill",
        "block-fp8",
        phase="prefill",
        required=("slots", "chunk"),
    ),
)


from akas.ops.moe.validation import VALIDATIONS


def routed_workspace(graph, rows, config, *, prefix="", grouped=False):
    """Declare router, expert and shared-branch buffers from operation geometry."""
    for name, value in (
        ("width", 2048),
        ("expert_width", 512),
        ("experts", 256),
        ("top_k", 8),
    ):
        if getattr(config, name) != value:
            raise ValueError(f"native routed kernels require {name}={value}")
    width, hidden, top = config.width, config.expert_width, config.top_k
    specs = {
        "router": (rows, config.experts),
        "shared_gate": (rows, 1),
        "shared_g": (rows, hidden),
        "shared_u": (rows, hidden),
        "shared_out": (rows, width),
        "ids": (rows, top),
        "router_weights": (rows, top),
        "expert_g": (rows, top, hidden),
        "expert_u": (rows, top, hidden),
        "expert_out": (rows, top, width),
        "expert_activation": (rows, top, hidden),
        "shared_activation": (rows, hidden),
    }
    if grouped:
        specs.update(counts=(config.experts,), routes=(config.experts, rows))
    else:
        specs.update(experts=(rows * top,), routes=(rows * top, rows))
    return {
        name: graph.buffer(
            prefix + name,
            shape,
            "int32"
            if name in ("ids", "experts", "routes", "counts")
            else "bfloat16"
            if name.endswith("activation")
            else "float32",
        )
        for name, shape in specs.items()
    }


from akas.ops.moe.lowering import lower
from akas.ops.moe.pipeline import validation as pipeline_validation

IMPLEMENTATIONS += variants(
    ("expert_partial",),
    "moe.pipeline",
    "block-fp8",
    semantic_keys=(
        "r",
        "k",
        "o",
        "experts",
        "top",
        "routed_input",
        "partitions",
        "block_m",
        "compact",
        "hopper",
        "resident_weights",
    ),
    space=(("columns", (64, 32, 128)), ("threads", (128, 256)), ("stages", (2, 1, 3))),
)
IMPLEMENTATIONS += variants(
    ("expert_tiles", "expert_merge"), "moe.pipeline", "block-fp8"
)
VALIDATIONS.update(
    {
        name: pipeline_validation
        for name in ("expert_tiles", "expert_partial", "expert_merge")
    }
)

IMPLEMENTATIONS += variants(
    ("expert_routes",),
    "moe.cuda.prefill",
    "block-fp8",
    required=("bucket_policy",),
    priority=30,
)

IMPLEMENTATIONS += variants(
    ("expert_partial",),
    "moe.hardware.hopper.experts",
    "block-fp8",
    priority=20,
    targets=("sm_90a",),
    accepts=lambda p: p.get("hopper", False),
    factory="expert_kernel",
    parameters_only=True,
    semantic_keys=(
        "r",
        "rows",
        "k",
        "o",
        "experts",
        "top",
        "routed_input",
        "partitions",
        "block_m",
        "compact",
        "hopper",
        "resident_weights",
    ),
    space=(("columns", (64, 128)), ("threads", (128, 256)), ("stages", (1, 2, 3))),
)

IMPLEMENTATIONS += variants(
    ("moe_groups",),
    "moe.cuda.decode",
    "block-fp8",
    required=("fixed_rows",),
    priority=30,
    variant="fixed-rows",
)


from akas.ops.contracts import OutputABI

OUTPUT_ARGUMENTS = {}

for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update({'router': OutputABI((1, 2), ('ids', 'weights')), 'moe_groups': OutputABI((1, 2), ('experts', 'routes')), 'expert_routes': OutputABI((1, 2), ('counts', 'routes')), 'fp8_experts': OutputABI((-1,), ('out',)), 'moe_combine': OutputABI((-1,), ('out',)), 'expert_partial': OutputABI((-1,), ('out',)), 'expert_merge': OutputABI((-1,), ('out',))})

OUTPUT_ARGUMENTS["block-fp8"]["expert_tiles"] = OutputABI((1, 2), ("tile_experts", "tile_offsets"))
