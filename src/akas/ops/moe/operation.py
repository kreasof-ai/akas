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
    width, hidden, top = config.width, config.expert_width, config.top_k
    specs = {"router": (rows, config.experts), "shared_gate": (rows, 1),
             "shared_g": (rows, hidden), "shared_u": (rows, hidden),
             "shared_out": (rows, width), "ids": (rows, top),
             "router_weights": (rows, top), "expert_g": (rows, top, hidden),
             "expert_u": (rows, top, hidden), "expert_out": (rows, top, width),
             "expert_activation": (rows, top, hidden), "shared_activation": (rows, hidden)}
    if grouped:
        specs.update(counts=(config.experts,), routes=(config.experts, rows))
    else:
        specs.update(experts=(rows * top,), routes=(rows * top, rows))
    return {name: graph.buffer(prefix + name, shape,
                              "int32" if name in ("ids", "experts", "routes", "counts") else
                              "bfloat16" if name.endswith("activation") else "float32")
            for name, shape in specs.items()}
