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
    *variants(
        ("linear",), "linear.cuda.bf16_batch", "bf16",
        semantic_keys=("r", "k", "o"),
        space=(("m", (16, 32, 64)), ("n", (64, 128)),
               ("bk", (128, 64, 256)), ("stages", (2, 1, 3))),
    ),
    *variants(
        ("gemv",), "linear.cuda.bf16_batch", "bf16",
        semantic_keys=("r", "k", "o"), space=(("columns", (16, 8, 32)),),
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


from akas.ops.linear.validation import VALIDATIONS


def fuse_projections(plan):
    """Fuse private paired projections into the established FFN operation.

    Decode with one row keeps separate projections: the established FP16 GEMV
    route is faster there. Partitioned or externally consumed intermediates
    keep their original calls. The runtime recomposes this same graph.
    """
    from collections import Counter
    from dataclasses import replace

    uses = Counter(
        name
        for call in plan.calls
        for name in (*call.arguments, *call.bindings.values())
    )
    result, index = [], 0
    while index < len(plan.calls):
        calls = plan.calls[index : index + 3]
        if len(calls) == 3:
            gate, up, activation = calls
            p = gate.parameters
            if (
                (gate.kind, up.kind, activation.kind) == ("linear", "linear", "swiglu")
                and p == up.parameters
                and p.get("optimized")
                and p["r"] > 1
                and not p.get("matrix_output")
                and p.get("output_offset", 0) == 0
                and p.get("output_stride", p["o"]) == p["o"]
                and len(gate.arguments)
                == len(up.arguments)
                == len(activation.arguments)
                == 3
                and gate.arguments[0] == up.arguments[0]
                and activation.arguments[:2] == (gate.arguments[2], up.arguments[2])
                and activation.parameters == dict(r=p["r"], c=p["o"])
                and uses[gate.arguments[2]] == uses[up.arguments[2]] == 2
            ):
                result.append(
                    replace(
                        gate,
                        kind="ffn",
                        arguments=(
                            gate.arguments[0],
                            gate.arguments[1],
                            up.arguments[1],
                            activation.arguments[2],
                        ),
                        label=activation.label,
                    )
                )
                index += 3
                continue
        result.append(plan.calls[index])
        index += 1
    plan.calls = result
    used = {
        name for call in result for name in (*call.arguments, *call.bindings.values())
    }
    plan.buffers = {
        name: spec
        for name, spec in plan.buffers.items()
        if spec.role != "scratch" or name in used
    }
    return plan


REWRITES = (fuse_projections,)
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
