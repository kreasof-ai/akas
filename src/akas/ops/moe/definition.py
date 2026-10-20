"""Contracts and implementation choices for moe."""

from akas.kernels.contracts import OutputABI
from akas.kernels.contracts import Implementation
from akas.ops.moe.validation import pipeline_validation as pipeline_validation
from akas.ops.moe.validation import VALIDATIONS


def routed_moe(
    graph,
    x,
    logits,
    bias,
    weights,
    *,
    top,
    active,
    prefix,
    scale=1.0,
    normalize=True,
    provider="cuda",
):
    """Sigmoid/bias routing over encoded experts without a shared branch."""
    rows, width = x.shape
    experts, hidden, _ = weights["gate"].shape
    ids = graph.buffer(prefix + "ids", (rows, top), "int32")
    probability = graph.buffer(prefix + "probability", (rows, top), "float32")
    graph(
        "biased_router",
        dict(r=rows, experts=experts, top=top, scale=scale, normalize=normalize),
        logits,
        bias,
        ids,
        probability,
    )
    counts = graph.buffer(prefix + "counts", (experts,), "int32")
    routes = graph.buffer(prefix + "routes", (experts, rows), "int32")
    graph(
        "quantized_routes",
        dict(r=rows, experts=experts, top=top),
        ids,
        active,
        counts,
        routes,
    )
    tiles = ()
    if rows > 1:
        capacity = (rows * top + 3) // 4 + experts
        tiles = tuple(
            graph.buffer(prefix + name, (5, capacity), "int32")
            for name in ("tile_experts", "tile_rows")
        )
        graph("expert_tile_map", dict(r=rows, experts=experts, top=top), counts, *tiles)

    def projection(part, source, routed):
        weight = weights[part]
        out = graph.buffer(prefix + part, (rows, top, weight.shape[1]), "float32")
        routing = (ids, counts) if rows == 1 else (counts, routes, *tiles)
        graph(
            "quantized_experts",
            dict(
                r=rows,
                top=top,
                experts=experts,
                o=weight.shape[1],
                k=weight.shape[2],
                input_dtype=x.dtype,
                round_inputs=provider == "cuda",
                tile_map=rows > 1,
                routed_input=routed,
                weight_parts=weight.parts,
            ),
            source,
            *weight.buffers,
            *routing,
            out,
        )
        return out

    gate = projection("gate", x, False)
    up = projection("up", x, False)
    activation = graph.buffer(prefix + "activation", (rows, top, hidden), "float32")
    graph(
        "swiglu_experts",
        dict(
            r=rows,
            c=hidden,
            top=top,
            expert_rows=True,
            input_dtype="float32",
            output_dtype="float32",
        ),
        gate,
        up,
        activation,
    )
    expert_out = projection("down", activation, True)
    out = graph.buffer(prefix + "out", (rows, width), "float32")
    graph(
        "routed_combine",
        dict(r=rows, top=top, c=width),
        expert_out,
        probability,
        active,
        out,
    )
    return out


def encoded_moe(graph, x, weights, *, config, prefix, active):
    """Route tokens into packed experts and the shared gated expert."""
    from akas.ops.linear.definition import project

    rows, width = x.shape
    top, experts, hidden = config.top_k, config.experts, config.expert_width
    router = project(
        graph, x, weights[prefix + "gate.weight"], width=width, name=prefix + "router"
    )
    shared_gate = project(
        graph,
        x,
        weights[prefix + "shared_expert_gate.weight"],
        width=width,
        name=prefix + "shared_gate",
    )
    ids = graph.buffer(prefix + "ids", (rows, top), "int32")
    probability = graph.buffer(prefix + "probability", (rows, top), "float32")
    graph("router", dict(r=rows, experts=experts, top=top), router, ids, probability)
    counts = graph.buffer(prefix + "counts", (experts,), "int32")
    routes = graph.buffer(prefix + "routes", (experts, rows), "int32")
    graph(
        "quantized_routes",
        dict(r=rows, experts=experts, top=top),
        ids,
        active,
        counts,
        routes,
    )

    def project_expert(part, source, columns, depth, routed):
        weight = weights[prefix + "experts." + part + "_proj.weight"]
        out = graph.buffer(prefix + "expert_" + part, (rows, top, columns), "float32")
        routing = (ids, counts) if rows == 1 else (counts, routes)
        graph(
            "quantized_experts",
            dict(
                r=rows,
                top=top,
                experts=experts,
                o=columns,
                k=depth,
                routed_input=routed,
                weight_parts=weight.parts,
            ),
            source,
            *weight.buffers,
            *routing,
            out,
        )
        return out

    gate = project_expert("gate", x, hidden, width, False)
    up = project_expert("up", x, hidden, width, False)
    activated = graph.buffer(
        prefix + "expert_activation", (rows, top, hidden), "bfloat16"
    )
    graph("swiglu_experts", dict(r=rows, top=top, c=hidden), gate, up, activated)
    expert_out = project_expert("down", activated, width, hidden, True)
    shared = project(
        graph,
        x,
        weights[prefix + "shared_expert.input.weight"],
        width=width,
        name=prefix + "shared_projection",
    )
    shared_activation = graph.buffer(
        prefix + "shared_activation", (rows, hidden), "bfloat16"
    )
    graph("swiglu", dict(r=rows, c=hidden), shared, shared_activation)
    shared_out = project(
        graph,
        shared_activation,
        weights[prefix + "shared_expert.down_proj.weight"],
        width=width,
        name=prefix + "shared_output",
    )
    out = graph.buffer(prefix + "output", (rows, width), "float32")
    graph(
        "moe_combine",
        dict(r=rows, top=top, c=width),
        expert_out,
        probability,
        shared_out,
        shared_gate,
        out,
    )
    return out


def fp8_decode_moe(ctx, block):
    prefix = block.feed_forward_prefix
    ctx.bf16(prefix + "gate", ctx.buffers["normal"], ctx.buffers["router"])
    ctx.bf16(
        prefix + "shared_expert_gate", ctx.buffers["normal"], ctx.buffers["shared_gate"]
    )
    ctx.emit(
        "router",
        dict(r=ctx.rows, experts=256, top=8),
        ctx.buffers["router"],
        ctx.buffers["ids"],
        ctx.buffers["router_weights"],
    )
    ctx.emit(
        "moe_groups",
        dict(r=ctx.rows, top=8),
        ctx.buffers["ids"],
        ctx.buffers["experts"],
        ctx.buffers["routes"],
    )
    for name, dest in (("gate_proj", "expert_g"), ("up_proj", "expert_u")):
        ctx.emit(
            "fp8_experts",
            dict(r=ctx.rows, experts=256, top=8, o=512, k=2048, routed_input=False),
            ctx.buffers["normal"],
            ctx.weights[prefix + "experts." + name + ".weight"],
            ctx.weights[prefix + "experts." + name + ".weight_scale_inv"],
            ctx.buffers["experts"],
            ctx.buffers["routes"],
            ctx.buffers[dest],
        )
    ctx.emit(
        "swiglu_experts",
        dict(r=ctx.rows, top=8, c=512),
        ctx.buffers["expert_g"],
        ctx.buffers["expert_u"],
        ctx.buffers["expert_activation"],
    )
    ctx.emit(
        "fp8_experts",
        dict(r=ctx.rows, experts=256, top=8, o=2048, k=512, routed_input=True),
        ctx.buffers["expert_activation"],
        ctx.weights[prefix + "experts.down_proj.weight"],
        ctx.weights[prefix + "experts.down_proj.weight_scale_inv"],
        ctx.buffers["experts"],
        ctx.buffers["routes"],
        ctx.buffers["expert_out"],
    )
    for name, dest in (("gate_proj", "shared_g"), ("up_proj", "shared_u")):
        ctx.fp8(
            prefix + "shared_expert." + name, ctx.buffers["normal"], ctx.buffers[dest]
        )
    ctx.emit(
        "swiglu",
        dict(r=ctx.rows, c=512),
        ctx.buffers["shared_g"],
        ctx.buffers["shared_u"],
        ctx.buffers["shared_activation"],
    )
    ctx.fp8(
        prefix + "shared_expert.down_proj",
        ctx.buffers["shared_activation"],
        ctx.buffers["shared_out"],
    )
    ctx.emit(
        "moe_combine",
        dict(r=ctx.rows, top=8, c=2048),
        ctx.buffers["expert_out"],
        ctx.buffers["router_weights"],
        ctx.buffers["shared_out"],
        ctx.buffers["shared_gate"],
        ctx.buffers["ffn"],
        label=(block.index, "mlp"),
    )


def fp8_prefill_moe(ctx, block):
    prefix = block.feed_forward_prefix
    ctx.bf16(prefix + "gate", "normal", "router")
    ctx.bf16(prefix + "shared_expert_gate", "normal", "shared_gate")
    ctx.emit(
        "router",
        dict(r=ctx.rows, experts=256, top=8),
        logits=ctx.buffers["router"],
        ids=ctx.buffers["ids"],
        weights=ctx.buffers["router_weights"],
    )
    ctx.small(
        "expert_routes",
        ids=ctx.buffers["ids"],
        active=ctx.buffers["flat_active"],
        counts=ctx.buffers["counts"],
        routes=ctx.buffers["routes"],
    )
    for name, out in (("gate_proj", "expert_g"), ("up_proj", "expert_u")):
        ctx.experts(prefix + "experts." + name, "normal", out, 512, 2048, False)
    ctx.emit(
        "swiglu_experts",
        dict(r=ctx.rows, top=8, c=512),
        g=ctx.buffers["expert_g"],
        u=ctx.buffers["expert_u"],
        out=ctx.buffers["expert_activation"],
    )
    ctx.experts(
        prefix + "experts.down_proj", "expert_activation", "expert_out", 2048, 512, True
    )
    for name, out in (("gate_proj", "shared_g"), ("up_proj", "shared_u")):
        ctx.fp8(prefix + "shared_expert." + name, "normal", out)
    ctx.emit(
        "swiglu",
        dict(r=ctx.rows, c=512),
        g=ctx.buffers["shared_g"],
        u=ctx.buffers["shared_u"],
        out=ctx.buffers["shared_activation"],
    )
    ctx.fp8(prefix + "shared_expert.down_proj", "shared_activation", "shared_out")
    ctx.emit(
        "moe_combine",
        dict(r=ctx.rows, top=8, c=2048),
        experts=ctx.buffers["expert_out"],
        weights=ctx.buffers["router_weights"],
        shared=ctx.buffers["shared_out"],
        gate=ctx.buffers["shared_gate"],
        out=ctx.buffers["ffn"],
    )


def _expert_swiglu_groups(graph):
    """Fuse only private, single-token projection pairs consumed by SwiGLU."""
    from collections import Counter
    from akas.graph import Call

    references = Counter(
        name
        for call in graph.calls
        for name in (*call.arguments, *call.bindings.values())
    )
    exposed = {b.alias_of for b in graph.buffers.values() if b.alias_of is not None}
    if graph.request:
        exposed.update(graph.request["inputs"].values())
        exposed.update(graph.request.get("features", {}).values())
        exposed.update(graph.request.get("reset", ()))
        exposed.add(graph.request["prediction"])

    def fields(call, names):
        if call.bindings:
            return dict(call.bindings) if set(call.bindings) == set(names) else None
        return (
            dict(zip(names, call.arguments))
            if len(call.arguments) == len(names)
            else None
        )

    def simple(parts, rows):
        return (
            len(parts) == 1
            and parts[0]["rows"] == rows
            and not parts[0].get("row_order")
            and not parts[0].get("column_order")
        )

    groups = {}
    for index in range(len(graph.calls) - 2):
        gate, up, activation = graph.calls[index : index + 3]
        if (gate.kind, up.kind, activation.kind) != (
            "quantized_experts",
            "quantized_experts",
            "swiglu_experts",
        ):
            continue
        p, q, a = gate.parameters, up.parameters, activation.parameters
        keys = ("r", "top", "experts", "o", "k", "input_dtype", "round_inputs")
        if not all(k in p and k in q for k in (*keys, "weight_parts")):
            continue
        if (
            p.get("r") != 1
            or p.get("input_dtype") != "float32"
            or p.get("routed_input")
            or q.get("routed_input")
            or any(p.get(k) != q.get(k) for k in keys)
            or not a.get("expert_rows")
            or a.get("round_inputs", False)
            or (a.get("r"), a.get("top"), a.get("c")) != (1, p["top"], p["o"])
            or a.get("input_dtype") != "float32"
            or a.get("output_dtype") != "float32"
            or not simple(p["weight_parts"], p["experts"] * p["o"])
            or not simple(q["weight_parts"], p["experts"] * p["o"])
        ):
            continue
        g = fields(gate, ("x", "w", "ids", "counts", "out"))
        u = fields(up, ("x", "w", "ids", "counts", "out"))
        act = fields(activation, ("g", "u", "out"))
        if not g or not u or not act:
            continue
        if (
            any(g[k] != u[k] for k in ("x", "ids", "counts"))
            or (act["g"], act["u"]) != (g["out"], u["out"])
            or any(
                name in exposed
                or references[name] != 2
                or graph.buffers[name].role != "scratch"
                for name in (g["out"], u["out"])
            )
        ):
            continue
        parameters = {k: p[k] for k in keys}
        parameters.update(
            weight_parts=p["weight_parts"], up_weight_parts=q["weight_parts"]
        )
        groups[index] = Call(
            "expert_swiglu",
            parameters,
            (g["x"], g["w"], u["w"], g["ids"], g["counts"], act["out"]),
            label=activation.label if activation.label is not None else gate.label,
        )
    return groups


def search_policies(graph, commit, *, provider, target):
    if graph.contract == "reference":
        return [{"fused_swiglu": True}] if _expert_swiglu_groups(graph) else []
    if (
        graph.contract != "block-fp8"
        or provider != "cuda"
        or not any(call.kind == "fp8_experts" for call in graph.calls)
    ):
        return []
    return [{"compact": True}, {"partitions": 2}, {"resident_weights": True}]


def lower(graph, commit, settings, *, provider, target):
    from akas.graph import Graph
    from akas.quantization import get

    allowed = {
        "fused_swiglu",
        "compact",
        "partitions",
        "columns",
        "block_m",
        "threads",
        "stages",
        "hopper",
        "packed_gather",
        "bf16_mma",
        "mma_reduction",
        "mma_reorder",
        "async_mma",
        "packed_widen",
        "persistent_tiles",
        "resident_weights",
        "fixed_rows",
    }
    if set(settings) - allowed:
        raise ValueError("unknown routed projection policy")
    if "fused_swiglu" in settings and type(settings["fused_swiglu"]) is not bool:
        raise ValueError("expert SwiGLU fusion must be a boolean")
    if settings.get("fused_swiglu") and graph.contract != "reference":
        raise ValueError("expert SwiGLU fusion requires the reference contract")
    if graph.contract == "reference" and settings.get("fused_swiglu"):
        groups = _expert_swiglu_groups(graph)
        calls, index = [], 0
        while index < len(graph.calls):
            if index in groups:
                calls.append(groups[index])
                index += 3
            else:
                calls.append(graph.calls[index])
                index += 1
        graph.calls = calls
        return graph, commit
    if graph.contract != "block-fp8":
        return (graph, commit)
    if provider != "cuda":
        raise ValueError("native routed pipelines require CUDA")
    if settings.get("hopper") and target != "sm_90a":
        raise ValueError("Hopper routed policy requires an sm_90a target")
    fixed = settings.get("fixed_rows", False)
    if fixed and any(
        (settings.get(name) for name in ("hopper", "compact", "resident_weights"))
    ):
        raise ValueError("fixed row groups use the native grouped projection layout")
    recorder = Graph(contract=graph.contract, phase=graph.phase)
    recorder.buffers = graph.buffers
    calls = []
    maps = {}
    decode_routes = None
    quantized = {}
    for index, call in enumerate(graph.calls):
        if fixed and call.kind in ("moe_groups", "expert_routes"):
            ids = call.arguments[0] if call.arguments else call.bindings["ids"]
            count, top = graph.buffers[ids].shape
            if count > 16:
                raise ValueError("fixed row groups support at most sixteen rows")
            experts = graph.buffer(
                f"pipeline.fixed_experts.{index}", (count * top,), "int32"
            )
            routes = graph.buffer(
                f"pipeline.fixed_routes.{index}", (count * top, count), "int32"
            )
            recorder(
                "moe_groups",
                dict(r=count, top=top, fixed_rows=True),
                graph.buffers[ids],
                experts,
                routes,
            )
            calls.extend(recorder.calls)
            recorder.calls = []
            decode_routes = (experts, routes)
            quantized.clear()
            continue
        if call.kind == "moe_groups":
            count, top = (call.parameters["r"], call.parameters["top"])
            ids = call.arguments[0] if call.arguments else call.bindings["ids"]
            counts = graph.buffer(f"pipeline.counts.{index}", (256,), "int32")
            routes = graph.buffer(f"pipeline.routes.{index}", (256, count), "int32")
            active = graph.buffers[graph.request["inputs"]["lengths"]]
            recorder(
                "expert_routes",
                dict(slots=count, chunk=1, bucket_policy=True),
                ids=graph.buffers[ids],
                active=active,
                counts=counts,
                routes=routes,
            )
            calls.extend(recorder.calls)
            recorder.calls = []
            decode_routes = (counts, routes)
            maps.clear()
            quantized.clear()
            continue
        if call.kind != "fp8_experts":
            calls.append(call)
            quantized.clear()
            if call.kind == "expert_routes":
                maps.clear()
            continue
        p = {
            **call.parameters,
            "compact": False,
            "partitions": 1,
            "block_m": 16,
            "columns": 64,
            "threads": 128,
            "stages": 2,
            **settings,
        }
        if p.get("hopper"):
            p["rows"] = p["r"]
        parts = p["partitions"]
        if type(parts) is not int or parts < 1 or p["k"] % (128 * parts):
            raise ValueError("expert partitions must divide block-128 depth")
        names = (
            dict(zip(("x", "w", "scales", "experts", "routes", "out"), call.arguments))
            if call.arguments
            else dict(call.bindings)
        )
        if fixed:
            if decode_routes is None:
                raise ValueError("fixed projections need a matching group declaration")
            experts, routes = decode_routes
            names.pop("counts", None)
            names.update(experts=experts.name, routes=routes.name)
        elif "counts" not in names:
            if decode_routes is None:
                raise ValueError("routed pipeline needs a matching bucket declaration")
            counts, routes = decode_routes
            names.pop("experts", None)
            names.update(counts=counts.name, routes=routes.name)
        if "activation_scales" not in names:
            source = names["x"]
            if source not in quantized:
                shape = graph.buffers[source].shape
                storage = get("activations", "fp8").storage(shape)
                bits = graph.buffer(
                    f"pipeline.expert_bits.{index}",
                    storage.values.shape,
                    storage.values.dtype,
                )
                scales = graph.buffer(
                    f"pipeline.expert_scales.{index}",
                    storage.metadata[0].shape,
                    storage.metadata[0].dtype,
                )
                parameters = dict(
                    r=p["r"],
                    k=p["k"],
                    **{"top": p["top"]} if p.get("routed_input") else {},
                )
                recorder(
                    "quantize",
                    parameters,
                    x=graph.buffers[source],
                    out=bits,
                    scales=scales,
                )
                quantized[source] = (bits, scales)
            bits, scales = quantized[source]
            names.update(x=bits.name, activation_scales=scales.name)
        if p["compact"]:
            if parts > 1 and (not p.get("hopper")):
                raise ValueError("compact split experts require Hopper")
            identity = (names["counts"], p["block_m"])
            if identity not in maps:
                maximum = (
                    (p["r"] * p["top"] + p["block_m"] - 1) // p["block_m"]
                    + p["experts"]
                    - 1
                )
                tile_experts = graph.buffer(
                    f"pipeline.expert_tiles.{index}", (maximum,), "int32"
                )
                tile_offsets = graph.buffer(
                    f"pipeline.expert_offsets.{index}", (maximum,), "int32"
                )
                recorder(
                    "expert_tiles",
                    dict(r=p["r"], block_m=p["block_m"]),
                    counts=graph.buffers[names["counts"]],
                    tile_experts=tile_experts,
                    tile_offsets=tile_offsets,
                )
                maps[identity] = (tile_experts, tile_offsets)
            names.update(
                zip(
                    ("tile_experts", "tile_offsets"),
                    (value.name for value in maps[identity]),
                )
            )
        if p.get("resident_weights"):
            from dataclasses import replace

            weight = graph.buffers[names["w"]]
            graph.buffers[weight.name] = replace(weight, dtype="float16")
            object.__setattr__(graph.buffers[weight.name], "_graph", graph)
        output = graph.buffers[names["out"]]
        partial = (
            output
            if parts == 1
            else graph.buffer(
                f"pipeline.expert_partial.{index}",
                (p["r"], p["top"], parts, p["o"]),
                "float32",
            )
        )
        recorder(
            "expert_partial",
            p,
            **{
                name: graph.buffers[value]
                for name, value in {**names, "out": partial.name}.items()
            },
            label=call.label if parts == 1 else None,
        )
        if parts > 1:
            recorder(
                "expert_merge",
                dict(r=p["r"], top=p["top"], o=p["o"], partitions=parts),
                partial=partial,
                out=output,
                label=call.label,
            )
        calls.extend(recorder.calls)
        recorder.calls = []
    graph.calls = calls
    return (graph, commit)


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
    width, hidden, top = (config.width, config.expert_width, config.top_k)
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


OUTPUT_ARGUMENTS = {}
for _contract in ("reference", "packed", "packed-baseline", "bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update(
        {
            "router": OutputABI((1, 2), ("ids", "weights")),
            "moe_groups": OutputABI((1, 2), ("experts", "routes")),
            "expert_routes": OutputABI((1, 2), ("counts", "routes")),
            "fp8_experts": OutputABI((-1,), ("out",)),
            "moe_combine": OutputABI((-1,), ("out",)),
            "expert_partial": OutputABI((-1,), ("out",)),
            "expert_merge": OutputABI((-1,), ("out",)),
        }
    )
OUTPUT_ARGUMENTS["block-fp8"]["expert_tiles"] = OutputABI(
    (1, 2), ("tile_experts", "tile_offsets")
)
VALIDATIONS.update(
    {
        name: pipeline_validation
        for name in ("expert_tiles", "expert_partial", "expert_merge")
    }
)
IMPLEMENTATIONS = (
    Implementation(
        "moe_groups",
        "akas.ops.moe.cuda",
        "block-fp8",
        phase="decode",
        factory="decode_moe_groups",
        name="moe.cuda.decode",
    ),
    Implementation(
        "fp8_experts",
        "akas.ops.moe.cuda",
        "block-fp8",
        phase="decode",
        factory="decode_fp8_experts",
        name="moe.cuda.decode",
    ),
    Implementation(
        "router",
        "akas.ops.moe.cuda",
        "block-fp8",
        factory="decode_router",
        name="moe.cuda.decode",
    ),
    Implementation(
        "moe_combine",
        "akas.ops.moe.cuda",
        "block-fp8",
        factory="decode_moe_combine",
        name="moe.cuda.decode",
    ),
    Implementation(
        "fp8_experts",
        "akas.ops.moe.cuda",
        "block-fp8",
        phase="prefill",
        defaults=(("block_m", 16), ("threads", 128)),
        space=(
            ("block_m", (16, 32, 64)),
            ("columns", (64, 32, 128)),
            ("threads", (128, 256)),
        ),
        factory="prefill_fp8_experts",
        name="moe.cuda.prefill",
    ),
    Implementation(
        "expert_routes",
        "akas.ops.moe.cuda",
        "block-fp8",
        phase="prefill",
        required=("slots", "chunk"),
        factory="prefill_expert_routes",
        name="moe.cuda.prefill",
    ),
    Implementation(
        "expert_partial",
        "akas.ops.moe.cuda",
        "block-fp8",
        space=(
            ("columns", (64, 32, 128)),
            ("threads", (128, 256)),
            ("stages", (2, 1, 3)),
        ),
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
        factory="bucketed",
        name="moe.pipeline",
        accepts=lambda p: (
            not p.get("fixed_rows")
            and (not p.get("hopper"))
            and (p.get("partitions", 1) == 1)
        ),
    ),
    Implementation(
        "expert_partial",
        "akas.ops.moe.cuda",
        "block-fp8",
        space=(
            ("columns", (64, 32, 128)),
            ("threads", (128, 256)),
            ("stages", (2, 1, 3)),
        ),
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
        factory="split",
        name="moe.pipeline",
        accepts=lambda p: (
            not p.get("fixed_rows")
            and (not p.get("hopper"))
            and (p.get("partitions", 1) > 1)
        ),
    ),
    Implementation(
        "expert_partial",
        "akas.ops.moe.cuda",
        "block-fp8",
        space=(
            ("columns", (64, 32, 128)),
            ("threads", (128, 256)),
            ("stages", (2, 1, 3)),
        ),
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
        factory="fp8_prequantized",
        name="moe.pipeline",
        accepts=lambda p: p.get("fixed_rows", False),
    ),
    Implementation(
        "expert_tiles",
        "akas.ops.moe.cuda",
        "block-fp8",
        factory="tiles",
        name="moe.pipeline",
    ),
    Implementation(
        "expert_merge",
        "akas.ops.linear.cuda",
        "block-fp8",
        factory="fp8",
        name="moe.pipeline",
    ),
    Implementation(
        "expert_routes",
        "akas.ops.moe.cuda",
        "block-fp8",
        priority=30,
        required=("bucket_policy",),
        factory="prefill_expert_routes",
        name="moe.cuda.prefill",
    ),
    Implementation(
        "expert_partial",
        "akas.ops.moe.cuda",
        "block-fp8",
        priority=20,
        targets=("sm_90a",),
        space=(("columns", (64, 128)), ("threads", (128, 256)), ("stages", (1, 2, 3))),
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
        accepts=lambda p: p.get("hopper", False),
        factory="hopper_experts_partial",
        name="moe.hardware.hopper.experts",
    ),
    Implementation(
        "moe_groups",
        "akas.ops.moe.cuda",
        "block-fp8",
        priority=30,
        required=("fixed_rows",),
        variant="fixed-rows",
        factory="decode_moe_groups",
        name="moe.cuda.decode:fixed-rows",
    ),
)


IMPLEMENTATIONS += (
    Implementation(
        "router",
        "akas.ops.moe.cuda",
        "bf16",
        factory="decode_router",
        name="moe.cuda.encoded:router",
    ),
    Implementation(
        "moe_combine",
        "akas.ops.moe.cuda",
        "bf16",
        factory="decode_moe_combine",
        name="moe.cuda.encoded:combine",
    ),
    Implementation(
        "quantized_experts",
        "akas.ops.moe.cuda",
        "bf16",
        factory="quantized_experts",
        defaults=(("block_m", 16),),
        space=(("block_m", (16, 32)),),
        semantic_keys=("r", "top", "experts", "o", "k", "routed_input", "weight_parts"),
    ),
)
OUTPUT_ARGUMENTS["bf16"].update(
    {
        "quantized_routes": OutputABI((2, 3), ("counts", "routes")),
        "quantized_experts": OutputABI((-1,), ("out",)),
    }
)
from .validation import encoded_validation

VALIDATIONS.update(
    {name: encoded_validation for name in ("quantized_routes", "quantized_experts")}
)


IMPLEMENTATIONS += tuple(
    Implementation(
        "quantized_routes",
        "akas.ops.moe.kernel",
        "bf16",
        provider=provider,
        factory="quantized_routes",
    )
    for provider in ("cuda", "webgpu")
)

for _contract in ("reference",):
    OUTPUT_ARGUMENTS[_contract].update(
        {
            "biased_router": OutputABI((2, 3), ("ids", "weights")),
            "routed_combine": OutputABI((-1,), ("out",)),
            "quantized_routes": OutputABI((2, 3), ("counts", "routes")),
            "quantized_experts": OutputABI((-1,), ("out",)),
            "expert_tile_map": OutputABI((1, 2), ("tile_experts", "tile_rows")),
        }
    )

from .validation import routed_validation, tile_validation

VALIDATIONS.update(
    {name: routed_validation for name in ("biased_router", "routed_combine")}
)
VALIDATIONS["expert_tile_map"] = tile_validation
IMPLEMENTATIONS += tuple(
    Implementation(
        kind, "akas.ops.moe.kernel", "reference", provider=provider, factory=kind
    )
    for provider in ("cuda", "webgpu")
    for kind in (
        "biased_router",
        "routed_combine",
        "quantized_routes",
        "expert_tile_map",
    )
)
IMPLEMENTATIONS += (
    Implementation(
        "quantized_experts",
        "akas.ops.moe.cuda",
        "reference",
        factory="quantized_experts",
        accepts=lambda p: not p.get("tile_map", False),
        defaults=(("block_m", 16), ("columns", 64), ("block_k", 32)),
        space=(
            ("block_m", (16, 32)),
            ("columns", (32, 64, 128)),
            ("block_k", (32, 64, 128)),
        ),
    ),
    Implementation(
        "quantized_experts",
        "akas.ops.moe.kernel",
        "reference",
        provider="webgpu",
        factory="portable_experts",
        accepts=lambda p: not p.get("tile_map", False),
        defaults=(("columns", 4), ("block_k", 128)),
        space=(("columns", (4, 8)), ("block_k", (128, 256))),
    ),
)

IMPLEMENTATIONS += (
    Implementation(
        "quantized_experts",
        "akas.ops.moe.webgpu",
        "reference",
        provider="webgpu",
        factory="register_experts",
        priority=40,
        accepts=lambda p: p["r"] > 1 and p.get("tile_map", False),
        defaults=(("block_m", 16), ("columns", 32), ("block_k", 32), ("rhs_pad", 1)),
        space=(
            ("block_m", (16, 32, 8, 4)),
            ("columns", (32, 64)),
            ("block_k", (32, 64)),
            ("rhs_pad", (1, 0, 4)),
            ("dot_width", (4, 1)),
            ("micro_cols", (2, 1)),
            ("lhs_transpose", (True, False)),
            ("lhs_pad", (1, 0)),
        ),
    ),
    Implementation(
        "quantized_experts",
        "akas.ops.moe.cuda",
        "reference",
        factory="compact_experts",
        priority=40,
        accepts=lambda p: p["r"] > 1 and p.get("tile_map", False),
        defaults=(("block_m", 16), ("columns", 64), ("block_k", 32), ("threads", 128)),
        space=(
            ("block_m", (16, 32, 64)),
            ("columns", (32, 64, 128)),
            ("block_k", (32, 64, 128)),
            ("threads", (128, 256)),
        ),
    ),
    Implementation(
        "quantized_experts",
        "akas.ops.moe.cuda",
        "reference",
        factory="quantized_experts_warp",
        defaults=(("columns_per_warp", 1), ("threads", 128), ("unroll", 1)),
        space=(
            ("columns_per_warp", (1, 2, 4)),
            ("threads", (128, 256)),
            ("unroll", (1, 2, 4)),
        ),
        priority=30,
        accepts=lambda p: p["r"] == 1,
    ),
    Implementation(
        "quantized_experts",
        "akas.ops.moe.webgpu",
        "reference",
        provider="webgpu",
        factory="warp_experts",
        priority=30,
        accepts=lambda p: p["r"] == 1,
    ),
    Implementation(
        "quantized_experts",
        "akas.ops.moe.webgpu",
        "reference",
        provider="webgpu",
        factory="tiled_experts",
        priority=30,
        accepts=lambda p: p["r"] > 1 and not p.get("tile_map", False),
        defaults=(("block_m", 8), ("columns", 32), ("block_k", 32), ("rhs_pad", 1)),
        space=(
            ("block_m", (4, 8, 16, 32)),
            ("columns", (32, 64)),
            ("block_k", (32, 64)),
            ("rhs_pad", (0, 1, 4)),
        ),
    ),
)


from akas.quantization.weights import supports_parts

IMPLEMENTATIONS += tuple(
    Implementation(
        "quantized_experts",
        f"akas.ops.moe.{provider}",
        "reference",
        provider=provider,
        factory="packed_float_experts",
        name=f"moe.{provider}.packed_float_decode",
        priority=45,
        accepts=lambda p, provider=provider: (
            p["r"] == 1
            and p.get("input_dtype") == "float32"
            and p.get("round_inputs", True) is False
            and supports_parts(
                p["weight_parts"], p["k"], "dot_float", provider=provider
            )
        ),
        defaults=(("columns_per_warp", 1), ("unroll", 1)),
        space=(("columns_per_warp", (1, 2, 4)), ("unroll", (1, 2, 4))),
    )
    for provider in ("cuda", "webgpu")
)

from .validation import expert_swiglu_validation

OUTPUT_ARGUMENTS["reference"]["expert_swiglu"] = OutputABI((-1,), ("out",))
VALIDATIONS["expert_swiglu"] = expert_swiglu_validation

IMPLEMENTATIONS += (
    Implementation(
        "expert_swiglu",
        "akas.ops.moe.kernel",
        "reference",
        factory="expert_swiglu",
        priority=10,
        accepts=lambda p: p["r"] == 1 and p.get("input_dtype") == "float32",
        defaults=(("threads", 128),),
        space=(("threads", (64, 128, 256)),),
    ),
    Implementation(
        "expert_swiglu",
        "akas.ops.moe.webgpu",
        "reference",
        provider="webgpu",
        factory="expert_swiglu",
        priority=10,
        accepts=lambda p: p["r"] == 1 and p.get("input_dtype") == "float32",
        defaults=(("threads", 128),),
        space=(("threads", (64, 128, 256)),),
    ),
    Implementation(
        "expert_swiglu",
        "akas.ops.moe.cuda",
        "reference",
        factory="expert_swiglu",
        name="moe.cuda.packed_swiglu",
        priority=40,
        accepts=lambda p: (
            p["r"] == 1
            and p.get("input_dtype") == "float32"
            and p.get("round_inputs", False)
            and supports_parts(p["weight_parts"], p["k"], "dot_half", provider="cuda")
            and supports_parts(
                p["up_weight_parts"], p["k"], "dot_half", provider="cuda"
            )
        ),
        defaults=(("threads", 128), ("columns_per_warp", 1), ("unroll", 1)),
        space=(
            ("threads", (128, 256)),
            ("columns_per_warp", (1, 2)),
            ("unroll", (1, 2)),
        ),
    ),
)

IMPLEMENTATIONS += (
    Implementation(
        "quantized_experts",
        "akas.ops.moe.cuda",
        "reference",
        factory="packed_experts",
        name="moe.cuda.packed_decode",
        priority=40,
        defaults=(("threads", 128),),
        space=(("threads", (128, 256)),),
        accepts=lambda p: (
            p["r"] == 1
            and p.get("round_inputs", True)
            and supports_parts(p["weight_parts"], p["k"], "dot_half", provider="cuda")
        ),
    ),
)


IMPLEMENTATIONS += (
    Implementation(
        "quantized_routes",
        "akas.ops.moe.cuda",
        "reference",
        factory="parallel_routes",
        name="moe.cuda.parallel_routes",
        priority=30,
        accepts=lambda p: p["r"] > 1,
    ),
)
