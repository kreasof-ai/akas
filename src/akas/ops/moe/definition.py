"""Contracts and implementation choices for moe."""

from akas.kernels.contracts import OutputABI
from akas.kernels.contracts import Implementation
from akas.ops.moe.validation import pipeline_validation as pipeline_validation
from akas.ops.moe.validation import VALIDATIONS


def tile_capacity(rows, top, experts, block_m):
    """Bound nonempty expert tiles by both route count and partition overhead."""
    if any(
        type(value) is not int or value < 1 for value in (rows, top, experts, block_m)
    ):
        raise ValueError("expert tile geometry must contain positive integers")
    routes = rows * top
    return min(routes, (routes + block_m - 1) // block_m + experts - 1)


def tile_layouts(parameters):
    """Number of power-of-two row layouts declared by a shared route map."""
    maximum = parameters.get("max_block_m", 64)
    if type(maximum) is not int or maximum not in (4, 8, 16, 32, 64, 128):
        raise ValueError(
            "expert maximum row tile must be a supported power of two from 4 through 128"
        )
    return maximum.bit_length() - 2


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
    has_fp8 = any(weight.parts[0]["format"] == "fp8" for weight in weights.values())
    if rows > 1 or has_fp8:
        capacity = (rows * top + 3) // 4 + experts
        tiles = tuple(
            graph.buffer(prefix + name, (5, capacity), "int32")
            for name in ("tile_experts", "tile_rows")
        )
        graph("expert_tile_map", dict(r=rows, experts=experts, top=top), counts, *tiles)

    quantized = {}
    fp8_tiles = tiles if has_fp8 else ()

    def projection(part, source, routed):
        weight = weights[part]
        out = graph.buffer(prefix + part, (rows, top, weight.shape[1]), "float32")
        if (
            len(weight.parts) == 1
            and weight.parts[0]["format"] == "fp8"
            and not weight.parts[0].get("row_order")
            and not weight.parts[0].get("column_order")
        ):
            if (
                provider != "cuda"
                or weight.parts[0]["options"].get("scale_layout", "block") != "block"
            ):
                raise ValueError(
                    "native routed FP8 currently requires CUDA block scales"
                )
            from akas.quantization import get

            if source.name not in quantized:
                storage = get("activations", "fp8").storage(
                    source.shape, provider=provider
                )
                encoded = graph.buffer(
                    source.name + ".fp8", storage.values.shape, storage.values.dtype
                )
                scales = graph.buffer(
                    source.name + ".fp8_scales", storage.metadata[0].shape, "float32"
                )
                graph(
                    "quantize",
                    dict(
                        r=rows,
                        k=weight.shape[2],
                        input_dtype=source.dtype,
                        **({"top": top} if routed else {}),
                    ),
                    source,
                    encoded,
                    scales,
                )
                quantized[source.name] = encoded, scales
            encoded, scales = quantized[source.name]
            graph(
                "expert_partial",
                dict(
                    r=rows,
                    rows=rows,
                    top=top,
                    experts=experts,
                    o=weight.shape[1],
                    k=weight.shape[2],
                    routed_input=routed,
                    partitions=1,
                    compact=True,
                    tile_map=True,
                ),
                x=encoded,
                activation_scales=scales,
                w=weight.buffers[0],
                scales=weight.buffers[1],
                counts=counts,
                routes=routes,
                out=out,
                **(
                    dict(zip(("tile_experts", "tile_offsets"), fp8_tiles))
                    if fp8_tiles
                    else {}
                ),
            )
            return out
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


def _fusion_scope(graph):
    """Count direct users and conservatively exclude externally visible aliases."""
    from collections import Counter

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
    exposed.update(
        name for name, spec in graph.buffers.items() if spec.role != "scratch"
    )
    return references, exposed


def _fp8_expert_swiglu_groups(graph, *, quantize_minimum_rows=1):
    """Recognize shared compact products, SwiGLU and 128-wide FP8 quantization."""
    from akas.graph import Call

    references, exposed = _fusion_scope(graph)
    names = (
        "x",
        "activation_scales",
        "w",
        "scales",
        "counts",
        "routes",
        "out",
        "tile_experts",
        "tile_offsets",
    )
    groups = {}
    for index in range(len(graph.calls) - 3):
        gate, up, activation, quantize = graph.calls[index : index + 4]
        if tuple(c.kind for c in (gate, up, activation, quantize)) != (
            "expert_partial",
            "expert_partial",
            "swiglu_experts",
            "quantize",
        ):
            continue
        p, q, a, z = (call.parameters for call in (gate, up, activation, quantize))
        if (
            not p.get("compact")
            or p.get("routed_input")
            or q != p
            or p.get("partitions", 1) != 1
            or p["o"] % 128
            or not a.get("expert_rows")
            or a.get("round_inputs", False)
            or (a.get("r"), a.get("top"), a.get("c")) != (p["r"], p["top"], p["o"])
            or a.get("input_dtype") not in ("float32", "float16", "bfloat16")
            or a.get("input_dtype") != p.get("output_dtype", "float32")
            or a.get("output_dtype") not in ("float32", "float16", "bfloat16")
            or (z.get("r"), z.get("top"), z.get("k"), z.get("input_dtype"))
            != (p["r"], p["top"], p["o"], a.get("output_dtype"))
        ):
            continue
        g = dict(gate.bindings) if gate.bindings else dict(zip(names, gate.arguments))
        u = dict(up.bindings) if up.bindings else dict(zip(names, up.arguments))
        act = (
            dict(activation.bindings)
            if activation.bindings
            else dict(zip(("g", "u", "out"), activation.arguments))
        )
        encoded = (
            dict(quantize.bindings)
            if quantize.bindings
            else dict(zip(("x", "out", "scales"), quantize.arguments))
        )
        if (
            set(g) != set(names)
            or set(u) != set(names)
            or set(act) != {"g", "u", "out"}
            or set(encoded) != {"x", "out", "scales"}
            or any(
                g[key] != u[key] for key in names if key not in ("w", "scales", "out")
            )
            or (act["g"], act["u"]) != (g["out"], u["out"])
            or encoded["x"] != act["out"]
            or any(
                name in exposed or references[name] != 2
                for name in (g["out"], u["out"], act["out"])
            )
        ):
            continue
        bindings = {key: g[key] for key in names if key != "out"}
        bindings.update(w2=u["w"], scales2=u["scales"])
        parameters = dict(p)
        if a["input_dtype"] != "float32" or a["output_dtype"] != "float32":
            parameters.update(
                product_dtype=a["input_dtype"], activation_dtype=a["output_dtype"]
            )
        if p["r"] >= quantize_minimum_rows:
            bindings.update(out=encoded["out"], output_scales=encoded["scales"])
            parameters.pop("output_dtype", None)
            groups[index] = Call(
                "fp8_expert_swiglu", parameters, (), bindings, label=activation.label
            )
        else:
            bindings["out"] = act["out"]
            groups[index] = Call(
                "expert_swiglu",
                dict(
                    parameters,
                    input_dtype="uint8",
                    weight_format="fp8",
                    output_dtype=a["output_dtype"],
                ),
                (),
                bindings,
                label=activation.label,
            )
    return groups


def _expert_swiglu_groups(graph):
    """Fuse only private, single-token projection pairs consumed by SwiGLU."""
    from akas.graph import Call

    references, exposed = _fusion_scope(graph)

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
        choices = (
            [{"fused_swiglu": True}]
            if (
                _expert_swiglu_groups(graph)
                or provider == "cuda"
                and _fp8_expert_swiglu_groups(graph)
            )
            else []
        )
        if provider == "cuda":
            groups = _fp8_expert_swiglu_groups(graph)
            if groups:
                choices.append(
                    {
                        "fused_swiglu": True,
                        "fused_quantize_minimum_rows": max(
                            c.parameters["r"] for c in groups.values()
                        )
                        + 1,
                    }
                )
        if provider == "cuda" and any(
            c.kind == "expert_partial" and c.parameters.get("compact")
            for c in graph.calls
        ):
            choices.extend({"block_m": m} for m in (32, 64))
        return choices
    if (
        graph.contract != "block-fp8"
        or provider != "cuda"
        or not any(call.kind == "fp8_experts" for call in graph.calls)
    ):
        return []
    return [{"compact": True}, {"partitions": 2}, {"resident_weights": True}]


def _contribution_storage(graph, dtype):
    """Narrow private routed contributions and their complete alias family."""
    from dataclasses import replace

    def base(name):
        while graph.buffers[name].alias_of is not None:
            name = graph.buffers[name].alias_of
        return name

    def fields(call, names):
        if call.bindings:
            return dict(call.bindings)
        return (
            dict(zip(names, call.arguments))
            if len(names) == len(call.arguments)
            else {}
        )

    request = graph.request or {}
    visible = {
        *request.get("inputs", {}).values(),
        *request.get("features", {}).values(),
        *request.get("reset", ()),
    }
    if request.get("prediction"):
        visible.add(request["prediction"])
    changed = False
    calls = list(graph.calls)
    for index, call in enumerate(calls):
        p = call.parameters
        if (
            call.kind not in ("expert_partial", "expert_merge")
            or not p.get("routed_input")
            or not p.get("compact")
            or call.kind == "expert_partial"
            and p.get("partitions", 1) != 1
        ):
            continue
        producer = fields(
            call,
            ("partial", "out")
            if call.kind == "expert_merge"
            else (
                "x",
                "activation_scales",
                "w",
                "scales",
                "counts",
                "routes",
                "out",
                "tile_experts",
                "tile_offsets",
            ),
        )
        if "out" not in producer:
            continue
        owner = base(producer["out"])
        aliases = {name for name in graph.buffers if base(name) == owner}
        if (
            visible & aliases
            or any(graph.buffers[name].role != "scratch" for name in aliases)
            or any(name in aliases for key, name in producer.items() if key != "out")
        ):
            continue
        users = [
            i
            for i, user in enumerate(calls)
            if i != index
            and aliases.intersection((*user.arguments, *user.bindings.values()))
        ]
        if len(users) != 1 or users[0] <= index:
            continue
        consumer = calls[users[0]]
        inputs = fields(consumer, ("experts", "weights", "active", "out"))
        if (
            consumer.kind != "routed_combine"
            or inputs.get("experts") not in aliases
            or any(name in aliases for key, name in inputs.items() if key != "experts")
            or tuple(consumer.parameters.get(key) for key in ("r", "top", "c"))
            != tuple(p[key] for key in ("r", "top", "o"))
        ):
            continue
        if any(graph.buffers[name].dtype not in ("float32", dtype) for name in aliases):
            continue
        for name in aliases:
            value = replace(graph.buffers[name], dtype=dtype)
            object.__setattr__(value, "_graph", graph)
            graph.buffers[name] = value
        calls[index] = replace(call, parameters={**p, "output_dtype": dtype})
        calls[users[0]] = replace(
            consumer, parameters={**consumer.parameters, "input_dtype": dtype}
        )
        changed = True
    graph.calls = calls
    return changed


def _activation_storage(graph, dtype):
    """Retype the same private product/activation chain recognized by fusion."""
    from dataclasses import replace

    # The recognizer excludes views, public values and additional consumers.
    # Keep those guards identical whether the chain is fused or left separate.
    for index in _fp8_expert_swiglu_groups(graph):
        gate, up, activation, quantize = graph.calls[index : index + 4]
        for position, call in ((index, gate), (index + 1, up), (index + 2, activation)):
            name = call.bindings["out"] if call.bindings else call.arguments[-1]
            value = replace(graph.buffers[name], dtype=dtype)
            object.__setattr__(value, "_graph", graph)
            graph.buffers[name] = value
            parameters = {**call.parameters, "output_dtype": dtype}
            if call.kind == "swiglu_experts":
                parameters["input_dtype"] = dtype
            graph.calls[position] = replace(call, parameters=parameters)
        graph.calls[index + 3] = replace(
            quantize, parameters={**quantize.parameters, "input_dtype": dtype}
        )


def _partition_experts(graph, parts, maximum_rows):
    """Keep the existing product/ordered-merge contract for compact reductions."""
    from dataclasses import replace
    from akas.graph import Call

    calls = []
    for index, call in enumerate(graph.calls):
        p = call.parameters
        if (
            call.kind not in ("expert_partial", "expert_swiglu", "fp8_expert_swiglu")
            or not p.get("compact")
            or p.get("partitions", 1) != 1
            or maximum_rows is not None
            and p["r"] > maximum_rows
        ):
            calls.append(call)
            continue
        if p["k"] % (128 * parts):
            raise ValueError("expert reduction partitions must divide block-128 depth")
        if call.kind != "expert_partial":
            fields = (
                "x",
                "activation_scales",
                "w",
                "scales",
                "w2",
                "scales2",
                "counts",
                "routes",
                "tile_experts",
                "tile_offsets",
                "out",
                "output_scales",
            )
            names = (
                dict(call.bindings)
                if call.bindings
                else dict(zip(fields, call.arguments))
            )
            shape = (p["r"], p["top"], parts, p["o"])
            gate = graph.buffer(
                f"pipeline.expert_gate_partial.{index}", shape, "float32"
            )
            up = graph.buffer(f"pipeline.expert_up_partial.{index}", shape, "float32")
            calls.append(
                Call(
                    "expert_pair_partial",
                    {**p, "partitions": parts, "output_dtype": "float32"},
                    (),
                    {
                        **{
                            key: name
                            for key, name in names.items()
                            if key not in ("out", "output_scales")
                        },
                        "gate": gate.name,
                        "up": up.name,
                    },
                )
            )
            dtype = p.get("activation_dtype", p.get("output_dtype", "float32"))
            output = (
                graph.buffer(
                    f"pipeline.expert_activation.{index}",
                    (p["r"], p["top"], p["o"]),
                    dtype,
                ).name
                if call.kind == "fp8_expert_swiglu"
                else names["out"]
            )
            calls.append(
                Call(
                    "swiglu_experts",
                    dict(
                        r=p["r"],
                        top=p["top"],
                        c=p["o"],
                        partitions=parts,
                        input_dtype="float32",
                        output_dtype=dtype,
                        round_inputs=True,
                        round_dtype=p.get("product_dtype", "float32"),
                    ),
                    (),
                    dict(g=gate.name, u=up.name, out=output),
                    label=call.label if call.kind == "expert_swiglu" else None,
                )
            )
            if call.kind == "fp8_expert_swiglu":
                calls.append(
                    Call(
                        "quantize",
                        dict(
                            r=p["r"],
                            top=p["top"],
                            k=p["o"],
                            routed_input=True,
                            input_dtype=dtype,
                        ),
                        (),
                        dict(x=output, out=names["out"], scales=names["output_scales"]),
                        label=call.label,
                    )
                )
            continue
        fields = (
            "x",
            "activation_scales",
            "w",
            "scales",
            "counts",
            "routes",
            "out",
            "tile_experts",
            "tile_offsets",
        )
        names = (
            dict(call.bindings) if call.bindings else dict(zip(fields, call.arguments))
        )
        output = names["out"]
        partial = graph.buffer(
            f"pipeline.expert_partial.{index}",
            (p["r"], p["top"], parts, p["o"]),
            "float32",
        )
        calls.append(
            replace(
                call,
                parameters={**p, "partitions": parts, "output_dtype": "float32"},
                arguments=(),
                bindings={**names, "out": partial.name},
                label=None,
            )
        )
        parameters = dict(
            r=p["r"],
            top=p["top"],
            o=p["o"],
            partitions=parts,
            output_dtype=graph.buffers[output].dtype,
            compact=True,
            routed_input=p.get("routed_input", False),
        )
        calls.append(
            Call(
                "expert_merge",
                parameters,
                (),
                {"partial": partial.name, "out": output},
                label=call.label,
            )
        )
    graph.calls = calls


def _pack_experts(graph, minimum_rows):
    """Keep a private paired/down chain in one shared routed layout."""
    from dataclasses import replace
    from akas.graph import Call

    calls = []
    families, exposed = graph.storage_families(), graph.exposed_buffers()
    index = 0
    while index < len(graph.calls):
        pair = graph.calls[index]
        down = graph.calls[index + 1] if index + 1 < len(graph.calls) else None
        p = pair.parameters
        if (
            pair.kind != "fp8_expert_swiglu"
            or p["r"] < minimum_rows
            or down is None
            or down.kind != "expert_partial"
            or p.get("partitions", 1) != 1
            or down.parameters.get("partitions", 1) != 1
            or not p.get("tile_map")
            or not down.parameters.get("tile_map")
            or not pair.bindings
            or not down.bindings
        ):
            calls.append(pair)
            index += 1
            continue
        a, b, q = pair.bindings, down.bindings, down.parameters
        keys = ("counts", "routes", "tile_experts", "tile_offsets")
        outputs = (a["out"], a["output_scales"])
        if (
            b["x"] != a["out"]
            or b["activation_scales"] != a["output_scales"]
            or any(a[key] != b[key] for key in keys)
            or tuple(p[key] for key in ("r", "top", "experts", "o"))
            != tuple(q[key] for key in ("r", "top", "experts", "k"))
            or not q.get("routed_input")
            or not p.get("compact")
            or not q.get("compact")
            or p.get("max_block_m", 64) != q.get("max_block_m", 64)
            or any(
                name in exposed or len(families[graph.storage_owner(name)]) != 1
                for name in outputs
            )
            or any(
                name in (*call.arguments, *call.bindings.values())
                for call in graph.calls[:index] + graph.calls[index + 2 :]
                for name in outputs
            )
        ):
            calls.append(pair)
            index += 1
            continue
        rows, top, experts = (p[key] for key in ("r", "top", "experts"))
        maximum = p.get("max_block_m", 64)
        capacity = tile_capacity(rows, top, experts, maximum) * maximum
        prefix = f"pipeline.expert_pack.{index}"
        packed = graph.buffer(prefix + ".values", (capacity, p["k"]), "uint8")
        scales = graph.buffer(prefix + ".scales", (capacity, p["k"] // 128), "float32")
        offsets = graph.buffer(prefix + ".offsets", (experts,), "int32")
        calls.append(
            Call(
                "expert_pack",
                dict(r=rows, top=top, experts=experts, k=p["k"], max_block_m=maximum),
                (),
                dict(
                    x=a["x"],
                    activation_scales=a["activation_scales"],
                    **{key: a[key] for key in keys},
                    packed=packed.name,
                    packed_scales=scales.name,
                    expert_offsets=offsets.name,
                ),
            )
        )
        for name, shape in (
            (a["out"], (capacity, p["o"])),
            (a["output_scales"], (capacity, p["o"] // 128)),
        ):
            value = replace(graph.buffers[name], shape=shape)
            object.__setattr__(value, "_graph", graph)
            graph.buffers[name] = value
        calls.append(
            replace(
                pair,
                parameters={**p, "packed_input": True, "packed_output": True},
                bindings={
                    **a,
                    "x": packed.name,
                    "activation_scales": scales.name,
                    "expert_offsets": offsets.name,
                },
            )
        )
        calls.append(
            replace(
                down,
                parameters={**q, "packed_input": True},
                bindings={**b, "expert_offsets": offsets.name},
            )
        )
        index += 2
    graph.calls = calls


def lower(graph, commit, settings, *, provider, target):
    from akas.graph import Graph
    from akas.quantization import get

    allowed = {
        "fused_swiglu",
        "fused_quantize_minimum_rows",
        "intermediate_dtype",
        "activation_dtype",
        "compact",
        "partitions",
        "partition_maximum_rows",
        "packed",
        "packed_minimum_rows",
        "columns",
        "block_m",
        "max_block_m",
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
    if settings.get("intermediate_dtype", "float32") not in (
        "float32",
        "float16",
        "bfloat16",
    ):
        raise ValueError(
            "expert intermediate dtype must be float32, float16 or bfloat16"
        )
    if "intermediate_dtype" in settings and (
        provider != "cuda" or graph.contract != "reference"
    ):
        raise ValueError(
            "expert intermediate storage currently requires CUDA reference arithmetic"
        )
    if settings.get("activation_dtype", "float32") not in (
        "float32",
        "float16",
        "bfloat16",
    ):
        raise ValueError("expert activation dtype must be float32, float16 or bfloat16")
    if "activation_dtype" in settings and (
        provider != "cuda" or graph.contract != "reference"
    ):
        raise ValueError(
            "expert activation storage currently requires CUDA reference arithmetic"
        )
    minimum = settings.get("fused_quantize_minimum_rows", 1)
    if type(minimum) is not int or minimum < 1:
        raise ValueError("expert quantization minimum rows must be a positive integer")
    if "fused_quantize_minimum_rows" in settings and not settings.get("fused_swiglu"):
        raise ValueError("expert quantization minimum rows requires SwiGLU fusion")
    if settings.get("fused_swiglu") and graph.contract != "reference":
        raise ValueError("expert SwiGLU fusion requires the reference contract")
    if type(settings.get("packed", False)) is not bool:
        raise ValueError("expert packing must be a boolean")
    packed_minimum = settings.get("packed_minimum_rows", 1)
    if type(packed_minimum) is not int or packed_minimum < 1:
        raise ValueError("expert packing minimum rows must be positive")
    if settings.get("packed") and (
        provider != "cuda"
        or graph.contract != "reference"
        or settings.get("partitions", 1) != 1
    ):
        raise ValueError("expert packing requires complete CUDA reference products")
    if "max_block_m" in settings:
        tile_layouts(settings)
        if graph.contract != "reference":
            raise ValueError(
                "shared expert tile domains require the reference contract"
            )
    if graph.contract == "reference":
        parts = settings.get("partitions", 1)
        maximum_rows = settings.get("partition_maximum_rows")
        if type(parts) is not int or parts < 1:
            raise ValueError("expert reduction partitions must be a positive integer")
        if maximum_rows is not None and (
            type(maximum_rows) is not int or maximum_rows < 1
        ):
            raise ValueError("expert partition maximum rows must be a positive integer")
        if parts > 1 and provider != "cuda":
            raise ValueError(
                "compact expert reduction partitions currently require CUDA"
            )
    if settings.get("activation_dtype", "float32") != "float32":
        _activation_storage(graph, settings["activation_dtype"])
    if graph.contract == "reference" and settings.get("fused_swiglu"):
        groups = _expert_swiglu_groups(graph)
        fp8_groups = (
            _fp8_expert_swiglu_groups(graph, quantize_minimum_rows=minimum)
            if provider == "cuda"
            else {}
        )
        calls, index = [], 0
        while index < len(graph.calls):
            if index in fp8_groups:
                calls.append(fp8_groups[index])
                index += 4 if fp8_groups[index].kind == "fp8_expert_swiglu" else 3
            elif index in groups:
                calls.append(groups[index])
                index += 3
            else:
                calls.append(graph.calls[index])
                index += 1
        graph.calls = calls
        if not (
            set(settings)
            & {
                "block_m",
                "max_block_m",
                "intermediate_dtype",
                "activation_dtype",
                "partitions",
                "packed",
                "packed_minimum_rows",
            }
        ):
            return graph, commit
    if graph.contract == "reference":
        if set(settings) - {
            "fused_swiglu",
            "fused_quantize_minimum_rows",
            "block_m",
            "max_block_m",
            "intermediate_dtype",
            "activation_dtype",
            "partitions",
            "partition_maximum_rows",
            "packed",
            "packed_minimum_rows",
        }:
            raise ValueError("unsupported reference expert policy")
        if "block_m" in settings or "max_block_m" in settings:
            _retile_experts(
                graph, settings.get("block_m"), max_block_m=settings.get("max_block_m")
            )
        if settings.get("packed"):
            _pack_experts(graph, packed_minimum)
        if parts > 1:
            _partition_experts(graph, parts, maximum_rows)
        if settings.get("intermediate_dtype", "float32") != "float32":
            _contribution_storage(graph, settings["intermediate_dtype"])
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
                maximum = tile_capacity(p["r"], p["top"], p["experts"], p["block_m"])
                tile_experts = graph.buffer(
                    f"pipeline.expert_tiles.{index}", (maximum,), "int32"
                )
                tile_offsets = graph.buffer(
                    f"pipeline.expert_offsets.{index}", (maximum,), "int32"
                )
                recorder(
                    "expert_tiles",
                    dict(
                        r=p["r"],
                        experts=p["experts"],
                        top=p["top"],
                        block_m=p["block_m"],
                    ),
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


def _retile_experts(graph, block_m, *, max_block_m=None):
    """Retile existing compact FP8 calls and their shared routing tables."""
    from dataclasses import replace

    if block_m is not None and (
        type(block_m) is not int or block_m not in (16, 32, 64, 128)
    ):
        raise ValueError("compact expert block_m must be 16, 32, 64 or 128")
    if max_block_m is not None:
        layouts = tile_layouts({"max_block_m": max_block_m})
    calls = []
    for call in graph.calls:
        p = call.parameters
        if call.kind == "expert_tile_map" and max_block_m is not None:
            names = (
                (call.bindings["tile_experts"], call.bindings["tile_rows"])
                if call.bindings
                else call.arguments[1:]
            )
            for name in names:
                spec = graph.buffers[name]
                if (
                    spec.role != "scratch"
                    or spec.alias_of is not None
                    or any(b.alias_of == name for b in graph.buffers.values())
                ):
                    raise ValueError(
                        "shared tile maps must own unaliased scratch descriptors"
                    )
                value = replace(spec, shape=(layouts, spec.shape[1]))
                object.__setattr__(value, "_graph", graph)
                graph.buffers[name] = value
            call = replace(call, parameters={**p, "max_block_m": max_block_m})
        elif call.kind == "expert_tiles" and block_m is not None:
            capacity = tile_capacity(p["r"], p["top"], p["experts"], block_m)
            names = (
                (call.bindings["tile_experts"], call.bindings["tile_offsets"])
                if call.bindings
                else call.arguments[1:]
            )
            for name in names:
                spec = graph.buffers[name]
                if spec.role != "scratch" or spec.alias_of is not None:
                    raise ValueError("compact tile maps must own scratch descriptors")
                value = replace(spec, shape=(capacity,))
                object.__setattr__(value, "_graph", graph)
                graph.buffers[name] = value
            call = replace(call, parameters={**p, "block_m": block_m})
        elif call.kind in (
            "expert_partial",
            "fp8_expert_swiglu",
            "expert_swiglu",
        ) and p.get("compact"):
            parameters = {**p, **({"block_m": block_m} if block_m is not None else {})}
            if p.get("tile_map"):
                if max_block_m is not None:
                    parameters["max_block_m"] = max_block_m
                if parameters.get("block_m", 16) > parameters.get("max_block_m", 64):
                    raise ValueError("expert row tile exceeds its shared map's maximum")
            call = replace(call, parameters=parameters)
        elif (
            call.kind == "quantized_experts"
            and p.get("tile_map")
            and max_block_m is not None
        ):
            call = replace(call, parameters={**p, "max_block_m": max_block_m})
        calls.append(call)
    graph.calls = calls


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

OUTPUT_ARGUMENTS["reference"]["fp8_expert_swiglu"] = OutputABI(
    (-2, -1), ("out", "output_scales")
)
VALIDATIONS["fp8_expert_swiglu"] = pipeline_validation
IMPLEMENTATIONS += (
    Implementation(
        "fp8_expert_swiglu",
        "akas.ops.moe.cuda",
        "reference",
        factory="fp8_expert_swiglu",
        name="moe.cuda.fp8-swiglu:reference",
        defaults=(("columns", 128),),
        accepts=lambda p: (
            p.get("compact")
            and not p.get("routed_input")
            and not p.get("tile_map")
            and p.get("partitions", 1) == 1
            and p["o"] % 128 == 0
        ),
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
            "product_dtype",
            "activation_dtype",
            "packed_input",
            "packed_output",
        ),
        space=(
            ("threads", (128, 256)),
            ("stages", (1, 2, 3)),
            ("pair_layout", ("wide", "separate")),
            ("activation_vector", (16, 8, 4, 1)),
            ("mma_dtype", ("float8_e4m3fn", "float16")),
        ),
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

IMPLEMENTATIONS += (
    Implementation(
        "quantized_experts",
        "akas.ops.moe.cuda",
        "reference",
        factory="staged_experts",
        name="moe.cuda.staged",
        priority=35,
        accepts=lambda p: (
            p["r"] > 1
            and p.get("tile_map", False)
            and p["o"] % 32 == 0
            and p["k"] % 256 == 0
            and supports_parts(p["weight_parts"], p["k"], "stage_tile", provider="cuda")
        ),
        defaults=(
            ("block_m", 16),
            ("columns", 64),
            ("block_k", 256),
            ("threads", 128),
            ("stages", 1),
        ),
        space=(
            ("block_m", (16, 32)),
            ("columns", (32, 64, 128)),
            ("block_k", (128, 256)),
            ("threads", (128, 256)),
            ("stages", (1, 2)),
        ),
    ),
)

OUTPUT_ARGUMENTS["reference"]["expert_swiglu"] = OutputABI((-1,), ("out",))
VALIDATIONS["expert_swiglu"] = expert_swiglu_validation

IMPLEMENTATIONS += (
    Implementation(
        "expert_swiglu",
        "akas.ops.moe.cuda",
        "reference",
        factory="fp8_paired_swiglu",
        name="moe.cuda.fp8-paired-swiglu:reference",
        accepts=lambda p: (
            p.get("weight_format") == "fp8"
            and p.get("input_dtype") == "uint8"
            and p.get("compact")
            and not p.get("routed_input")
            and p.get("partitions", 1) == 1
            and p["o"] % 128 == 0
            and p.get("block_m", 16) // 16 * (p.get("columns", 64) // 8)
            >= p.get("threads", 128) // 32
        ),
        defaults=(("columns", 64), ("activation_vector", 16)),
        semantic_keys=(
            "r",
            "rows",
            "k",
            "o",
            "experts",
            "top",
            "routed_input",
            "partitions",
            "compact",
            "tile_map",
            "max_block_m",
            "input_dtype",
            "weight_format",
            "output_dtype",
            "product_dtype",
            "activation_dtype",
            "packed_input",
            "packed_output",
        ),
        space=(
            ("block_m", (16, 32, 64, 128)),
            ("columns", (64, 32, 128)),
            ("threads", (128, 256, 512)),
            ("pair_layout", ("wide", "separate")),
            ("stages", (1, 2, 3)),
            ("mma_dtype", ("float8_e4m3fn", "float16")),
            ("activation_vector", (16,)),
        ),
    ),
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

OUTPUT_ARGUMENTS["reference"]["expert_partial"] = OutputABI((-1,), ("out",))
OUTPUT_ARGUMENTS["reference"]["expert_tiles"] = OutputABI(
    (1, 2), ("tile_experts", "tile_offsets")
)
IMPLEMENTATIONS += (
    Implementation(
        "expert_tile_map",
        "akas.ops.moe.cuda",
        "reference",
        factory="expert_tile_map",
        name="moe.cuda.parallel_tile_map",
        priority=30,
    ),
    Implementation(
        "expert_tiles",
        "akas.ops.moe.cuda",
        "reference",
        factory="tiles",
        name="moe.cuda.fp8-tiles:reference",
    ),
    Implementation(
        "expert_partial",
        "akas.ops.moe.cuda",
        "reference",
        factory="bucketed",
        name="moe.cuda.fp8-bucketed:reference",
        accepts=lambda p: (
            p.get("partitions", 1) == 1
            and not p.get("tile_map")
            and p.get("block_m", 16) // 16 * (p.get("columns", 64) // 8)
            >= p.get("threads", 128) // 32
        ),
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
        ),
        space=(
            ("columns", (64, 32, 128)),
            ("threads", (128, 256)),
            ("stages", (1, 2, 3)),
            ("mma_dtype", ("float8_e4m3fn", "float16")),
            ("activation_vector", (1, 16, 8, 4)),
        ),
    ),
)

# All tile layouts share one input ABI, so M is a schedule choice rather than
# a graph rewrite. The same route map serves paired and ordinary projections.
IMPLEMENTATIONS += (
    Implementation(
        "expert_partial",
        "akas.ops.moe.cuda",
        "reference",
        factory="bucketed",
        name="moe.cuda.fp8-bucketed:mapped",
        defaults=(("activation_vector", 16),),
        accepts=lambda p: (
            p.get("tile_map")
            and p.get("compact")
            and type(p.get("partitions", 1)) is int
            and p.get("partitions", 1) >= 1
            and p["k"] % (128 * p.get("partitions", 1)) == 0
            and p.get("block_m", 16) // 16 * (p.get("columns", 64) // 8)
            >= p.get("threads", 128) // 32
        ),
        semantic_keys=(
            "r",
            "rows",
            "k",
            "o",
            "experts",
            "top",
            "routed_input",
            "partitions",
            "compact",
            "tile_map",
            "max_block_m",
            "output_dtype",
            "packed_input",
        ),
        space=(
            ("block_m", (16, 32, 64, 128)),
            ("columns", (128, 64, 32)),
            ("threads", (128, 256, 512)),
            ("stages", (1, 2, 3)),
            ("mma_dtype", ("float8_e4m3fn", "float16")),
            ("activation_vector", (16,)),
        ),
    ),
    Implementation(
        "fp8_expert_swiglu",
        "akas.ops.moe.cuda",
        "reference",
        factory="fp8_expert_swiglu",
        name="moe.cuda.fp8-swiglu:mapped",
        defaults=(("columns", 128), ("activation_vector", 16)),
        accepts=lambda p: (
            p.get("tile_map")
            and p.get("compact")
            and not p.get("routed_input")
            and p.get("partitions", 1) == 1
            and p["o"] % 128 == 0
        ),
        semantic_keys=(
            "r",
            "rows",
            "k",
            "o",
            "experts",
            "top",
            "routed_input",
            "partitions",
            "compact",
            "tile_map",
            "max_block_m",
            "product_dtype",
            "activation_dtype",
            "packed_input",
            "packed_output",
        ),
        space=(
            ("block_m", (16, 32, 64, 128)),
            ("threads", (128, 256, 512)),
            ("pair_layout", ("wide", "separate")),
            ("stages", (1, 2, 3)),
            ("mma_dtype", ("float8_e4m3fn", "float16")),
            ("activation_vector", (16,)),
        ),
    ),
)

OUTPUT_ARGUMENTS["reference"]["expert_merge"] = OutputABI((-1,), ("out",))
VALIDATIONS["expert_merge"] = pipeline_validation
IMPLEMENTATIONS += (
    Implementation(
        "expert_merge",
        "akas.ops.linear.cuda",
        "reference",
        factory="fp8",
        name="moe.cuda.merge:reference",
        semantic_keys=("r", "top", "o", "partitions", "output_dtype", "routed_input"),
    ),
)

from .validation import paired_partial_validation

OUTPUT_ARGUMENTS["reference"]["expert_pair_partial"] = OutputABI(
    (-2, -1), ("gate", "up")
)
VALIDATIONS["expert_pair_partial"] = paired_partial_validation
IMPLEMENTATIONS += (
    Implementation(
        "expert_pair_partial",
        "akas.ops.moe.cuda",
        "reference",
        factory="paired_partial",
        name="moe.cuda.fp8-paired-partial:reference",
        defaults=(("columns", 128), ("activation_vector", 16)),
        accepts=lambda p: (
            p.get("compact")
            and not p.get("routed_input")
            and type(p.get("partitions", 1)) is int
            and p.get("partitions", 1) >= 1
            and p["k"] % (128 * p.get("partitions", 1)) == 0
            and p["o"] % 128 == 0
        ),
        semantic_keys=(
            "r",
            "rows",
            "k",
            "o",
            "experts",
            "top",
            "partitions",
            "compact",
            "tile_map",
            "max_block_m",
        ),
        space=(
            ("block_m", (16, 32, 64, 128)),
            ("columns", (128, 64, 32)),
            ("threads", (128, 256, 512)),
            ("pair_layout", ("wide", "separate")),
            ("stages", (1, 2, 3)),
            ("mma_dtype", ("float8_e4m3fn", "float16")),
            ("activation_vector", (16,)),
        ),
    ),
)

from .validation import pack_validation

OUTPUT_ARGUMENTS["reference"]["expert_pack"] = OutputABI(
    (-3, -2, -1), ("packed", "packed_scales", "expert_offsets")
)
VALIDATIONS["expert_pack"] = pack_validation
IMPLEMENTATIONS += (
    Implementation(
        "expert_pack",
        "akas.ops.moe.cuda",
        "reference",
        factory="pack_experts",
        name="moe.cuda.pack:reference",
        semantic_keys=("r", "k", "experts", "top", "routed_input", "max_block_m"),
    ),
)
