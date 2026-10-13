"""Compact and split routed projection graphs share the common executor."""


def lower(graph, commit, settings, *, provider, target):
    from akas.graph import Graph

    allowed = {
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
    if graph.contract != "block-fp8":
        return graph, commit
    if provider != "cuda":
        raise ValueError("native routed pipelines require CUDA")
    if settings.get("hopper") and target != "sm_90a":
        raise ValueError("Hopper routed policy requires an sm_90a target")
    fixed = settings.get("fixed_rows", False)
    if fixed and any(
        settings.get(name) for name in ("hopper", "compact", "resident_weights")
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
            decode_routes = experts, routes
            quantized.clear()
            continue
        if call.kind == "moe_groups":
            count, top = call.parameters["r"], call.parameters["top"]
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
            decode_routes = counts, routes
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
                bits = graph.buffer(f"pipeline.expert_bits.{index}", shape, "uint8")
                scales = graph.buffer(
                    f"pipeline.expert_scales.{index}",
                    (*shape[:-1], shape[-1] // 128),
                    "float32",
                )
                parameters = dict(
                    r=p["r"],
                    k=p["k"],
                    **({"top": p["top"]} if p.get("routed_input") else {}),
                )
                recorder(
                    "quantize",
                    parameters,
                    x=graph.buffers[source],
                    out=bits,
                    scales=scales,
                )
                quantized[source] = bits, scales
            bits, scales = quantized[source]
            names.update(x=bits.name, activation_scales=scales.name)
        if p["compact"]:
            if parts > 1 and not p.get("hopper"):
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
                maps[identity] = tile_experts, tile_offsets
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
    return graph, commit
