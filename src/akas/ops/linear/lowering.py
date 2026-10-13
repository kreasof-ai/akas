"""Graph declarations for shared quantization and split-K projection pipelines."""


def lower(graph, commit, settings, *, provider, target):
    from akas.graph import Graph

    allowed = {
        "partitions",
        "columns",
        "block_m",
        "threads",
        "stages",
        "packed_copy",
        "hopper",
        "packed_gather",
        "mma_reduction",
        "mma_reorder",
        "async_mma",
        "packed_widen",
    }
    if set(settings) - allowed:
        raise ValueError("unknown projection pipeline policy")
    if graph.contract != "block-fp8":
        return graph, commit
    if provider != "cuda":
        raise ValueError("native FP8 projection pipelines require CUDA")
    if settings.get("hopper") and target != "sm_90a":
        raise ValueError("Hopper projection policy requires an sm_90a target")
    partitions = settings.get("partitions", 1)
    if type(partitions) is not int or partitions < 1:
        raise ValueError("projection partitions must be positive")
    recorder = Graph(contract=graph.contract, phase=graph.phase)
    recorder.buffers = graph.buffers
    quantized, valid = {}, set()
    calls = []
    for index, call in enumerate(graph.calls):
        if call.kind != "fp8_linear":
            calls.append(call)
            for name in (call.bindings.get("out"),):
                valid.discard(name)
            continue
        if call.arguments:
            names = dict(zip(("x", "w", "scales", "out"), call.arguments))
        else:
            names = dict(call.bindings)
        p = call.parameters
        count, depth, width = p["r"], p["k"], p["o"]
        parts = min(partitions, depth // 128)
        if depth % (128 * parts):
            raise ValueError("split projection requires whole block-128 partitions")
        if "activation_scales" not in names:
            source = names["x"]
            if source not in quantized:
                bits = graph.buffer(
                    f"pipeline.{source}.bits", graph.buffers[source].shape, "uint8"
                )
                scales = graph.buffer(
                    f"pipeline.{source}.scales", (count, depth // 128), "float32"
                )
                quantized[source] = bits, scales
            bits, scales = quantized[source]
            if source not in valid:
                recorder(
                    "quantize",
                    dict(r=count, k=depth),
                    x=graph.buffers[source],
                    out=bits,
                    scales=scales,
                )
                calls.extend(recorder.calls)
                recorder.calls = []
                valid.add(source)
            names.update(x=bits.name, activation_scales=scales.name)
        original = graph.buffers[names["out"]]
        partial = (
            original
            if parts == 1
            else graph.buffer(
                f"pipeline.partial.{index}", (count, parts, width), "float32"
            )
        )
        parameters = {
            **p,
            "partitions": parts,
            "columns": 64,
            "block_m": 16,
            "threads": 128,
            "stages": 1,
            "packed_copy": False,
            **settings,
        }
        parameters["partitions"] = parts
        recorder(
            "projection_partial",
            parameters,
            **{
                name: graph.buffers[value]
                for name, value in {**names, "out": partial.name}.items()
            },
            label=call.label if parts == 1 else None,
        )
        if parts > 1:
            recorder(
                "projection_merge",
                dict(r=count, o=width, partitions=parts),
                partial=partial,
                out=original,
                label=call.label,
            )
        calls.extend(recorder.calls)
        recorder.calls = []
        valid.discard(original.name)
    graph.calls = calls
    return graph, commit
