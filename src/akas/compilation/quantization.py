"""Explicit activation-storage lowering before operation schedule search."""


def q16_activations(graph, *, subgroup=True, fixed_residual=False, schedules=None):
    """Reuse two-plane activations for Q4_0 projections sharing one input.

    This changes the declared arithmetic policy: half-rounded inputs become two
    signed Q8 planes. It must be enabled before compiling/tuning; kernel search
    cannot silently introduce quantized activation storage into a floating ABI.
    All geometries use their own rows/depth, with no model names or shape table.
    """
    from dataclasses import replace
    from akas.graph import Call

    result, inputs = [], {}
    for call in graph.calls:
        p = call.parameters
        if call.kind not in ("linear", "ffn") or not p.get("ggml") or p.get("type") != 2 or p["r"] == 1:
            result.append(call)
            continue
        source = call.arguments[0]
        key = source, p["r"], p["k"]
        if key not in inputs:
            prefix = source + ".q16"
            rows, depth = p["r"], p["k"]
            buffers = [graph.buffer(prefix+"."+name, (count,), dtype)
                       for name,count,dtype in (("packed", rows*depth//2, "uint32"),
                                                ("scales", rows*depth//16, "float32"),
                                                ("sums", rows*depth//16, "int32"))]
            names = tuple(value.name for value in buffers)
            inputs[key] = names
            result.append(Call("quantize_q16", dict(r=rows,k=depth,sg=subgroup,
                                                    fixed_residual=fixed_residual),
                               (source, *names)))
        integer = dict((schedules or {}).get((call.kind,p["r"],p["k"],p["o"]), {}))
        integer["fixed_residual"] = fixed_residual
        result.append(replace(call, parameters={**p,"q16":True,"integer":integer},
                              arguments=(*inputs[key], *call.arguments[1:])))
    graph.calls = result
    return graph
