"""Explicit activation-storage lowering before operation schedule search."""


def q16_activations(graph, *, subgroup=True, fixed_residual=False, schedules=None, mixed=False,
                    half_only=False):
    """Reuse two-plane activations for Q4_0 projections sharing one input.

    This changes the declared arithmetic policy: half-rounded inputs become two
    signed Q8 planes. It must be enabled before compiling/tuning; kernel search
    cannot silently introduce quantized activation storage into a floating ABI.
    All geometries use their own rows/depth, with no model names or shape table.
    """
    from dataclasses import replace

    from akas.graph import Call
    from akas.quantization import get

    result, inputs = [], {}
    for call in graph.calls:
        p = call.parameters
        if (
            call.kind not in ("linear", "ffn")
            or not p.get("ggml")
            or p.get("type") != 2
            or p["r"] == 1
        ):
            result.append(call)
            continue
        if half_only or mixed and (call.kind == "ffn" or p["o"] == 3 * p["k"]):
            result.append(replace(call, parameters={**p, "half_accum": True}))
            continue
        source = call.arguments[0]
        key = source, p["r"], p["k"]
        if key not in inputs:
            prefix = source + ".q16"
            rows, depth = p["r"], p["k"]
            storage = get("activations", "q16").storage(
                (rows, depth), provider="webgpu"
            )
            buffers = [
                graph.buffer(prefix + "." + name, spec.shape, spec.dtype)
                for name, spec in zip(
                    ("packed", "scales", "sums"), (storage.values, *storage.metadata)
                )
            ]
            names = tuple(value.name for value in buffers)
            inputs[key] = names
            result.append(
                Call(
                    "quantize_q16",
                    dict(r=rows, k=depth, sg=subgroup, fixed_residual=fixed_residual),
                    (source, *names),
                )
            )
        integer = dict((schedules or {}).get((call.kind, p["r"], p["k"], p["o"]), {}))
        integer["fixed_residual"] = fixed_residual
        result.append(
            replace(
                call,
                parameters={**p, "q16": True, "integer": integer},
                arguments=(*inputs[key], *call.arguments[1:]),
            )
        )
    graph.calls = result
    return graph


class FP8ActivationCache:
    """FP8 graph scratch reuse; invalidate borrowed storage on source writes."""

    def __init__(self, sources, quantized, *, rows):
        self.sources = sources
        self.quantized = quantized
        self.rows = rows
        self.valid = set()

    def get(self, name, emit):
        if name not in self.valid:
            source = self.sources[name]
            parameters = dict(r=self.rows, k=source.shape[-1])
            if len(source.shape) == 3:
                parameters["top"] = source.shape[-2]
            bits, scales = self.quantized[name]
            emit("quantize", parameters, x=source, out=bits, scales=scales)
            self.valid.add(name)
        return self.quantized[name]

    def invalidate(self, buffer):
        self.valid.difference_update(
            name for name in tuple(self.valid) if self.sources[name] is buffer
        )
