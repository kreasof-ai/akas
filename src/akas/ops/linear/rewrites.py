"""Projection fusions that preserve externally visible values and buffer ABIs."""

from collections import Counter
from dataclasses import replace


def fuse_residual_projection(graph):
    """Join packed decode projection and its private residual addition."""
    uses = Counter(name for call in graph.calls for name in (*call.arguments, *call.bindings.values()))
    aliases = {value.alias_of for value in graph.buffers.values() if value.alias_of}
    result, index = [], 0
    while index < len(graph.calls):
        pair = graph.calls[index:index + 2]
        if len(pair) == 2:
            projection, addition = pair
            p = projection.parameters
            if (projection.kind == "linear" and addition.kind == "add"
                and p.get("ggml") and p["r"] == 1
                and len(projection.arguments) == len(addition.arguments) == 3
                and addition.parameters == dict(r=1, c=p["o"])):
                temporary = projection.arguments[-1]
                if (temporary in addition.arguments[:2] and uses[temporary] == 2
                    and temporary not in aliases and graph.buffers[temporary].role == "scratch"):
                    residual = addition.arguments[1] if addition.arguments[0] == temporary else addition.arguments[0]
                    result.append(replace(projection, kind="linear_add",
                                          arguments=(*projection.arguments[:2], residual, addition.arguments[-1]),
                                          label=addition.label))
                    index += 2
                    continue
        result.append(graph.calls[index])
        index += 1
    graph.calls = result
    used = {name for call in result for name in (*call.arguments, *call.bindings.values())}
    graph.buffers = {name: spec for name, spec in graph.buffers.items()
                     if spec.role != "scratch" or name in used or name in aliases}
    return graph


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
    aliases = {value.alias_of for value in plan.buffers.values() if value.alias_of}
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
                and (p["r"] > 1 or (p.get("ggml") and p.get("type") not in (0, 1)))
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
                and all(plan.buffers[name].role == "scratch" and name not in aliases
                        for name in (gate.arguments[2], up.arguments[2]))
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
        if spec.role != "scratch" or name in used or name in aliases
    }
    return plan
