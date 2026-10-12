"""Conservative operation fusion; no compiler, provider or model imports."""


def last_rows(plan):
    """Compute only each slot's final row after its last stateful operation.

    Persistent state still consumes the full chunk. Only the terminal pointwise
    graph is cropped, with gathers for inputs crossing that boundary. Unknown
    operations keep the original graph rather than assuming row independence.
    """
    from dataclasses import replace
    from math import prod
    from akas.planning.forward import Call

    last = next(
        (i for i, call in enumerate(plan.calls) if call.kind == "chunk_last"), None
    )
    if last is None:
        return plan
    select = plan.calls[last]
    batch, chunk = select.parameters["r"], select.parameters["chunk"]
    rows = batch * chunk
    boundary = (
        max(
            (
                i
                for i, call in enumerate(plan.calls[:last])
                if any(
                    plan.buffers[name].role == "state"
                    for name in (*call.arguments, *call.bindings.values())
                )
            ),
            default=-1,
        )
        + 1
    )
    tail = plan.calls[boundary:last]
    if not tail or any(
        call.kind not in ("linear", "add", "rms", "swiglu")
        or call.parameters.get("r") != rows
        or call.parameters.get("matrix_output", False)
        for call in tail
    ):
        return plan
    produced, gathers, replacements = set(), [], {}
    for call in tail:
        produced.add(call.arguments[-1])
        for name in call.arguments[:-1]:
            spec = plan.buffers[name]
            if spec.role == "weight" or name in produced or name in replacements:
                continue
            if (
                spec.role != "scratch"
                or spec.dtype != "float32"
                or len(spec.shape) != 1
                or prod(spec.shape) % rows
            ):
                return plan
            width = prod(spec.shape) // rows
            target = name + ".last"
            replacements[name] = target
            gathers.append((name, target, width))
    prefix = []
    for source, target, width in gathers:
        plan.buffer(target, (batch * width,), "float32")
        prefix.append(
            Call(
                "chunk_last",
                dict(r=batch, c=width, chunk=chunk),
                (source, target, select.arguments[-1]),
            )
        )
    for name in produced:
        spec = plan.buffers[name]
        plan.buffers[name] = replace(spec, shape=(batch * prod(spec.shape) // rows,))
    cropped = [
        replace(
            call,
            parameters={**call.parameters, "r": batch},
            arguments=tuple(replacements.get(name, name) for name in call.arguments),
        )
        for call in tail
    ]
    after = [
        replace(
            call,
            arguments=tuple(
                select.arguments[0] if name == select.arguments[1] else name
                for name in call.arguments
            ),
        )
        for call in plan.calls[last + 1 :]
    ]
    plan.calls = [*plan.calls[:boundary], *prefix, *cropped, *after]
    return plan


def swiglu(plan):
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
