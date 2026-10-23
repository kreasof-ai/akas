"""Derive output-free token work from operation-owned write declarations."""

from akas.graph import Graph
from akas.kernels.registry import graph_outputs


def state_prefill(graph):
    """Keep persistent writes and their dependencies, excluding terminal heads.

    Unknown operations remain conservatively live. In-place state is both an
    input and an output, so earlier recurrent/cache writes retain their order.
    """
    graph = Graph.from_metadata(graph.metadata())

    def base(name):
        while graph.buffers[name].alias_of is not None:
            name = graph.buffers[name].alias_of
        return name

    request = graph.request or {}
    retained = {
        name: request["features"][name] for name in request.get("prefill_features", ())
    }
    live = {base(name) for name, spec in graph.buffers.items() if spec.role == "state"}
    live.update(base(name) for name in retained.values())
    kept = []
    for call in reversed(graph.calls):
        outputs = graph_outputs(call, contract=graph.contract)
        if outputs is not None and not any(base(name) in live for name in outputs):
            continue
        kept.append(call)
        if outputs is not None:
            live.difference_update(base(name) for name in outputs)
        live.update(
            base(name)
            for name in (*call.arguments, *call.bindings.values())
            if outputs is None
            or name not in outputs
            or graph.buffers[name].role == "state"
        )
    if len(kept) == len(graph.calls):
        return None
    graph.calls = list(reversed(kept))
    request = graph.request
    if request is None:
        return None
    if retained:
        request["features"] = retained
    else:
        request.pop("features", None)
    needed = {
        name
        for call in graph.calls
        for name in (*call.arguments, *call.bindings.values())
    }
    needed.update(request["inputs"].values())
    needed.update(retained.values())
    needed.update(request.get("reset", ()))
    needed.add(request["prediction"])
    needed.update(name for name, spec in graph.buffers.items() if spec.role == "state")
    for name in tuple(needed):
        while graph.buffers[name].alias_of is not None:
            name = graph.buffers[name].alias_of
            needed.add(name)
    graph.buffers = {
        name: spec for name, spec in graph.buffers.items() if name in needed
    }
    return graph
