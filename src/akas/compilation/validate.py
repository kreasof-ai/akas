"""Check candidate buffer contracts against every call in the model graph."""


def kernel_bindings(graph, key, signature):
    """Reject schedules or algorithms that change the declared operation ABI."""
    for call in graph.calls:
        if call.key != key:
            continue
        if call.arguments:
            names = call.arguments
        else:
            if set(call.bindings) != {spec["name"] for spec in signature}:
                raise ValueError("candidate changes named operation bindings")
            names = [call.bindings[spec["name"]] for spec in signature]
        if len(names) != len(signature):
            raise ValueError("candidate changes operation argument count")
        for name, argument in zip(names, signature):
            spec = graph.buffers[name]
            if argument.get("kind", "buffer") != "buffer":
                raise ValueError("graph buffer cannot bind a scalar argument")
            if tuple(argument["shape"]) != spec.shape or argument["dtype"] != spec.dtype:
                raise ValueError(f"candidate changes storage contract for {name}")
