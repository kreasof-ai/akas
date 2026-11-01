"""Fold private storage conversions into operation-owned typed outputs."""


def fuse_output_casts(graph, producers, *, cast_kind, dtype):
    """Move one conversion's unchanged rounding point into its sole producer.

    Owners declare which positional output supports ``output_dtype``. Public
    source values, other consumers and in-place writes retain their conversion.
    Named calls use their explicit ``out`` binding.
    """
    from dataclasses import replace

    calls = list(graph.calls)
    families, exposed = graph.storage_families(), graph.exposed_buffers()
    removed = set()
    for index, cast in enumerate(calls):
        if cast.kind != cast_kind:
            continue
        names = (
            dict(cast.bindings)
            if cast.bindings
            else dict(zip(("x", "out"), cast.arguments))
        )
        if set(names) != {"x", "out"}:
            continue
        source, target = (graph.buffers[names[key]] for key in ("x", "out"))
        family = families[graph.storage_owner(source.name)]
        if (
            source.dtype != "float32"
            or target.dtype != dtype
            or source.shape != target.shape
            or family & exposed
        ):
            continue
        users = [
            i
            for i, call in enumerate(calls)
            if family.intersection((*call.arguments, *call.bindings.values()))
        ]
        if len(users) != 2 or users[-1] != index:
            continue
        producer_index = users[0]
        producer = calls[producer_index]
        if producer.kind not in producers:
            continue
        output = (
            producer.bindings.get("out")
            if producer.bindings
            else producer.arguments[producers[producer.kind]]
        )
        references = (*producer.arguments, *producer.bindings.values())
        if output not in family or sum(name in family for name in references) != 1:
            continue
        calls[producer_index] = replace(
            producer,
            parameters={**producer.parameters, "output_dtype": dtype},
            arguments=tuple(
                target.name if name == output else name for name in producer.arguments
            ),
            bindings={
                key: target.name if name == output else name
                for key, name in producer.bindings.items()
            },
        )
        removed.add(index)
    graph.calls = [call for index, call in enumerate(calls) if index not in removed]
    return graph
