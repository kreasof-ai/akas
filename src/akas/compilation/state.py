"""Adapt row-local state addressing to the common request mapping contract."""

from dataclasses import replace
from importlib import import_module
from akas.artifacts.schema import Source


def mapped_source(source, mapping):
    states,rows,capacity = mapping
    return Source("akas.ops.state.mapping","make_kernel",
                  (source.module,source.factory,source.args,states,rows,capacity),
                  dependencies=(source.module, *source.dependencies))


def map_requests(graph, kernels):
    """Keep numerical operations unchanged while binding persistent slot pools.

    Kernels with an explicit mapping already implement this contract. Older
    row-local kernels are lifted during compilation, never by runtime dispatch.
    """
    if "mapping" not in graph.buffers:
        return graph,kernels
    from akas.graph import Graph
    graph = Graph.from_metadata(graph.metadata())
    rows = graph.buffers["mapping"].shape[0]
    calls = {call.key:call for call in graph.calls}
    mappings = {}
    result = []
    for declaration in kernels:
        source = declaration.source
        kernel = getattr(import_module(source.module),source.factory)(*source.args)
        signature = [kernel.buffer_map[p] for p in kernel.params]
        if any(buffer.name == "mapping" for buffer in signature):
            result.append(declaration)
            continue
        call = calls[declaration.key]
        names = list(call.arguments) if call.arguments else [call.bindings[buffer.name] for buffer in signature]
        states = tuple(index for index,name in enumerate(names) if graph.buffers[name].role=="state")
        if not states:
            result.append(declaration)
            continue
        capacities = set()
        for index in states:
            old = tuple(int(n) for n in signature[index].shape)
            new = graph.buffers[names[index]].shape
            if old[0]%rows or old[1:] != new[1:] or new[0]%(old[0]//rows):
                raise ValueError("state layout cannot use request-row mapping")
            capacities.add(new[0]//(old[0]//rows))
        if len(capacities) != 1:
            raise ValueError("operation state pools disagree")
        mapping = (states,rows,capacities.pop())
        mappings[declaration.key] = mapping
        result.append(replace(declaration,source=mapped_source(source,mapping),mapping=mapping))
    graph.calls = [replace(call,arguments=(*call.arguments,"mapping")) if call.arguments and call.key in mappings
                   else replace(call,bindings={**call.bindings,"mapping":"mapping"}) if call.key in mappings
                   else call for call in graph.calls]
    return graph,result
