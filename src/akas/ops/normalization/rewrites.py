"""Fuse decode residual updates without dropping their observable outputs."""

from dataclasses import replace


def fuse_residual_normalization(graph):
    def storage(name):
        while graph.buffers[name].alias_of is not None:
            name = graph.buffers[name].alias_of
        return name

    result, index = [], 0
    while index < len(graph.calls):
        pair = graph.calls[index:index + 2]
        if len(pair) == 2:
            addition, normalization = pair
            if (graph.contract == "reference" and addition.kind == "add"
                and normalization.kind == "rms" and addition.parameters.get("r") == 1
                and len(addition.arguments) == len(normalization.arguments) == 3
                and addition.arguments[-1] == normalization.arguments[0]
                and storage(addition.arguments[-1]) not in {storage(name) for name in addition.arguments[:2]}
                and addition.parameters["c"] == normalization.parameters["c"]
                and all(len(graph.buffers[name].shape) == 1 and graph.buffers[name].dtype == "float32"
                        for name in (*addition.arguments, *normalization.arguments))):
                result.append(replace(normalization, kind="add_rms",
                                      arguments=(*addition.arguments, *normalization.arguments[1:])))
                index += 2
                continue
        result.append(graph.calls[index])
        index += 1
    graph.calls = result
    return graph
