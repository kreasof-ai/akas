"""Persistent-state estimates derived from the allocation layouts."""

import math

from math import prod
from tensor.runtime.dtypes import dtype
from akas.memory.layouts import hybrid_state_specs


def state_bytes(config, *, slots, context, block=128, kv_dtype="bfloat16"):
    """Rounded KV (including FP8 scales), recurrent and convolution storage."""
    if any(type(v) is not int or v <= 0 for v in (slots, context, block)):
        raise ValueError("state sizes must be positive integers")
    limit = getattr(config, "context_limit", getattr(config, "max_context", None))
    if limit is not None and context > limit:
        raise ValueError("context exceeds the checkpoint limit")
    capacity = (context + block - 1) // block * block
    specifications = hybrid_state_specs(config, slots, capacity, kv_dtype=kv_dtype)
    result = dict(
        slots=slots,
        context=context,
        capacity=capacity,
        kv=0,
        recurrent=0,
        convolution=0,
    )
    for layer, specs in specifications.items():
        sizes = [prod(spec.shape) * dtype(spec.dtype).itemsize for spec in specs]
        kind = config.layers[layer]
        if kind == "linear_attention":
            result["recurrent"] += sizes[0]
            result["convolution"] += sizes[1]
        elif kind == "conv":
            result["convolution"] += sum(sizes)
        else:
            result["kv"] += sum(sizes)
    return result


def estimate(graph):
    """Conservative raw buffer bounds, before scratch reuse and shared phases."""
    from tensor.runtime.dtypes import dtype

    sizes = [
        math.prod(spec.shape) * dtype(spec.dtype).itemsize
        for spec in graph.buffers.values()
        if spec.alias_of is None
    ]
    return dict(graph_bytes=sum(sizes), largest_buffer_bytes=max(sizes, default=0))


def memory_bound(graphs):
    from akas.compilation.buffers import scratch_slots
    from tensor.runtime.dtypes import dtype

    persistent = {}
    private = 0
    for graph in graphs:
        size = lambda spec: math.prod(spec.shape) * dtype(spec.dtype).itemsize
        for name, spec in graph.buffers.items():
            if spec.alias_of is None and spec.role in ("weight", "state"):
                persistent[name] = max(persistent.get(name, 0), size(spec))
        groups = scratch_slots(graph)
        private += sum(size(graph.buffers[names[0]]) for names in groups)
        private += sum(
            size(spec)
            for spec in graph.buffers.values()
            if spec.alias_of is None and spec.role not in ("weight", "state", "scratch")
        )
    return dict(
        persistent_bytes=sum(persistent.values()),
        cached_frame_upper_bytes=private,
        total_upper_bytes=sum(persistent.values()) + private,
    )


def program_memory(graphs):
    """Return the per-geometry and resident memory limits for a program bank."""
    graphs = tuple(graphs)
    return [estimate(graph) for graph in graphs], memory_bound(graphs)
