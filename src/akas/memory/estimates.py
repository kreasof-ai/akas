"""Persistent-state estimates derived from the allocation layouts."""

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
