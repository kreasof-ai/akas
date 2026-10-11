"""Persistent convolution, recurrent, and KV state for every model adapter."""

from akas.memory.state import BufferSpec
from akas.memory.state import allocate_states


def hybrid_state_specs(
    config,
    capacity,
    context,
    *,
    kv_dtype="bfloat16",
    zero_recurrent=False,
    zero_cache=False,
    flat=False,
):
    """Describe persistent state; the request owns prefix and slot resets."""
    if kv_dtype not in ("float16", "bfloat16", "fp8"):
        raise ValueError("unsupported hybrid KV dtype")
    states = {}
    for layer, kind in enumerate(config.layers):
        values = states[layer] = []
        if kind == "linear_attention":
            width = (
                2 * config.key_heads * config.key_dim
                + config.value_heads * config.value_dim
            )
            shapes = (
                (capacity, config.value_heads, config.value_dim, config.key_dim),
                (capacity, width, getattr(config, "conv_width", 4) - 1),
            )
            for shape in shapes:
                values.append(BufferSpec(shape, "float32", zero=zero_recurrent))
        elif kind == "conv":
            shape = (capacity, (config.conv - 1) * config.width)
            if flat:
                shape = (capacity * (config.conv - 1) * config.width,)
            values.append(BufferSpec(shape, "float32", zero=zero_recurrent))
        elif kind in ("full_attention", "attention"):
            shape = (capacity, config.kv_heads, context, config.head_dim)
            cache_shape = (
                (capacity * config.kv_heads * context * config.head_dim,)
                if flat
                else shape
            )
            for _ in range(2):
                if zero_cache:
                    if kv_dtype != "float16":
                        raise ValueError("zeroed flat cache profile requires FP16")
                    values.append(BufferSpec(cache_shape, "float16", zero=True))
                    continue
                values.append(
                    BufferSpec(cache_shape, "uint8" if kv_dtype == "fp8" else kv_dtype)
                )
            if kv_dtype == "fp8":
                scales = shape[:-1] + ((config.head_dim + 127) // 128,)
                for _ in range(2):
                    values.append(BufferSpec(scales, "float32"))
        else:
            raise ValueError(f"unsupported hybrid layer: {kind}")
    return states


def allocate_hybrid_states(device, config, capacity, context, **options):
    return allocate_states(
        device, hybrid_state_specs(config, capacity, context, **options)
    )
