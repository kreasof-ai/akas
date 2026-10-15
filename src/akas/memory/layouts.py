"""Persistent convolution, recurrent, and KV state for every model adapter."""

from akas.memory.state import BufferSpec
from akas.memory.state import allocate_states
from akas.quantization.kv import pair_layout, is_quantized


def hybrid_state_specs(
    config,
    capacity,
    context,
    *,
    kv_dtype="bfloat16",
    zero_recurrent=False,
    zero_cache=False,
    flat=False,
    provider=None,
):
    """Describe persistent state; the request owns prefix and slot resets."""
    quantized = is_quantized(kv_dtype)
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
            if quantized:
                if zero_cache:
                    raise ValueError("zeroed flat cache profile requires FP16")
                specs = pair_layout(kv_dtype, shape, provider=provider)
                if flat:
                    from math import prod

                    specs = tuple(
                        BufferSpec((prod(spec.shape),), spec.dtype) if i < 2 else spec
                        for i, spec in enumerate(specs)
                    )
                values.extend(specs)
            else:
                for _ in range(2):
                    if zero_cache and kv_dtype != "float16":
                        raise ValueError("zeroed flat cache profile requires FP16")
                    values.append(BufferSpec(cache_shape, kv_dtype, zero=zero_cache))
        else:
            raise ValueError(f"unsupported hybrid layer: {kind}")
    return states


def allocate_hybrid_states(device, config, capacity, context, **options):
    options.setdefault("provider", getattr(device, "info", {}).get("provider"))
    return allocate_states(
        device, hybrid_state_specs(config, capacity, context, **options)
    )
