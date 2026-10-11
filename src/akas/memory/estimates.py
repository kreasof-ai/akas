"""Persistent-state memory estimates for Qwen MoE execution."""


def state_bytes(config, *, slots, context, block=128):
    """Logical BF16 KV and FP32 recurrent/conv storage; no workspaces.

    Context rounds up to the allocator's page size. Each request retains
    its own states, including during prefill and until completion.
    """
    if any(type(v) is not int or v <= 0 for v in (slots, context, block)):
        raise ValueError("state sizes must be positive integers")
    if context > config.context_limit:
        raise ValueError("context exceeds the checkpoint limit")
    capacity = (context + block - 1) // block * block
    full, linear = config.layers.count("full_attention"), config.layers.count(
        "linear_attention"
    )
    return dict(
        slots=slots,
        context=context,
        capacity=capacity,
        kv=full * slots * capacity * config.kv_heads * config.head_dim * 2 * 2,
        recurrent=linear
        * slots
        * config.value_heads
        * config.key_dim
        * config.value_dim
        * 4,
        convolution=linear
        * slots
        * (
            2 * config.key_heads * config.key_dim
            + config.value_heads * config.value_dim
        )
        * (config.conv_width - 1)
        * 4,
    )
