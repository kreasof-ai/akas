"""Decoder block operations for explicit packed, FP8 and BF16 execution contracts."""

from akas.quantization.kv import is_quantized


def fp8_decode_attention(ctx, block):
    prefix = block.mixer_prefix
    for name, dest in (("q", "attq"), ("k", "attk"), ("v", "attv")):
        ctx.fp8(prefix + name + "_proj", ctx.buffers["normal"], ctx.buffers[dest])
    kc, vc, *scales = ctx.executor.states[block.index]
    p = dict(
        r=ctx.rows,
        eps=ctx.executor.config.epsilon,
        capacity=ctx.executor.context,
        theta=ctx.executor.config.rope_theta,
    )
    ctx.emit(
        "attention_q",
        p,
        ctx.buffers["attq"],
        ctx.weights[prefix + "q_norm.weight"],
        ctx.buffers["positions"],
        ctx.buffers["active"],
        ctx.buffers["attq_normal"],
    )
    kv = (
        dict(kv_dtype=ctx.executor.kv_dtype)
        if is_quantized(ctx.executor.kv_dtype)
        else {}
    )
    ctx.emit(
        "attention_kv",
        dict(p, **kv),
        ctx.buffers["attk"],
        ctx.buffers["attv"],
        ctx.weights[prefix + "k_norm.weight"],
        ctx.buffers["positions"],
        ctx.buffers["active"],
        kc,
        vc,
        *scales,
    )
    ctx.emit(
        "attention_partial",
        dict(
            r=ctx.rows, capacity=ctx.executor.context, splits=ctx.executor.splits, **kv
        ),
        ctx.buffers["attq_normal"],
        kc,
        vc,
        *scales,
        ctx.buffers["positions"],
        ctx.buffers["active"],
        ctx.buffers["att_parts"],
        ctx.buffers["att_stats"],
    )
    ctx.emit(
        "attention_merge",
        dict(r=ctx.rows, splits=ctx.executor.splits),
        ctx.buffers["att_parts"],
        ctx.buffers["att_stats"],
        ctx.buffers["attq"],
        ctx.buffers["active"],
        ctx.buffers["att_out"],
    )
    ctx.fp8(prefix + "o_proj", ctx.buffers["att_out"], ctx.buffers["mixed"])


def fp8_prefill_attention(ctx, block):
    prefix = block.mixer_prefix
    for name, dest in (("q", "attq"), ("k", "attk"), ("v", "attv")):
        ctx.fp8(prefix + name + "_proj", "normal", dest)
    kc, vc, *scales = ctx.model.states[block.index]
    kv = dict(kv_dtype=ctx.model.kv_dtype) if is_quantized(ctx.model.kv_dtype) else {}
    cache = dict(ks=scales[0], vs=scales[1]) if scales else {}
    p = dict(
        r=ctx.rows,
        eps=ctx.model.config.epsilon,
        capacity=ctx.model.context,
        theta=ctx.model.config.rope_theta,
    )
    ctx.emit(
        "attention_q",
        p,
        projection=ctx.buffers["attq"],
        w=ctx.weights[prefix + "q_norm.weight"],
        positions=ctx.buffers["flat_positions"],
        active=ctx.buffers["flat_active"],
        q=ctx.buffers["attq_normal"],
    )
    p = dict(
        slots=ctx.slots,
        chunk=ctx.executor.chunk,
        capacity=ctx.model.context,
        eps=ctx.model.config.epsilon,
        theta=ctx.model.config.rope_theta,
        **kv,
    )
    ctx.emit(
        "attention_kv",
        p,
        projection=ctx.buffers["attk"],
        values=ctx.buffers["attv"],
        w=ctx.weights[prefix + "k_norm.weight"],
        positions=ctx.buffers["flat_positions"],
        active=ctx.buffers["flat_active"],
        kc=kc,
        vc=vc,
        **cache,
    )
    ctx.emit(
        "attention",
        dict(
            slots=ctx.slots, chunk=ctx.executor.chunk, capacity=ctx.model.context, **kv
        ),
        q=ctx.buffers["attq_normal"],
        kc=kc,
        vc=vc,
        **cache,
        projection=ctx.buffers["attq"],
        positions=ctx.model.buffers["positions"],
        lengths=ctx.buffers["lengths"],
        out=ctx.buffers["att_out"],
    )
    ctx.fp8(prefix + "o_proj", "att_out", "mixed")






def floating_attention(plan, p, q, kc, vc, ks, vs, out, positions, active):
    """Bind partitioned decode or tiled prefill through the floating cache ABI."""
    if p.get("optimized") and p.get("chunk", 1) == 1:
        p = {**p, "splits": 16}
        parts = plan.buffer(
            out.name + ".parts", (p["r"] * p["h"] * p["splits"] * 66,), "float32"
        )
        plan("attention_partial", p, q, kc, vc, ks, vs, parts, positions, active)
        plan("attention_merge", p, parts, out, active)
    else:
        plan("attention", p, q, kc, vc, ks, vs, out, positions, active)
