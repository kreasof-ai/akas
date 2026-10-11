"""Decoder block operations for explicit packed, FP8 and BF16 execution contracts."""

from akas.execution.profiles.packed import QUANT_ATTENTION, SUBGROUP_PROFILES


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
    kv = dict(kv_dtype="fp8") if ctx.executor.kv_dtype == "fp8" else {}
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
    kv = dict(kv_dtype="fp8") if ctx.model.kv_dtype == "fp8" else {}
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


def bf16_batch_attention(ctx, block):
    prefix = block.mixer_prefix
    ctx.linear(prefix + "input", ctx.buffers["normal"], ctx.buffers["att_input"])
    kc, vc = ctx.executor.owner.states[block.index]
    att = dict(
        ctx.common,
        heads=ctx.config.heads,
        kv_heads=ctx.config.kv_heads,
        d=ctx.config.head_dim,
        capacity=ctx.executor.owner.context,
        eps=ctx.config.epsilon,
    )
    ctx.emit(
        "attention_qkv",
        dict(att, theta=ctx.config.theta),
        ctx.buffers["att_input"],
        ctx.weights[prefix + "q_norm.weight"],
        ctx.weights[prefix + "k_norm.weight"],
        ctx.buffers["mapping"],
        ctx.buffers["flat_position"],
        ctx.buffers["active"],
        kc,
        vc,
        ctx.buffers["att_q"],
    )
    ctx.emit(
        "attention",
        att,
        ctx.buffers["att_q"],
        kc,
        vc,
        ctx.buffers["att_input"],
        ctx.buffers["mapping"],
        ctx.buffers["position"],
        ctx.buffers["lengths"],
        ctx.buffers["att_out"],
    )
    ctx.linear(prefix + "o_proj", ctx.buffers["att_out"], ctx.buffers["mixed"])


def packed_attention(ctx, block):
    prefix = block.mixer_prefix
    tail_attention = (
        ctx.request.prefill_tail
        and ctx.scheduled_rows == 128
        and (block.index == len(ctx.config.layers) - 3)
    )
    for name in ("k", "v") if tail_attention else ("q", "k", "v"):
        ctx.linear(
            prefix + "" + name + ".weight", ctx.buffers["normal"], ctx.buffers[name]
        )
    kc, vc = ctx.request.caches[block.index]
    p = dict(
        r=ctx.rows,
        h=ctx.config.heads,
        kh=ctx.config.kv_heads,
        d=ctx.config.head_dim,
        cap=ctx.request.capacity,
        eps=ctx.config.epsilon,
        theta=ctx.config.theta,
    )
    if ctx.request.provider == "webgpu" or (
        ctx.request.cuda_optimized and tail_attention
    ):
        if tail_attention:
            ctx.emit(
                "kvnorm",
                p,
                ctx.buffers["k"],
                ctx.buffers["v"],
                ctx.request.weights[prefix + "k_norm.weight"],
                kc,
                vc,
                ctx.control,
            )
            ctx.emit(
                "prefill_tail",
                dict(r=ctx.rows, c=ctx.width, t=ctx.request.prefill_tail),
                ctx.buffers["normal"],
                ctx.buffers["tail"]["normal"],
                ctx.control,
                ctx.buffers["tail_control"],
            )
            ctx.emit(
                "prefill_tail",
                dict(r=ctx.rows, c=ctx.width, t=ctx.request.prefill_tail),
                ctx.hidden,
                ctx.buffers["tail"]["hidden"],
                ctx.control,
                ctx.buffers["tail_control"],
            )
            ctx.control = ctx.buffers["tail_control"]
            ctx.buffers = ctx.buffers["tail"]
            ctx.rows = ctx.request.prefill_tail
            ctx.hidden, ctx.other = (ctx.buffers["hidden"], ctx.buffers["hidden2"])
            ctx.linear(prefix + "q.weight", ctx.buffers["normal"], ctx.buffers["q"])
            ctx.emit(
                "qnorm",
                dict(p, r=ctx.rows),
                ctx.buffers["q"],
                ctx.request.weights[prefix + "q_norm.weight"],
                ctx.buffers["qo"],
                ctx.control,
            )
        else:
            ctx.emit(
                "qnorm",
                p,
                ctx.buffers["q"],
                ctx.request.weights[prefix + "q_norm.weight"],
                ctx.buffers["qo"],
                ctx.control,
            )
            ctx.emit(
                "kvnorm",
                p,
                ctx.buffers["k"],
                ctx.buffers["v"],
                ctx.request.weights[prefix + "k_norm.weight"],
                kc,
                vc,
                ctx.control,
            )
    else:
        ctx.emit(
            "qkv",
            p,
            ctx.buffers["q"],
            ctx.buffers["k"],
            ctx.buffers["v"],
            ctx.request.weights[prefix + "q_norm.weight"],
            ctx.request.weights[prefix + "k_norm.weight"],
            ctx.buffers["qo"],
            kc,
            vc,
            ctx.request.control,
        )
    attention_input = ctx.buffers["qo"]
    fused = (
        ctx.request.provider == "webgpu"
        and ctx.rows == 1
        and (
            ctx.request.webgpu_profile == "decode_fused"
            and (ctx.config.heads, ctx.config.kv_heads, ctx.config.head_dim)
            == (16, 8, 64)
            or (
                ctx.request.webgpu_profile
                in ("quant_searched", "prefill_q16", "prefill_mixed")
                and (ctx.config.heads, ctx.config.kv_heads, ctx.config.head_dim)
                == (32, 8, 64)
                and QUANT_ATTENTION.get("fused_scores")
            )
        )
    )
    if (
        ctx.request.provider == "webgpu"
        and ctx.request.webgpu_profile in SUBGROUP_PROFILES
        and (ctx.rows == 1)
        and (not fused)
    ):
        ctx.emit(
            "attention_scores",
            dict(
                r=ctx.rows,
                h=ctx.config.heads,
                kh=ctx.config.kv_heads,
                d=ctx.config.head_dim,
                cap=ctx.request.capacity,
            ),
            ctx.buffers["qo"],
            kc,
            ctx.buffers["scores"],
            ctx.request.control,
        )
        attention_input = ctx.buffers["scores"]
    arguments = (
        (attention_input, vc, ctx.buffers["attn"], ctx.control)
        if attention_input is not ctx.buffers["qo"]
        else (attention_input, kc, vc, ctx.buffers["attn"], ctx.control)
    )
    if ctx.request.cuda_optimized and ctx.rows == 1:
        ctx.emit(
            "attention_grouped" if ctx.grouped else "attention_partial",
            dict(
                r=1,
                h=ctx.config.heads,
                kh=ctx.config.kv_heads,
                d=ctx.config.head_dim,
                cap=ctx.request.capacity,
                splits=16,
            ),
            ctx.buffers["qo"],
            kc,
            vc,
            ctx.buffers["parts"],
            ctx.control,
        )
        ctx.emit(
            "attention_merge",
            dict(h=ctx.config.heads, d=ctx.config.head_dim, splits=16),
            ctx.buffers["parts"],
            ctx.buffers["attn"],
        )
    else:
        ctx.emit(
            "attention",
            dict(
                r=ctx.rows,
                h=ctx.config.heads,
                kh=ctx.config.kv_heads,
                d=ctx.config.head_dim,
                cap=ctx.request.capacity,
            ),
            *arguments,
        )
    ctx.linear(prefix + "output.weight", ctx.buffers["attn"], ctx.buffers["mixed"])
