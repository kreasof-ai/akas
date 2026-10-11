"""Decoder block operations for explicit packed, FP8 and BF16 execution contracts."""


def fp8_decode_moe(ctx, block):
    prefix = block.feed_forward_prefix
    ctx.bf16(prefix + "gate", ctx.buffers["normal"], ctx.buffers["router"])
    ctx.bf16(
        prefix + "shared_expert_gate", ctx.buffers["normal"], ctx.buffers["shared_gate"]
    )
    ctx.emit(
        "router",
        dict(r=ctx.rows, experts=256, top=8),
        ctx.buffers["router"],
        ctx.buffers["ids"],
        ctx.buffers["router_weights"],
    )
    ctx.emit(
        "moe_groups",
        dict(r=ctx.rows, top=8),
        ctx.buffers["ids"],
        ctx.buffers["experts"],
        ctx.buffers["routes"],
    )
    for name, dest in (("gate_proj", "expert_g"), ("up_proj", "expert_u")):
        ctx.emit(
            "fp8_experts",
            dict(r=ctx.rows, experts=256, top=8, o=512, k=2048, routed_input=False),
            ctx.buffers["normal"],
            ctx.weights[prefix + "experts." + name + ".weight"],
            ctx.weights[prefix + "experts." + name + ".weight_scale_inv"],
            ctx.buffers["experts"],
            ctx.buffers["routes"],
            ctx.buffers[dest],
        )
    ctx.emit(
        "swiglu_experts",
        dict(r=ctx.rows, top=8, c=512),
        ctx.buffers["expert_g"],
        ctx.buffers["expert_u"],
        ctx.buffers["expert_activation"],
    )
    ctx.emit(
        "fp8_experts",
        dict(r=ctx.rows, experts=256, top=8, o=2048, k=512, routed_input=True),
        ctx.buffers["expert_activation"],
        ctx.weights[prefix + "experts.down_proj.weight"],
        ctx.weights[prefix + "experts.down_proj.weight_scale_inv"],
        ctx.buffers["experts"],
        ctx.buffers["routes"],
        ctx.buffers["expert_out"],
    )
    for name, dest in (("gate_proj", "shared_g"), ("up_proj", "shared_u")):
        ctx.fp8(
            prefix + "shared_expert." + name, ctx.buffers["normal"], ctx.buffers[dest]
        )
    ctx.emit(
        "swiglu",
        dict(r=ctx.rows, c=512),
        ctx.buffers["shared_g"],
        ctx.buffers["shared_u"],
        ctx.buffers["shared_activation"],
    )
    ctx.fp8(
        prefix + "shared_expert.down_proj",
        ctx.buffers["shared_activation"],
        ctx.buffers["shared_out"],
    )
    ctx.emit(
        "moe_combine",
        dict(r=ctx.rows, top=8, c=2048),
        ctx.buffers["expert_out"],
        ctx.buffers["router_weights"],
        ctx.buffers["shared_out"],
        ctx.buffers["shared_gate"],
        ctx.buffers["ffn"],
        label=(block.index, "mlp"),
    )


def fp8_prefill_moe(ctx, block):
    prefix = block.feed_forward_prefix
    ctx.bf16(prefix + "gate", "normal", "router")
    ctx.bf16(prefix + "shared_expert_gate", "normal", "shared_gate")
    ctx.emit(
        "router",
        dict(r=ctx.rows, experts=256, top=8),
        logits=ctx.buffers["router"],
        ids=ctx.buffers["ids"],
        weights=ctx.buffers["router_weights"],
    )
    ctx.small(
        "expert_routes",
        ids=ctx.buffers["ids"],
        active=ctx.buffers["flat_active"],
        counts=ctx.buffers["counts"],
        routes=ctx.buffers["routes"],
    )
    for name, out in (("gate_proj", "expert_g"), ("up_proj", "expert_u")):
        ctx.experts(prefix + "experts." + name, "normal", out, 512, 2048, False)
    ctx.emit(
        "swiglu_experts",
        dict(r=ctx.rows, top=8, c=512),
        g=ctx.buffers["expert_g"],
        u=ctx.buffers["expert_u"],
        out=ctx.buffers["expert_activation"],
    )
    ctx.experts(
        prefix + "experts.down_proj", "expert_activation", "expert_out", 2048, 512, True
    )
    for name, out in (("gate_proj", "shared_g"), ("up_proj", "shared_u")):
        ctx.fp8(prefix + "shared_expert." + name, "normal", out)
    ctx.emit(
        "swiglu",
        dict(r=ctx.rows, c=512),
        g=ctx.buffers["shared_g"],
        u=ctx.buffers["shared_u"],
        out=ctx.buffers["shared_activation"],
    )
    ctx.fp8(prefix + "shared_expert.down_proj", "shared_activation", "shared_out")
    ctx.emit(
        "moe_combine",
        dict(r=ctx.rows, top=8, c=2048),
        experts=ctx.buffers["expert_out"],
        weights=ctx.buffers["router_weights"],
        shared=ctx.buffers["shared_out"],
        gate=ctx.buffers["shared_gate"],
        out=ctx.buffers["ffn"],
    )
