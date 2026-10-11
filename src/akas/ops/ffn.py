"""Decoder block operations for explicit packed, FP8 and BF16 execution contracts."""

from akas.execution.profiles.selection import paired_ffn, projection_tile


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


def bf16_batch_ffn(ctx, block):
    ctx.linear(
        block.feed_forward_prefix + "input",
        ctx.buffers["normal"],
        ctx.buffers["mlp_input"],
    )
    ctx.emit(
        "swiglu",
        dict(ctx.common, c=ctx.config.intermediate),
        ctx.buffers["mlp_input"],
        ctx.buffers["activation"],
    )
    ctx.linear(
        block.feed_forward_prefix + "down_proj",
        ctx.buffers["activation"],
        ctx.buffers["ffn"],
    )


def packed_ffn(ctx, block):
    prefix = block.feed_forward_prefix
    if (
        ctx.request.provider == "webgpu" or ctx.request.cuda_optimized
    ) and ctx.rows == 1:
        ctx.emit(
            "add_rms",
            dict(r=ctx.rows, c=ctx.width, eps=ctx.config.epsilon),
            ctx.hidden,
            ctx.buffers["mixed"],
            ctx.other,
            ctx.request.weights[block.post_norm],
            ctx.buffers["normal"],
        )
        ctx.hidden, ctx.other = (ctx.other, ctx.hidden)
    else:
        ctx.emit(
            "add",
            dict(r=ctx.rows, c=ctx.width),
            ctx.hidden,
            ctx.buffers["mixed"],
            ctx.other,
        )
        ctx.hidden, ctx.other = (ctx.other, ctx.hidden)
        if ctx.last_only and block.index == len(ctx.config.layers) - 1:
            ctx.emit(
                "last",
                dict(r=ctx.rows, c=ctx.width),
                ctx.hidden,
                ctx.buffers["last"],
                ctx.control,
            )
            ctx.norm(
                block.post_norm, ctx.buffers["last"], ctx.buffers["last_normal"], rows=1
            )
            gate, up = (
                ctx.request.gguf.tensors[prefix + "" + name + ".weight"]
                for name in ("gate", "up")
            )
            ctx.emit(
                "ffn",
                dict(r=1, k=ctx.width, o=ctx.config.ff, type=gate.type),
                ctx.buffers["last_normal"],
                ctx.request.weights[gate.name],
                ctx.request.weights[up.name],
                ctx.buffers["last_activated"],
            )
            info = ctx.request.gguf.tensors[prefix + "down.weight"]
            o, k = info.shape
            ctx.emit(
                "linear_add",
                dict(r=1, k=k, o=o, type=info.type),
                ctx.buffers["last_activated"],
                ctx.request.weights[info.name],
                ctx.buffers["last"],
                ctx.buffers["last_output"],
            )
            ctx.final_hidden = ctx.buffers["last_output"]
            return True
        ctx.norm(block.post_norm, ctx.hidden, ctx.buffers["normal"])
    gate, up = (
        ctx.request.gguf.tensors[prefix + "" + name + ".weight"]
        for name in ("gate", "up")
    )
    if (ctx.request.provider == "webgpu" or ctx.request.cuda_optimized) and paired_ffn(
        ctx.rows,
        gate,
        up,
        "cuda_optimized" if ctx.request.cuda_optimized else ctx.request.webgpu_profile,
    ):
        p = dict(
            r=ctx.rows,
            k=ctx.width,
            o=ctx.config.ff,
            type=gate.type,
            **projection_tile(ctx.rows, ctx.config.ff),
        )
        ctx.emit(
            "ffn",
            p,
            *ctx.inputs("ffn", p, ctx.buffers["normal"]),
            ctx.request.weights[gate.name],
            ctx.request.weights[up.name],
            ctx.buffers["activated"],
        )
    else:
        ctx.linear(prefix + "gate.weight", ctx.buffers["normal"], ctx.buffers["gate"])
        ctx.linear(prefix + "up.weight", ctx.buffers["normal"], ctx.buffers["up"])
        ctx.emit(
            "swiglu",
            dict(r=ctx.rows, c=ctx.config.ff),
            ctx.buffers["gate"],
            ctx.buffers["up"],
            ctx.buffers["activated"],
        )
    if (
        ctx.request.provider == "webgpu" or ctx.request.cuda_optimized
    ) and ctx.rows == 1:
        info = ctx.request.gguf.tensors[prefix + "down.weight"]
        o, k = info.shape
        ctx.emit(
            "linear_add",
            dict(r=ctx.rows, k=k, o=o, type=info.type),
            ctx.buffers["activated"],
            ctx.request.weights[info.name],
            ctx.hidden,
            ctx.other,
        )
    else:
        ctx.linear(
            prefix + "down.weight", ctx.buffers["activated"], ctx.buffers["mixed"]
        )
        ctx.emit(
            "add",
            dict(r=ctx.rows, c=ctx.width),
            ctx.hidden,
            ctx.buffers["mixed"],
            ctx.other,
        )
    ctx.hidden, ctx.other = (ctx.other, ctx.hidden)
