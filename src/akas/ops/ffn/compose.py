"""Decoder block operations for explicit packed, FP8 and BF16 execution contracts."""

from akas.profiles.independent.selection import paired_ffn
from akas.profiles.independent.selection import projection_tile


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
