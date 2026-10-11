"""Decoder block operations for explicit packed, FP8 and BF16 execution contracts."""


def packed_shortconv(ctx, block):
    prefix = block.mixer_prefix
    ctx.linear(prefix + "in_proj.weight", ctx.buffers["normal"], ctx.buffers["project"])
    ctx.emit(
        "conv",
        dict(r=ctx.rows, c=ctx.width),
        ctx.buffers["project"],
        ctx.request.weights[prefix + "conv.weight"],
        ctx.request.states[block.index],
        ctx.buffers["convout"],
        ctx.control,
    )
    ctx.linear(prefix + "out_proj.weight", ctx.buffers["convout"], ctx.buffers["mixed"])
