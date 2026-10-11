"""Decoder block operations for explicit packed, FP8 and BF16 execution contracts."""


def fp8_decode_gdn(ctx, block):
    prefix = block.mixer_prefix
    for name, dest in (("in_proj_qkv", "qkv"), ("in_proj_z", "z")):
        ctx.fp8(prefix + name, ctx.buffers["normal"], ctx.buffers[dest])
    for name in ("a", "b"):
        ctx.bf16(prefix + "in_proj_" + name, ctx.buffers["normal"], ctx.buffers[name])
    state, history = ctx.executor.states[block.index]
    ctx.emit(
        "gdn_conv",
        dict(r=ctx.rows),
        ctx.buffers["qkv"],
        ctx.weights[prefix + "conv1d.weight"],
        ctx.buffers["active"],
        history,
        ctx.buffers["conv"],
    )
    ctx.emit(
        "gdn_prepare",
        dict(r=ctx.rows),
        ctx.buffers["conv"],
        ctx.buffers["a"],
        ctx.buffers["b"],
        ctx.weights[prefix + "dt_bias"],
        ctx.weights[prefix + "A_log"],
        ctx.buffers["gdn_q"],
        ctx.buffers["gdn_k"],
        ctx.buffers["gdn_v"],
        ctx.buffers["gdn_decay"],
        ctx.buffers["gdn_beta"],
    )
    ctx.emit(
        "gdn_recurrent",
        dict(slots=ctx.rows, heads=32, key=128, value=128),
        ctx.buffers["gdn_q"],
        ctx.buffers["gdn_k"],
        ctx.buffers["gdn_v"],
        ctx.buffers["gdn_decay"],
        ctx.buffers["gdn_beta"],
        ctx.buffers["active"],
        state,
        ctx.buffers["gdn_out"],
    )
    ctx.emit(
        "gdn_norm",
        dict(r=ctx.rows, eps=ctx.executor.config.epsilon),
        ctx.buffers["gdn_out"],
        ctx.buffers["z"],
        ctx.weights[prefix + "norm.weight"],
        ctx.buffers["gdn_normal"],
    )
    ctx.fp8(prefix + "out_proj", ctx.buffers["gdn_normal"], ctx.buffers["mixed"])


def fp8_prefill_gdn(ctx, block):
    prefix = block.mixer_prefix
    for name, dest in (("in_proj_qkv", "qkv"), ("in_proj_z", "z")):
        ctx.fp8(prefix + name, "normal", dest)
    for name in ("a", "b"):
        ctx.bf16(prefix + "in_proj_" + name, "normal", name)
    state, history = ctx.model.states[block.index]
    ctx.small(
        "gdn_conv",
        x=ctx.buffers["qkv"],
        w=ctx.weights[prefix + "conv1d.weight"],
        lengths=ctx.buffers["lengths"],
        state=history,
        out=ctx.buffers["conv"],
    )
    ctx.emit(
        "gdn_prepare",
        dict(r=ctx.rows),
        qkv=ctx.buffers["conv"],
        a=ctx.buffers["a"],
        b=ctx.buffers["b"],
        dt=ctx.weights[prefix + "dt_bias"],
        A=ctx.weights[prefix + "A_log"],
        q=ctx.buffers["gdn_q"],
        k=ctx.buffers["gdn_k"],
        v=ctx.buffers["gdn_v"],
        g=ctx.buffers["gdn_decay"],
        beta=ctx.buffers["gdn_beta"],
    )
    ctx.small(
        "gdn_scan",
        q=ctx.buffers["gdn_q"],
        k=ctx.buffers["gdn_k"],
        v=ctx.buffers["gdn_v"],
        g=ctx.buffers["gdn_decay"],
        beta=ctx.buffers["gdn_beta"],
        lengths=ctx.buffers["lengths"],
        state=state,
        out=ctx.buffers["gdn_out"],
    )
    ctx.emit(
        "gdn_norm",
        dict(r=ctx.rows, eps=ctx.model.config.epsilon),
        x=ctx.buffers["gdn_out"],
        z=ctx.buffers["z"],
        w=ctx.weights[prefix + "norm.weight"],
        out=ctx.buffers["gdn_normal"],
    )
    ctx.fp8(prefix + "out_proj", "gdn_normal", "mixed")


def bf16_batch_gdn(ctx, block):
    prefix = block.mixer_prefix
    proj = ctx.config.qkv_width + ctx.config.value_width + 2 * ctx.config.value_heads
    saved = {
        name: (
            ctx.buffers[f"{block.index}_{name}"]
            if ctx.executor.verify
            else ctx.buffers[name]
        )
        for name in ("gdn_input", "q", "k", "v", "g", "beta")
    }
    ctx.linear(prefix + "input", ctx.buffers["normal"], saved["gdn_input"])
    state, history = ctx.executor.owner.states[block.index]
    control = dict(commit=False) if ctx.executor.verify else {}
    conv_args = (
        saved["gdn_input"],
        ctx.weights[prefix + "conv1d.weight"],
        ctx.buffers["mapping"],
        ctx.buffers["lengths"],
        history,
        ctx.buffers["conv"],
    )
    ctx.emit(
        "gdn_conv",
        dict(ctx.common, **control, channels=ctx.config.qkv_width, projection=proj),
        *conv_args,
    )
    ctx.emit(
        "gdn_prepare",
        dict(
            ctx.common,
            heads=ctx.config.value_heads,
            key_heads=ctx.config.key_heads,
            d=ctx.config.key_dim,
            channels=ctx.config.qkv_width,
            projection=proj,
        ),
        ctx.buffers["conv"],
        saved["gdn_input"],
        ctx.weights[prefix + "dt_bias"],
        ctx.weights[prefix + "A_log"],
        saved["q"],
        saved["k"],
        saved["v"],
        saved["g"],
        saved["beta"],
    )
    scan_args = (
        saved["q"],
        saved["k"],
        saved["v"],
        saved["g"],
        saved["beta"],
        ctx.buffers["mapping"],
        ctx.buffers["lengths"],
        state,
        ctx.buffers["gdn_out"],
    )
    ctx.emit(
        "gdn_scan",
        dict(
            ctx.common,
            **control,
            heads=ctx.config.value_heads,
            d=ctx.config.key_dim,
            tile=32,
        ),
        *scan_args,
    )
    if ctx.executor.verify:
        ctx.emit(
            "gdn_conv",
            dict(ctx.common, channels=ctx.config.qkv_width, projection=proj),
            *conv_args,
            to_commit=True,
        )
        ctx.emit(
            "gdn_scan",
            dict(
                ctx.common, heads=ctx.config.value_heads, d=ctx.config.key_dim, tile=32
            ),
            *scan_args,
            to_commit=True,
        )
    ctx.emit(
        "gdn_norm",
        dict(
            ctx.common,
            heads=ctx.config.value_heads,
            d=ctx.config.value_dim,
            channels=ctx.config.qkv_width,
            projection=proj,
            eps=ctx.config.epsilon,
        ),
        ctx.buffers["gdn_out"],
        saved["gdn_input"],
        ctx.weights[prefix + "norm.weight"],
        ctx.buffers["gdn_normal"],
    )
    ctx.linear(prefix + "out_proj", ctx.buffers["gdn_normal"], ctx.buffers["mixed"])
