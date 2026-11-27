"""Fixed-pass Inkling 2 Mini text inference on the common CUDA executor."""

from akas.models.definition import Composition, Definition
from akas.models.inkling2.weights import Inkling2MiniCheckpoint


def compose(
    checkpoint,
    *,
    rows,
    chunk,
    max_requests,
    max_context,
    provider="cuda",
    phase=None,
    kv_dtype="bfloat16",
    reuse_residual_banks=True,
):
    from math import prod
    from akas.graph import Graph

    config = checkpoint.config
    if provider != "cuda" or kv_dtype != "bfloat16":
        raise ValueError("Inkling 2 Mini text requires CUDA and BF16 KV")
    if phase not in (None, "decode", "prefill"):
        raise ValueError("Inkling 2 Mini supports autoregressive text phases")
    if (
        not 1 <= rows <= max_requests
        or not 1 <= chunk <= max_context <= config.max_context
    ):
        raise ValueError("invalid Inkling 2 Mini request geometry")

    graph = Graph(contract="reference", phase=phase or "forward")
    count, width = rows * chunk, config.width
    positions = graph.buffer("positions", (rows,), "int32", role="state")
    lengths = graph.buffer("active", (rows,), "int32", role="input")
    tokens = graph.buffer("tokens", (count,), "int32", role="input")
    row_positions, row_active = positions, lengths
    if chunk > 1:
        row_positions = graph.buffer("row_positions", (count,), "int32")
        row_active = graph.buffer("row_active", (count,), "int32")
        graph(
            "chunk_controls",
            dict(r=rows, chunk=chunk),
            positions,
            lengths,
            row_positions,
            row_active,
        )

    def weight(name):
        return checkpoint.graph_weight(graph, name)

    def cast(x, name):
        if x.dtype == "bfloat16":
            return x
        out = graph.buffer(name, x.shape, "bfloat16")
        graph("mtp_cast", dict(r=x.shape[0], c=x.shape[1]), x, out)
        return out

    def linear(x, name, dest, *, output_role="scratch"):
        w = weight(name)
        inputs, columns = x.shape[1], w.shape[0]
        out = graph.buffer(dest, (x.shape[0], columns), "float32", role=output_role)
        graph(
            "bf16_linear",
            dict(r=x.shape[0], k=inputs, o=columns),
            cast(x, dest + ".cast"),
            w,
            out,
        )
        return out

    def norm(x, name, dest, *, dtype="bfloat16"):
        out = graph.buffer(dest, x.shape, dtype)
        graph(
            "rms",
            dict(
                r=x.shape[0],
                c=width,
                eps=config.epsilon,
                flat=True,
                input_dtype=x.dtype,
                weight_dtype="bfloat16",
                output_dtype=dtype,
            ),
            x.view((prod(x.shape),)),
            weight(name),
            out.view((prod(x.shape),)),
        )
        return out

    def join(left, right, name):
        size = left.shape[1] + right.shape[1]
        out = graph.buffer(name, (count, size), "float32")
        graph(
            "concatenate",
            dict(r=count, left=left.shape[1], right=right.shape[1]),
            left.view((prod(left.shape),)),
            right.view((prod(right.shape),)),
            out.view((count * size,)),
        )
        return out

    def add(left, right, name):
        out = graph.buffer(name, (count, width), "float32")
        shape = (count * width,)
        graph(
            "add",
            dict(r=count, c=width),
            left.view(shape),
            right.view(shape),
            out.view(shape),
        )
        return out

    banks = {}

    def bank(sources, name):
        result = sources[0]
        for i, source in enumerate(sources[1:], 1):
            prefix = sources[: i + 1]
            if reuse_residual_banks and prefix in banks:
                result = banks[prefix]
            else:
                result = join(result, source, name + f".{i}")
                if reuse_residual_banks:
                    banks[prefix] = result
        return result.view((count, len(sources), width))

    def mix(stack, query, index, name):
        out = graph.buffer(name, (count, width), "float32")
        w = weight(query)
        graph(
            "normalized_block_mix",
            dict(
                r=count,
                c=width,
                blocks=stack.shape[1],
                queries=prod(w.shape) // width,
                index=index,
                eps=config.epsilon,
            ),
            stack,
            w.view((prod(w.shape) // width, width)),
            out,
        )
        return out

    def head_norm(
        x,
        heads,
        depth,
        name,
        *,
        mode="rms",
        scale=None,
        clip=None,
        position_scale=False,
    ):
        out = graph.buffer(name, x.shape, "float32")
        params = dict(
            r=count,
            h=heads,
            d=depth,
            eps=config.epsilon,
            scaled=scale is not None,
            mode=mode,
        )
        if clip is not None:
            params["clip"] = clip
        if position_scale:
            params.update(log_floor=config.log_floor, log_alpha=config.log_alpha)
        graph(
            "projected_head_norm",
            params,
            x,
            *((weight(scale),) if scale is not None else ()),
            *((row_positions,) if position_scale else ()),
            out,
        )
        return out

    def blend(x, previous, layer, stream, dest):
        control = f"layers.{layer}.controllers.{stream}."
        if stream in ("k", "v"):
            query = "middle_cycle_cross_pass_kv_queries"
            index = 2 * (layer - config.loop_start) + (stream == "v")
        else:
            query = "middle_cycle_cross_pass_queries"
            index = 2 * (layer - config.loop_start) + (stream == "mlp")
        message = mix(previous, query, index, dest + ".message")
        left = head_norm(x, 1, width, dest + ".dest_norm")
        right = head_norm(message, 1, width, dest + ".message_norm")
        rank = linear(
            join(left, right, dest + ".pair"), control + "down_weight", dest + ".rank"
        )
        forget = linear(rank, control + "forget_weight", dest + ".forget")
        gain = linear(rank, control + "gain_weight", dest + ".gain")
        out = graph.buffer(dest, x.shape, "float32")
        graph("message_blend", dict(r=count, c=width), x, message, forget, gain, out)
        return out

    def conv(x, source, name, *, groups=1, dynamic_input=None):
        dynamic = dynamic_input is not None
        channels = x.shape[1]
        w = weight(source + (".static_w" if dynamic else ".weight"))
        dynamic_w = (
            linear(dynamic_input, source + ".weight_proj", name + ".dynamic")
            if dynamic
            else x
        )
        state = graph.buffer(
            name + ".conv_state",
            (rows, config.taps - 1, channels),
            "float32",
            role="state",
        )
        out = graph.buffer(name, x.shape, "float32")
        graph(
            "dynamic_residual_short_conv" if dynamic else "residual_short_conv",
            dict(
                r=rows,
                chunk=chunk,
                c=channels,
                taps=config.taps,
                groups=groups,
                dynamic=dynamic,
            ),
            x,
            w if dynamic else w.view((channels, config.taps)),
            *((dynamic_w,) if dynamic else ()),
            state,
            lengths,
            out,
        )
        return out

    def attention(inputs, source, dest, local):
        qx, kx, vx = inputs
        h, kh, d = config.heads, config.kv_heads, config.head_dim
        capacity = min(max_context, config.window) if local else max_context
        extent = config.window if local else config.relative_extent
        q = linear(qx, source + "wq_du.weight", dest + ".q_projection")
        q = head_norm(
            q,
            h,
            d,
            dest + ".q",
            mode="clip",
            scale=source + "q_l2_clip.scale",
            clip=config.qk_clip,
            position_scale=not local,
        )
        k = linear(kx, source + "wk_dv.weight", dest + ".k_projection")
        v = linear(vx, source + "wv_dv.weight", dest + ".v_projection")
        k = conv(
            k,
            source + "k_sconv",
            dest + ".k_conv",
            groups=kh,
            dynamic_input=None if local else kx,
        )
        v = conv(
            v,
            source + "v_sconv",
            dest + ".v_conv",
            groups=kh,
            dynamic_input=None if local else vx,
        )
        k = head_norm(k, kh, d, dest + ".key", mode="clip", clip=config.qk_clip)
        v = head_norm(v, kh, d, dest + ".value")
        rel = linear(qx, source + "wr_du.weight", dest + ".rel_projection")
        rel = head_norm(
            rel,
            h,
            config.relative_dim,
            dest + ".rel",
            mode="clip",
            scale=source + "rel_l2_clip.scale",
            clip=config.relative_clip,
            position_scale=not local,
        )
        sinks = linear(qx, source + "attn_sink_proj.weight", dest + ".sinks")
        kc, vc = (
            graph.buffer(
                dest + "." + name, (rows, kh, capacity, d), "bfloat16", role="state"
            )
            for name in ("keys", "values")
        )
        tile = 64
        splits = (capacity + tile - 1) // tile
        p = dict(r=count, h=h, kh=kh, d=d, capacity=capacity, chunk=chunk)
        attended = graph.buffer(dest + ".attended", (count, h * d), "float32")
        bias = weight(source + "rel_logits_proj.proj")
        params = dict(**p, relative_dim=config.relative_dim, extent=extent)
        operands = (q, k, v, kc, vc, rel, bias, sinks, row_positions, row_active)
        if chunk > 1:
            graph("relative_prefill_attention", params, *operands, attended)
        else:
            parts = graph.buffer(
                dest + ".parts", (count * h * splits * (d + 2),), "float32"
            )
            graph(
                "relative_decode_grouped",
                dict(**params, tile=tile, splits=splits),
                *operands,
                parts,
            )
            graph(
                "window_merge",
                dict(r=count, h=h, d=d, splits=splits),
                parts,
                attended.view((count * h * d,)),
                row_active,
            )
        graph("commit_relative_cache", p, k, v, kc, vc, row_positions, row_active)
        gate = linear(qx, source + "gate_attn.weight", dest + ".gate")
        gated = graph.buffer(dest + ".gated", attended.shape, "float32")
        flat = (count * h * d,)
        graph(
            "head_gate",
            dict(r=count, h=h * d, d=1),
            attended.view(flat),
            gate.view(flat),
            gated.view(flat),
        )
        return linear(gated, source + "wo_ud.weight", dest + ".projection")

    def moe(x, source, dest):
        from akas.ops.moe.definition import tile_layouts

        experts, top, middle = config.experts, config.top_k, config.expert_width
        logits = linear(x, source + "gate.weight", dest + ".router")
        ids = graph.buffer(dest + ".ids", (count, top), "int32")
        probability = graph.buffer(dest + ".probability", (count, top), "float32")
        route_params = dict(r=count, top=top, experts=experts)
        graph(
            "biased_router",
            dict(**route_params, scale=config.route_scale),
            logits,
            weight(source + "gate.bias"),
            ids,
            probability,
        )
        counts = graph.buffer(dest + ".counts", (experts,), "int32")
        routes = graph.buffer(dest + ".routes", (experts, count), "int32")
        graph("quantized_routes", route_params, ids, row_active, counts, routes)
        tile_params = dict(route_params)
        if count * top >= experts * 128:
            tile_params["max_block_m"] = 128
        tile_shape = (tile_layouts(tile_params), (count * top + 3) // 4 + experts)
        tiles = tuple(
            graph.buffer(dest + "." + name, tile_shape, "int32")
            for name in ("tile_experts", "tile_offsets")
        )
        graph("expert_tile_map", tile_params, counts, *tiles)

        def project(input_, part, columns, depth, routed):
            name = source + f"experts.{part}_weight"
            encoding = checkpoint.expert_format(name)
            suffixes = (
                ("_scale", "_amax", "_outer_scale_denominator")
                if encoding == "nvfp4"
                else ("_scale",)
            )
            out = graph.buffer(dest + "." + part, (count, top, columns), "float32")
            graph(
                "block_scaled_experts",
                dict(
                    **tile_params,
                    o=columns,
                    k=depth,
                    encoding=encoding,
                    routed_input=routed,
                ),
                input_,
                weight(name),
                *(weight(name + suffix) for suffix in suffixes),
                counts,
                routes,
                *tiles,
                out,
            )
            return out

        gate_up = project(x, "w13", 2 * middle, width, False)
        activated = graph.buffer(dest + ".activated", (count, top, middle), "bfloat16")
        graph(
            "capped_expert_activation",
            dict(
                r=count,
                top=top,
                c=middle,
                gate_cap=config.gate_cap,
                up_cap=config.up_cap,
            ),
            gate_up,
            probability,
            activated,
        )
        down = project(activated, "w2", width, middle, True)
        out = graph.buffer(dest + ".out", (count, width), "float32")
        graph(
            "routed_expert_sum", dict(r=count, top=top, c=width), down, row_active, out
        )
        return out

    embedded = graph.buffer("embedding", (count, width), "float32")
    graph(
        "embedding",
        dict(r=count, c=width, v=config.padded_vocab, weight_dtype="bfloat16"),
        tokens,
        weight("embed.weight").view((config.padded_vocab * width,)),
        embedded.view((count * width,)),
    )
    partial = norm(embedded, "embed_norm.weight", "embed_norm", dtype="float32")
    blocks = ()
    prelude = previous = None
    for layer, pass_index in config.visits():
        source, dest = f"layers.{layer}.", f"layer.{layer}.pass.{pass_index}"
        if layer == config.loop_start:
            if pass_index == 0:
                prelude = (blocks, partial)
            else:
                previous = bank(
                    (*blocks[len(prelude[0]) + 1 :], partial), dest + ".previous"
                )
                blocks, partial = prelude
        if layer == config.loop_end:
            previous = None

        local = layer in config.local_layers
        if layer:
            stack = bank((*blocks, partial), dest + ".attention_bank")
            q = mix(
                stack, f"attn_res.attn_queries.{layer - 1}", 0, dest + ".attention_mix"
            )
            streams = (
                (q,) * 3
                if local
                else (
                    q,
                    mix(
                        stack,
                        f"attn_res.attn_queries.{layer - 1}",
                        1,
                        dest + ".key_mix",
                    ),
                    mix(
                        stack,
                        f"attn_res.attn_queries.{layer - 1}",
                        2,
                        dest + ".value_mix",
                    ),
                )
            )
        else:
            streams = (partial,) * 3
        if previous is not None:
            streams = tuple(
                blend(x, previous, layer, key, dest + f".{key}_blend")
                for x, key in zip(streams, ("attn", "k", "v"), strict=True)
            )
        if local and previous is None:
            inp = norm(streams[0], source + "attn_norm.weight", dest + ".attn_norm")
            inputs = (inp,) * 3
        else:
            inputs = tuple(
                norm(
                    x,
                    source
                    + ("attn_norm.weight" if local else f"{key}_input_norm.weight"),
                    dest + f".{key}_norm",
                )
                for x, key in zip(streams, ("q", "k", "v"), strict=True)
            )
        attended = attention(inputs, source + "attn.", dest + ".attn", local)
        attended = conv(attended, source + "attn_sconv", dest + ".attn_sconv")
        if layer % config.block_size == 0:
            blocks = (*blocks, partial)
            residual = attended
        else:
            residual = add(partial, attended, dest + ".attn_residual")

        mlp_bank = bank((*blocks, residual), dest + ".mlp_bank")
        inp = mix(mlp_bank, f"attn_res.mlp_queries.{layer}", 0, dest + ".mlp_mix")
        if previous is not None:
            inp = blend(inp, previous, layer, "mlp", dest + ".mlp_blend")
        inp = norm(inp, source + "mlp_norm.weight", dest + ".mlp_norm")
        mixed = moe(inp, source + "mlp.", dest + ".mlp")
        mixed = conv(mixed, source + "mlp_sconv", dest + ".mlp_sconv")
        partial = add(residual, mixed, dest + ".output")

    final = mix(
        bank((*blocks, partial), "output_bank"),
        "attn_res.output_query",
        0,
        "output_mix",
    )
    if chunk > 1:
        last = graph.buffer("last_hidden", (rows, width), "float32")
        graph(
            "chunk_last",
            dict(r=rows, c=width, chunk=chunk),
            final.view((count * width,)),
            last.view((rows * width,)),
            lengths,
        )
        final = last
    hidden = norm(final, "norm.weight", "output_norm")
    raw_logits = linear(hidden, "unembed.weight", "raw_logits")
    logits = graph.buffer("logits", (rows * config.vocab,), "float32", role="output")
    graph(
        "logit_softcap",
        dict(r=rows, c=config.vocab, cap=config.logit_cap),
        raw_logits.view((rows * config.vocab,)),
        logits,
    )
    sampled = graph.buffer("sampled", (rows,), "int32", role="output")
    graph("argmax", dict(r=rows, c=config.vocab), logits, sampled, lengths)
    graph("advance", dict(r=rows, chunk=chunk), positions, lengths)

    graph.resize_state_rows(max_requests)
    graph.buffer("mapping", (rows,), "int32", role="input")
    graph.bind_requests(
        inputs=dict(tokens="tokens", mapping="mapping", lengths="active"),
        features=dict(logits="logits"),
        prediction="sampled",
        reset=[
            name
            for name in graph.buffers
            if name == "positions" or name.endswith(".conv_state")
        ],
    )
    return Composition(graph)


class Inkling2Mini(Definition):
    checkpoint_type = Inkling2MiniCheckpoint
    checkpoint_architectures = ("inkling2_loop_text",)
    model_defaults = {"kv_dtype": "bfloat16"}
    optimization_chunks = (2048, 128, 256, 64, 32, 1)
