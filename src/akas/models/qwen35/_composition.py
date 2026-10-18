"""Qwen layer graph construction; public entry is model.compose."""

from akas.models.blocks import DecoderArchitecture, DecoderBlock
from akas.models.qwen35.config import Qwen35MoEConfig


def architecture(config, checkpoint):
    feed_forward = "moe" if isinstance(config, Qwen35MoEConfig) else "dense"
    blocks = []
    for index, kind in enumerate(config.layers):
        prefix = f"model.language_model.layers.{index}."
        blocks.append(
            DecoderBlock(
                index,
                prefix,
                kind,
                prefix
                + ("linear_attn." if kind == "linear_attention" else "self_attn."),
                feed_forward,
                prefix + "mlp.",
                prefix + "input_layernorm",
                prefix + "post_attention_layernorm",
            )
        )
    embedding = "model.language_model.embed_tokens.weight"
    return DecoderArchitecture(
        tuple(blocks),
        embedding,
        "model.language_model.norm.weight",
        getattr(
            checkpoint,
            "head_name",
            "lm_head.weight" if feed_forward == "moe" else embedding,
        ),
    )


def compose_dense(
    checkpoint,
    *,
    slots=1,
    chunk=1,
    pool=256,
    context=4096,
    provider="cuda",
    kv_dtype=None,
    verify=False,
    fused_head=False,
    draft=False,
):
    """Declare native dense buffers and record operations without opening a device.

    Projection, normalization and stateful mixers own their scratch declarations.
    The shared compiler plans lifetimes and the executor binds request slots.
    """
    from types import SimpleNamespace

    from akas.graph import Graph
    from akas.memory.layouts import hybrid_state_specs

    act_dtype = "float16" if provider == "webgpu" else "bfloat16"
    if kv_dtype is None or (provider == "webgpu" and kv_dtype == "bfloat16"):
        kv_dtype = act_dtype
    graph = Graph(contract="bf16", phase="prefill" if chunk > 1 else "decode")
    commit = Graph(contract="bf16", phase=graph.phase)
    from dataclasses import replace

    config = (
        replace(checkpoint.config, layers=("full_attention",))
        if draft
        else checkpoint.config
    )
    namespace = "proposal." if draft else ""
    weights = (
        checkpoint.graph_weights(
            graph, provider=provider, branch="mtp" if draft else "text"
        )
        if getattr(checkpoint, "storage_format", None) == "ggml"
        or provider == "webgpu"
        else {
            name: graph.buffer(name, shape, dtype, role="weight")
            for name, (shape, dtype) in checkpoint.storage_specs(
                branch="mtp" if draft else "text"
            ).items()
        }
    )
    if draft:
        weights = {
            "model.language_model.norm.weight"
            if name == "mtp.norm.weight"
            else name.replace("mtp.layers.0.", "model.language_model.layers.0."): value
            for name, value in weights.items()
        }
        embedding = "model.language_model.embed_tokens.weight"
        if embedding not in weights:
            shape, dtype = checkpoint.storage_specs()[embedding]
            weights[embedding] = graph.buffer(embedding, shape, dtype, role="weight")
    states = {
        layer: [
            graph.buffer(
                f"{namespace}state.{layer}.{index}",
                spec.shape,
                spec.dtype,
                role="state",
            )
            for index, spec in enumerate(specs)
        ]
        for layer, specs in hybrid_state_specs(
            config, pool, context, kv_dtype=kv_dtype, provider=provider
        ).items()
    }
    owner = SimpleNamespace(
        config=config,
        weights=weights,
        capacity=pool,
        context=context,
        states=states,
        fused_head=fused_head,
        architecture=architecture(config, checkpoint),
    )
    if draft:
        owner.hidden = graph.buffer(
            "proposal.saved_hidden", (pool, config.width), act_dtype, role="state"
        )
    from akas.ops.attention.definition import attend
    from akas.ops.gdn.definition import gated_delta
    from akas.ops.linear.definition import project
    from akas.ops.linear.definition import schedule
    from akas.ops.normalization.definition import normalize

    rows = slots * chunk
    head_rows = rows if verify else slots
    inputs = {
        name: graph.buffer(
            namespace + name,
            (
                rows
                if name == "tokens"
                else head_rows
                if name in ("head_position", "head_active")
                else slots,
            ),
            "int32",
            role="input",
        )
        for name in (
            "tokens",
            "mapping",
            "position",
            "lengths",
            "head_position",
            "head_active",
        )
    }
    flat_positions = graph.buffer(namespace + "flat_position", (rows,), "int32")
    active = graph.buffer(namespace + "active", (rows,), "int32")
    common = dict(slots=slots, chunk=chunk, pool=pool)
    graph(
        "controls",
        common,
        inputs["position"],
        inputs["lengths"],
        flat_positions,
        active,
    )
    controls = dict(
        parameters=common,
        mapping=inputs["mapping"],
        positions=inputs["position"],
        lengths=inputs["lengths"],
        flat_positions=flat_positions,
        active=active,
    )
    residual = graph.buffer(namespace + "residual", (rows, config.width), act_dtype)
    from akas.ops.vocabulary.definition import embed

    embed(
        graph,
        inputs["tokens"],
        weights[owner.architecture.embedding],
        residual,
        rows=rows,
        width=config.width,
        vocab=config.vocab,
    )
    features = {}
    if draft:
        hidden = graph.buffer(
            "proposal.hidden", (rows, config.width), act_dtype, role="input"
        )
        mode = graph.buffer("proposal.mode", (1,), "int32", role="input")
        graph(
            "gather_hidden",
            dict(common, c=config.width),
            owner.hidden,
            inputs["mapping"],
            mode,
            hidden,
        )
        joined = graph.buffer("proposal.join", (rows, 2 * config.width), act_dtype)
        graph(
            "mtp_join",
            dict(r=rows, c=config.width, eps=config.epsilon),
            residual,
            hidden,
            weights["mtp.pre_fc_norm_embedding.weight"],
            weights["mtp.pre_fc_norm_hidden.weight"],
            joined,
        )
        projected = project(
            graph,
            joined,
            weights["mtp.fc.weight"],
            width=config.width,
            name="proposal.fc",
        )
        graph("mtp_cast", dict(r=rows, c=config.width), projected, residual)
        features.update(hidden_input=hidden.name, mode=mode.name)
    previous = None
    for block in owner.architecture.blocks:
        normal = normalize(
            graph,
            residual if previous is None else previous,
            weights[block.input_norm + ".weight"],
            epsilon=config.epsilon,
            name=namespace + block.prefix + "input_normal",
            residual=None if previous is None else residual,
            dtype=act_dtype,
        )
        prefix = block.mixer_prefix
        if block.mixer == "linear_attention":
            mixed = gated_delta(
                graph,
                normal,
                weights,
                states[block.index],
                controls,
                config=config,
                prefix=prefix,
                verify=verify,
                commit=commit,
                dtype=act_dtype,
            )
        else:
            mixed = attend(
                graph,
                normal,
                weights,
                states[block.index],
                controls,
                config=config,
                prefix=prefix,
                context=context,
                dtype=act_dtype,
            )
        normal = normalize(
            graph,
            mixed,
            weights[block.post_norm + ".weight"],
            epsilon=config.epsilon,
            name=namespace + block.prefix + "post_normal",
            residual=residual,
            dtype=act_dtype,
        )
        prefix = block.feed_forward_prefix
        if block.feed_forward == "moe":
            from akas.ops.moe.definition import encoded_moe

            previous = encoded_moe(
                graph,
                normal,
                weights,
                config=config,
                prefix=prefix,
                active=controls["active"],
            )
            continue
        projected = project(
            graph,
            normal,
            weights[prefix + "input.weight"],
            width=config.width,
            name=prefix + "projection",
        )
        activated = graph.buffer(
            prefix + "activated", (rows, config.intermediate), act_dtype
        )
        graph("swiglu", dict(common, c=config.intermediate), projected, activated)
        previous = project(
            graph,
            activated,
            weights[prefix + "down_proj.weight"],
            width=config.width,
            name=prefix + "output",
        )
    normal = normalize(
        graph,
        previous,
        weights[owner.architecture.final_norm],
        epsilon=config.epsilon,
        name=namespace + "normal",
        residual=residual,
        role="output",
        dtype=act_dtype,
    )
    features["hidden"] = normal.name
    head_input = normal
    if not verify:
        head_input = graph.buffer(namespace + "last", (slots, config.width), act_dtype)
        graph(
            "last_rows",
            dict(common, c=config.width),
            normal,
            inputs["lengths"],
            head_input,
        )
    if draft:
        graph(
            "store_hidden",
            dict(common, c=config.width),
            head_input,
            inputs["mapping"],
            inputs["lengths"],
            owner.hidden,
        )
    predicted = graph.buffer(
        namespace + "predicted", (head_rows,), "int32", role="output"
    )
    if fused_head:
        tiles = (config.vocab + 63) // 64
        maximum = graph.buffer(
            namespace + "head_maximum", (head_rows, tiles), "float32"
        )
        index = graph.buffer(namespace + "head_index", (head_rows, tiles), "int32")
        graph(
            "head_linear",
            schedule(head_rows, config.width, config.vocab, config.width),
            head_input,
            weights[owner.architecture.head],
            maximum,
            index,
        )
        graph(
            "head_argmax",
            dict(r=head_rows, tiles=tiles, vocab=config.vocab),
            maximum,
            index,
            inputs["head_active"],
            predicted,
        )
    else:
        logits = project(
            graph,
            head_input,
            weights[owner.architecture.head],
            width=config.width,
            name=namespace + "logits",
        )
        graph(
            "argmax",
            dict(r=head_rows, vocab=config.vocab),
            logits,
            predicted,
            inputs["head_position"],
            inputs["head_active"],
        )
    graph.bind_requests(
        inputs={
            field: inputs[name].name
            for field, name in (
                ("tokens", "tokens"),
                ("mapping", "mapping"),
                ("positions", "position"),
                ("lengths", "lengths"),
                ("head_positions", "head_position"),
                ("head_active", "head_active"),
            )
        },
        prediction=predicted.name,
        reset=["proposal.saved_hidden"]
        if draft
        else [
            name
            for name in graph.buffers
            if name.startswith("state.")
            and config.layers[int(name.split(".")[1])] == "linear_attention"
        ],
        features=features,
    )
    return (graph, commit)


def compose_moe_decode(checkpoint, **options):
    return compose_moe_program(checkpoint, chunk=1, **options)


def compose_moe_program(
    checkpoint,
    *,
    rows,
    chunk,
    max_requests,
    max_context,
    provider="cuda",
    kv_dtype="bfloat16",
    splits=16,
    phase=None,
):
    """Compose native MoE phases directly; operations own their workspaces."""
    from types import SimpleNamespace

    from akas.graph import Graph
    from akas.memory.layouts import hybrid_state_specs
    import akas.ops.attention.definition as attention
    from akas.ops.attention.definition import projected_workspace
    import akas.ops.gdn.definition as gdn
    from akas.ops.gdn.definition import separate_workspace
    from akas.ops.linear.definition import linear
    import akas.ops.moe.definition as moe
    from akas.ops.moe.definition import routed_workspace
    from akas.ops.normalization.definition import rms_norm
    from akas.compilation.quantization import FP8ActivationCache
    from akas.quantization import get

    if provider != "cuda":
        raise ValueError("native MoE kernels currently require CUDA")
    if phase not in (None, "decode", "prefill", "verify", "proposal"):
        raise ValueError("native MoE proposal graph migration is pending")
    verify = phase == "verify"
    if verify and chunk < 2:
        raise ValueError("verification requires a window of at least two tokens")
    proposal = phase == "proposal"
    prefill, count = (chunk > 1, rows * chunk)
    prefix = ("proposal." if proposal else "") + ("chunk." if prefill else "")
    graph = Graph(contract="block-fp8", phase="prefill" if prefill else "decode")
    config = checkpoint.config
    if proposal:
        from dataclasses import replace

        config = replace(config, layers=("full_attention",))
    specs = checkpoint.storage_specs(
        pack_experts=True, **{"branch": "all"} if proposal else {}
    )
    weights = {
        name: graph.buffer(name, shape, dtype, role="weight")
        for name, (shape, dtype) in specs.items()
        if not proposal
        or name.startswith("mtp.")
        or name in ("model.language_model.embed_tokens.weight", "lm_head.weight")
    }
    if proposal:
        weights = {
            "model.language_model.norm.weight"
            if name == "mtp.norm.weight"
            else name.replace("mtp.layers.0.", "model.language_model.layers.0."): value
            for name, value in weights.items()
        }
    states = {
        layer: [
            graph.buffer(
                f"{('proposal.' if proposal else '')}state.{layer}.{index}",
                spec.shape,
                spec.dtype,
                role="state",
            )
            for index, spec in enumerate(specs)
        ]
        for layer, specs in hybrid_state_specs(
            config, max_requests, max_context, kv_dtype=kv_dtype
        ).items()
    }
    controls = {
        name: graph.buffer(
            name,
            (max_requests if name == "positions" else rows,),
            "int32",
            role="state" if name == "positions" else "input",
        )
        for name in ("tokens", "positions", "active")
    }
    if proposal:
        for name in tuple(controls):
            old = controls[name]
            del graph.buffers[old.name]
            controls[name] = graph.buffer(
                "proposal." + name, (rows,), "int32", role="input"
            )
    elif verify:
        controls["positions"] = graph.buffer(
            "verify.positions", (rows,), "int32", role="input"
        )
    if proposal:
        controls["head_positions"] = graph.buffer(
            "proposal.head_positions", (rows,), "int32", role="input"
        )
    logits = graph.buffer(
        "logits", (count if verify else rows, config.vocab), "float32"
    )
    graph.buffer("mapping", (rows,), "int32", role="input")
    b = {
        name: graph.buffer(
            prefix + name,
            (count, config.width),
            "bfloat16" if name in ("residual", "normal") else "float32",
            role="output" if name == "normal" else "scratch",
        )
        for name in ("residual", "normal", "mixed", "ffn")
    }
    b.update(separate_workspace(graph, count, config, prefix=prefix))
    b.update(
        projected_workspace(
            graph, count, config, prefix=prefix, splits=None if prefill else splits
        )
    )
    b.update(routed_workspace(graph, count, config, prefix=prefix, grouped=prefill))
    owner = SimpleNamespace(
        config=config,
        weights=weights,
        states=states,
        slots=rows,
        context=max_context,
        kv_dtype=kv_dtype,
        splits=splits,
        architecture=architecture(config, checkpoint),
        buffers=dict(controls),
    )
    if prefill:
        owner.buffers["normal"] = graph.buffer(
            "proposal.last" if proposal else "normal", (rows, config.width), "bfloat16"
        )
        b["tokens"] = graph.buffer(prefix + "tokens", (count,), "int32", role="input")
        b["lengths"] = graph.buffer(prefix + "lengths", (rows,), "int32", role="input")
        for name in ("flat_positions", "flat_active"):
            b[name] = graph.buffer(prefix + name, (count,), "int32")
        quantized = {}
        for name in (
            "normal",
            "gdn_normal",
            "att_out",
            "shared_activation",
            "expert_activation",
        ):
            source = b[name]
            storage = get("activations", "fp8").storage(source.shape, provider=provider)
            quantized[name] = (
                graph.buffer(
                    prefix + "quantized_" + name,
                    storage.values.shape,
                    storage.values.dtype,
                ),
                graph.buffer(
                    prefix + "scales_" + name,
                    storage.named["scales"].shape,
                    storage.named["scales"].dtype,
                ),
            )
        cache = FP8ActivationCache(b, quantized, rows=count)
    else:
        b.update(controls)
        owner.buffers.update(b)
        cache = None

    def emit(kind, parameters, *arguments, label=None, **bindings):
        if prefill and kind == "fp8_linear":
            parameters = {**parameters, "prequantized": True}
        graph(kind, parameters, *arguments, label=label, **bindings)
        if cache is not None and "out" in bindings:
            cache.invalidate(bindings["out"])

    def resolve(value):
        return b[value] if isinstance(value, str) else value

    def fp8(name, x, out):
        bits, scale = cache.get(x, emit) if cache is not None else (resolve(x), None)
        linear(
            emit,
            bits,
            weights[name + ".weight"],
            resolve(out),
            rows=count,
            scales=weights[name + ".weight_scale_inv"],
            activation_scales=scale,
        )

    def bf16(name, x, out):
        linear(emit, resolve(x), weights[name + ".weight"], resolve(out), rows=count)

    def norm(name, x=None, label=None):
        rms_norm(
            emit,
            b["residual"] if x is None else resolve(x),
            weights[name + ".weight"],
            b["normal"],
            rows=count,
            width=config.width,
            epsilon=config.epsilon,
            residual=None if x is None else b["residual"],
            input_name="residual",
            label=label,
        )

    def small(kind, **bindings):
        emit(kind, dict(slots=rows, chunk=chunk), **bindings)

    def experts(name, x, out, o, k, routed):
        bits, scale = cache.get(x, emit)
        emit(
            "fp8_experts",
            dict(
                r=count,
                experts=config.experts,
                top=config.top_k,
                o=o,
                k=k,
                routed_input=routed,
            ),
            x=bits,
            activation_scales=scale,
            w=weights[name + ".weight"],
            scales=weights[name + ".weight_scale_inv"],
            counts=b["counts"],
            routes=b["routes"],
            out=b[out],
        )

    if prefill:
        small(
            "controls",
            position=controls["positions"],
            lengths=b["lengths"],
            flat_position=b["flat_positions"],
            active=b["flat_active"],
        )
        emit(
            "embedding",
            dict(r=count, c=config.width, vocab=config.vocab),
            ids=b["tokens"],
            w=weights[owner.architecture.embedding],
            out=b["residual"],
        )
    else:
        emit(
            "embedding",
            dict(r=count, c=config.width, vocab=config.vocab),
            controls["tokens"],
            weights[owner.architecture.embedding],
            b["residual"],
        )
    features = {"hidden": b["normal"].name}
    if proposal:
        saved_hidden = graph.buffer(
            "proposal.saved_hidden",
            (max_requests, config.width),
            "bfloat16",
            role="state",
        )
        hidden = graph.buffer(
            "proposal.hidden", (count, config.width), "bfloat16", role="input"
        )
        mode = graph.buffer("proposal.mode", (1,), "int32", role="input")
        graph(
            "gather_hidden",
            dict(slots=rows, chunk=chunk, pool=max_requests, c=config.width),
            saved_hidden,
            graph.buffers["mapping"],
            mode,
            hidden,
        )
        joined = graph.buffer("proposal.join", (count, 2 * config.width), "bfloat16")
        projected = graph.buffer("proposal.projected", (count, config.width), "float32")
        graph(
            "mtp_join",
            dict(r=count, c=config.width, eps=config.epsilon),
            b["residual"],
            hidden,
            weights["mtp.pre_fc_norm_embedding.weight"],
            weights["mtp.pre_fc_norm_hidden.weight"],
            joined,
        )
        linear(emit, joined, weights["mtp.fc.weight"], projected, rows=count)
        graph("mtp_cast", dict(r=count, c=config.width), projected, b["residual"])
        features.update(hidden_input=hidden.name, mode=mode.name)
    ctx = SimpleNamespace(
        executor=SimpleNamespace(chunk=chunk) if prefill else owner,
        model=owner,
        weights=weights,
        buffers=b,
        rows=count,
        slots=rows,
        emit=emit,
        fp8=fp8,
        bf16=bf16,
        norm=norm,
        small=small,
        experts=experts,
        mixers={
            "linear_attention": gdn.fp8_prefill_gdn if prefill else gdn.fp8_decode_gdn,
            "full_attention": attention.fp8_prefill_attention
            if prefill
            else attention.fp8_decode_attention,
        },
        feed_forwards={"moe": moe.fp8_prefill_moe if prefill else moe.fp8_decode_moe},
    )
    ctx.input_norm = lambda context, block: norm(
        block.input_norm,
        None if block.index == 0 else b["ffn"],
        None if prefill else (block.index, "input"),
    )
    ctx.post_norm = lambda context, block: norm(
        block.post_norm,
        b["mixed"],
        None if prefill else (block.index, "post_attention"),
    )
    for block in owner.architecture.blocks:
        block.compose(ctx)
    norm(
        owner.architecture.final_norm.removesuffix(".weight"),
        b["ffn"],
        None if prefill else (len(owner.architecture.blocks), "final"),
    )
    if prefill and (not verify):
        emit(
            "last_rows",
            dict(slots=rows, chunk=chunk, width=config.width),
            x=b["normal"],
            lengths=b["lengths"],
            out=owner.buffers["normal"],
        )
    head_input = owner.buffers["normal"] if prefill and (not verify) else b["normal"]
    if proposal:
        graph(
            "store_hidden",
            dict(slots=rows, chunk=chunk, pool=max_requests, c=config.width),
            head_input,
            graph.buffers["mapping"],
            b["lengths"] if prefill else controls["active"],
            saved_hidden,
        )
    emit(
        "bf16_linear",
        dict(
            r=count if verify else rows,
            k=config.width,
            o=config.vocab,
            **{"decode_head": True} if prefill else {},
        ),
        x=head_input,
        w=weights[owner.architecture.head],
        out=logits,
    )
    prediction = (
        graph.buffer("verify.predicted", (count,), "int32", role="output")
        if verify
        else controls["tokens"]
    )
    emit(
        "argmax",
        dict(r=count if verify else rows, vocab=config.vocab),
        logits,
        prediction,
        b["flat_positions"]
        if verify
        else controls.get("head_positions", controls["positions"]),
        b["flat_active"] if verify else controls["active"],
    )
    if prefill and (not verify):
        small(
            "advance",
            position=controls["positions"],
            lengths=b["lengths"],
            active=controls["active"],
        )
    resets = ([] if proposal else ["positions"]) + [
        value.name
        for layer, values in states.items()
        if config.layers[layer] == "linear_attention"
        for value in values
    ]
    if proposal:
        resets.append("proposal.saved_hidden")
    graph.bind_requests(
        inputs={
            "tokens": b["tokens"].name,
            "mapping": "mapping",
            "lengths": b["lengths"].name if prefill else controls["active"].name,
            **(
                {
                    "positions": controls["positions"].name,
                    "head_positions": controls["head_positions"].name,
                    "head_active": controls["active"].name,
                }
                if proposal
                else {
                    "positions": "verify.positions",
                    "head_positions": b["flat_positions"].name,
                    "head_active": b["flat_active"].name,
                }
                if verify
                else {"head_active": "active"}
                if prefill
                else {}
            ),
        },
        prediction=prediction.name,
        reset=resets,
        features=features,
    )
    if not verify:
        return graph
    from akas.ops.gdn.definition import journal_prefixes

    commit = Graph(contract="block-fp8", phase="commit")
    journal_prefixes(graph, commit, slots=rows, chunk=chunk, lengths=b["lengths"])
    # Decode reads device-side pool positions; accepted prefixes advance both
    # those positions and the executor's matching host counters.
    small_parameters = dict(slots=rows, chunk=chunk, all_tokens=True)
    commit(
        "advance",
        small_parameters,
        position=graph.buffers["positions"],
        lengths=b["lengths"],
        active=controls["active"],
    )
    return (graph, commit)
