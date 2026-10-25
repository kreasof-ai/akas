"""LFM2 layer graph construction; public entry is model.compose."""


def compose(
    config,
    *,
    batch=1,
    capacity=None,
    provider="cuda",
    kv_dtype="float16",
    kv_padding=0,
    output=False,
    max_weight_bytes=64 * 1024 * 1024,
    chunk=1,
    optimized=False,
    checkpoint=None,
    expert_weight_layout="raw",
    dense_weight_layout="raw",
    verify=False,
    auxiliary_layers=(),
):
    """Compose a chunk of tokens per slot with masked, persistent state."""
    from akas.graph import Graph
    from akas.models.lfm2.weights import tensor_shapes
    from akas.models.lfm2.config import LFM2MoEConfig
    from akas.ops.attention.definition import cache_storage

    capacity = min(4096, config.max_context) if capacity is None else capacity
    if (
        type(batch) is not int
        or batch < 1
        or type(capacity) is not int
        or (not 1 <= capacity <= config.max_context)
        or (type(chunk) is not int)
        or (not 1 <= chunk <= capacity)
    ):
        raise ValueError("batch/capacity must be positive and within model context")
    if provider not in ("cuda", "webgpu"):
        raise ValueError("reference requires CUDA or WebGPU")
    if kv_dtype not in ("float16", "float32"):
        from akas.quantization.kv import get

        get(kv_dtype)
    if type(kv_padding) is not int or not 0 <= kv_padding <= 4096 or kv_padding % 32:
        raise ValueError("KV padding must be a bounded multiple of 32")
    if kv_padding and (provider != "webgpu" or kv_dtype not in ("float16", "float32")):
        raise ValueError("KV padding currently requires native floating WebGPU storage")
    if any(
        type(i) is not int or not 0 <= i < len(config.layers) for i in auxiliary_layers
    ) or len(set(auxiliary_layers)) != len(auxiliary_layers):
        raise ValueError("LFM auxiliary layers must be distinct valid layer indices")
    plan = Graph(contract="reference", phase="verify" if verify else "forward")
    b, c = (batch * chunk, config.width)
    from akas.checkpoints.partition import Weights

    if expert_weight_layout not in ("raw", "planar"):
        raise ValueError("expert weight layout must be raw or planar")
    if dense_weight_layout not in ("raw", "planar"):
        raise ValueError("dense weight layout must be raw or planar")
    if dense_weight_layout != "raw" and (
        provider != "webgpu" or not optimized or not hasattr(checkpoint, "gguf")
    ):
        raise ValueError("dense planar weights require an optimized WebGPU GGUF graph")
    if expert_weight_layout != "raw" and (
        not isinstance(config, LFM2MoEConfig) or checkpoint is None
    ):
        raise ValueError("expert storage layouts require an encoded MoE checkpoint")
    matrix_dtype = "float16" if optimized else "float32"
    weights = (
        checkpoint.graph_weights(
            plan,
            provider=provider,
            **(
                dict(expert_weight_layout=expert_weight_layout)
                if expert_weight_layout != "raw"
                else {}
            ),
            **(dict(dense_weight_layout=dense_weight_layout) if dense_weight_layout != "raw" else {}),
        )
        if checkpoint is not None and hasattr(checkpoint, "graph_weights")
        else Weights(
            plan,
            tensor_shapes(config, output=output),
            max_bytes=max_weight_bytes,
            matrix_dtype=matrix_dtype,
        )
    )
    tokens = plan.buffer("tokens", (batch,), "int32", role="input")
    positions = plan.buffer("positions", (batch,), "int32", role="state")
    active = plan.buffer("active", (batch,), "int32", role="input")
    source_tokens, row_positions, row_active = (tokens, positions, active)
    if chunk > 1:
        source_tokens = plan.buffer("prefill_tokens", (b,), "int32", role="input")
        row_positions = plan.buffer("row_positions", (b,), "int32")
        row_active = plan.buffer("row_active", (b,), "int32")
        plan(
            "chunk_controls",
            dict(r=batch, chunk=chunk),
            positions,
            active,
            row_positions,
            row_active,
        )
    hidden = plan.buffer("hidden", (b * c,), "float32")
    for w, start, rows in weights.matrix("token_embd.weight"):
        plan(
            "embedding",
            dict(
                r=b,
                c=c,
                v=rows,
                start=start,
                **weights.parameters("token_embd.weight")
                if hasattr(weights, "parameters")
                else {"weight_dtype": matrix_dtype}
                if optimized
                else {},
            ),
            source_tokens,
            w,
            hidden,
        )

    def scratch(name, width, *, matrix=False, rows=b):
        return plan.buffer(
            name, (rows, width) if matrix else (rows * width,), "float32"
        )

    def norm(name, x, weight, width=c, rows=b):
        out = scratch(name, width, rows=rows)
        plan("rms", dict(r=rows, c=width, eps=config.epsilon), x, weights[weight], out)
        return out

    def project(x, weight, k, o, out, *, matrix=False, count=b):
        if matrix and hasattr(weights, "parameters"):
            out = out.view((count * o,))
        for w, start, rows in weights.matrix(weight):
            plan(
                "linear",
                dict(
                    r=count,
                    k=k,
                    o=rows,
                    **dict(matrix_output=matrix, output_stride=o, output_offset=start)
                    if not hasattr(weights, "parameters")
                    else {},
                    **{
                        "optimized": True,
                        **(
                            weights.parameters(weight)
                            if hasattr(weights, "parameters")
                            else {"weight_dtype": matrix_dtype}
                        ),
                    }
                    if optimized
                    else {},
                ),
                x,
                w,
                out,
            )

    def linear(name, x, weight, k, o, *, matrix=False):
        out = scratch(name, o, matrix=matrix)
        project(x, weight, k, o, out, matrix=matrix)
        return out

    def residual(name, x, y):
        out = scratch(name, c)
        plan("add", dict(r=b, c=c), x, y, out)
        return out

    for i, kind in enumerate(config.layers):
        p = f"blk.{i}."
        normalized = norm(p + "normalized", hidden, p + "attn_norm.weight")
        if kind == "conv":
            projection = linear(
                p + "projection", normalized, p + "shortconv.in_proj.weight", c, 3 * c
            )
            state = plan.buffer(
                p + "conv_state", (batch, config.conv - 1, c), "float32", role="state"
            )
            mixed = scratch(p + "mixed", c)
            if verify:
                journal = plan.buffer(
                    p + "conv_journal",
                    (batch, chunk, config.conv - 1, c),
                    "float32",
                    role="journal",
                )
                plan(
                    "conv_verify",
                    dict(r=batch, c=c, length=config.conv, chunk=chunk),
                    x=projection,
                    w=weights[p + "shortconv.conv.weight"],
                    state=state,
                    journal=journal,
                    out=mixed,
                    enabled=active,
                )
            else:
                plan(
                    "conv",
                    dict(
                        r=batch,
                        c=c,
                        length=config.conv,
                        **{"chunk": chunk} if chunk > 1 else {},
                    ),
                    projection,
                    weights[p + "shortconv.conv.weight"],
                    state,
                    mixed,
                    active,
                )
            mixed = linear(
                p + "operator_output", mixed, p + "shortconv.out_proj.weight", c, c
            )
        else:
            d, h, kh = (config.head_dim, config.heads, config.kv_heads)
            q = linear(p + "q", normalized, p + "attn_q.weight", c, h * d)
            k = linear(p + "k", normalized, p + "attn_k.weight", c, kh * d)
            v = linear(p + "v", normalized, p + "attn_v.weight", c, kh * d, matrix=True)
            q = norm(p + "qn", q, p + "attn_q_norm.weight", width=d, rows=b * h)
            k = norm(p + "kn", k, p + "attn_k_norm.weight", width=d, rows=b * kh)
            qr, kr = (
                scratch(p + "qr", h * d, matrix=True),
                scratch(p + "kr", kh * d, matrix=True),
            )
            plan(
                "rope",
                dict(
                    r=b,
                    h=h,
                    d=d,
                    theta=config.theta,
                    provider=provider,
                    position_limit=capacity,
                ),
                q,
                qr,
                row_positions,
            )
            plan(
                "rope",
                dict(
                    r=b,
                    h=kh,
                    d=d,
                    theta=config.theta,
                    provider=provider,
                    position_limit=capacity,
                ),
                k,
                kr,
                row_positions,
            )
            params = dict(
                r=b,
                h=h,
                kh=kh,
                d=d,
                capacity=capacity,
                kv_dtype=kv_dtype,
                unit_scales=kv_dtype in ("float16", "float32"),
                provider=provider,
                **({"cache_padding": kv_padding} if kv_padding else {}),
                **{"chunk": chunk} if chunk > 1 else {},
                **{"optimized": True} if optimized else {},
            )
            storage = cache_storage(params)
            kc, vc = [
                plan.buffer(
                    p + name, storage.values.shape, storage.values.dtype, role="state"
                )
                for name in ("kc", "vc")
            ]
            ks, vs = [
                plan.buffer(
                    p + name, storage.metadata[0].shape, "float32", role="state"
                )
                for name in ("ks", "vs")
            ]
            plan(
                "cache_write", params, kr, v, kc, vc, ks, vs, row_positions, row_active
            )
            mixed = scratch(p + "attended", c)
            from akas.ops.attention.definition import floating_attention

            floating_attention(
                plan, params, qr, kc, vc, ks, vs, mixed, row_positions, row_active
            )
            mixed = linear(p + "operator_output", mixed, p + "attn_output.weight", c, c)
        hidden = residual(p + "residual", hidden, mixed)
        normalized = norm(p + "ffn_normalized", hidden, p + "ffn_norm.weight")
        if isinstance(config, LFM2MoEConfig) and i >= config.dense_layers:
            from akas.ops.moe.definition import routed_moe
            from akas.quantization.weights import Weight

            # Router logits choose discrete experts; preserve their F32 operands.
            router = scratch(p + "router", config.experts)
            plan(
                "linear",
                dict(r=b, k=c, o=config.experts),
                normalized,
                weights[p + "ffn_gate_inp.weight"],
                router,
            )
            expert_weights = {}
            for part, shape in (
                ("gate", (config.experts, config.expert_width, c)),
                ("up", (config.experts, config.expert_width, c)),
                ("down", (config.experts, c, config.expert_width)),
            ):
                name = p + "ffn_" + part + "_exps.weight"
                expert_weights[part] = Weight(
                    shape, (weights[name],), weights.parts(name)
                )
            mixed = routed_moe(
                plan,
                normalized.view((b, c)),
                router.view((b, config.experts)),
                weights[p + "exp_probs_b.bias"],
                expert_weights,
                top=config.top_k,
                scale=config.routing_scale,
                normalize=config.normalize_routes,
                provider=provider,
                active=row_active,
                prefix=p + "moe.",
            ).view((b * c,))
        else:
            gate = linear(p + "gate", normalized, p + "ffn_gate.weight", c, config.ff)
            up = linear(p + "up", normalized, p + "ffn_up.weight", c, config.ff)
            activated = scratch(p + "activated", config.ff)
            plan("swiglu", dict(r=b, c=config.ff), gate, up, activated)
            mixed = linear(
                p + "ffn_output", activated, p + "ffn_down.weight", config.ff, c
            )
        hidden = residual(p + "output", hidden, mixed)
        if i in auxiliary_layers:
            from dataclasses import replace

            hidden = replace(hidden, role="output")
            object.__setattr__(hidden, "_graph", plan)
            plan.buffers[hidden.name] = hidden
    if chunk > 1 and not verify:
        final = scratch("last_hidden", c, rows=batch)
        plan("chunk_last", dict(r=batch, c=c, chunk=chunk), hidden, final, active)
        hidden = final
    head_rows = b if verify else batch
    hidden = norm("normalized_output", hidden, "token_embd_norm.weight", rows=head_rows)
    head = "output.weight" if output else "token_embd.weight"
    logits = plan.buffer(
        "logits", (head_rows * config.vocab,), "float32", role="output"
    )
    projected = (
        scratch("chunk_logits", config.vocab, rows=batch)
        if chunk > 1 and not verify
        else logits
    )
    project(hidden, head, c, config.vocab, projected, count=head_rows)
    if chunk > 1 and not verify:
        plan("masked_copy", dict(r=batch, c=config.vocab), projected, logits, active)
    sampled = plan.buffer("sampled", (head_rows,), "int32", role="output")
    plan(
        "argmax",
        dict(r=head_rows, c=config.vocab),
        logits,
        sampled,
        row_active if verify else active,
    )
    if not verify:
        plan("next_tokens", dict(r=batch), sampled, tokens)
        plan(
            "advance",
            dict(r=batch, **{"chunk": chunk} if chunk > 1 else {}),
            positions,
            active,
        )
    if optimized:
        from akas.compilation.rewrites import optimize

        return optimize(plan)
    return plan


def program(
    checkpoint,
    *,
    rows,
    chunk,
    max_requests,
    max_context,
    provider="cuda",
    kv_dtype="float16",
    kv_padding=0,
    max_weight_bytes=64 * 1024 * 1024,
    optimized=True,
    activation_quantization=None,
    expert_weight_layout="raw",
    dense_weight_layout="raw",
    auxiliary_layers=(),
    phase=None,
):
    """Compose token phases over the shared executor's persistent slot pool."""
    from dataclasses import replace

    graph = compose(
        checkpoint.config,
        batch=rows,
        chunk=chunk,
        capacity=max_context,
        provider=provider,
        kv_dtype=kv_dtype,
        kv_padding=kv_padding,
        max_weight_bytes=max_weight_bytes,
        optimized=optimized,
        checkpoint=checkpoint,
        expert_weight_layout=expert_weight_layout,
        dense_weight_layout=dense_weight_layout,
        auxiliary_layers=auxiliary_layers,
        verify=phase == "verify",
        **checkpoint.definition_options,
    )
    if activation_quantization is not None:
        if dense_weight_layout != "raw":
            raise ValueError("q16 activation storage currently requires raw dense weights")
        if activation_quantization not in ("q16", "mixed", "half") or provider != "webgpu":
            raise ValueError("q16/mixed/half activation arithmetic requires WebGPU")
        from akas.compilation.quantization import q16_activations

        graph = q16_activations(graph, mixed=activation_quantization == "mixed",
                                half_only=activation_quantization == "half")
    for name, spec in tuple(graph.buffers.items()):
        if spec.role == "state":
            if spec.shape[0] != rows:
                raise ValueError("LFM state must have one leading request dimension")
            value = replace(spec, shape=(max_requests, *spec.shape[1:]))
            object.__setattr__(value, "_graph", graph)
            graph.buffers[name] = value
    graph.buffer("mapping", (rows,), "int32", role="input")
    graph.bind_requests(
        inputs={
            "tokens": "prefill_tokens" if chunk > 1 else "tokens",
            "mapping": "mapping",
            "lengths": "active",
        },
        prediction="sampled",
        features={
            "logits": "logits",
            **{f"layer.{i}": f"blk.{i}.output" for i in auxiliary_layers},
        },
        prefill_features=tuple(f"layer.{i}" for i in auxiliary_layers),
        reset=[
            name
            for name, spec in graph.buffers.items()
            if spec.role == "state"
            and (name == "positions" or name.endswith("conv_state"))
        ],
    )

    if phase != "verify":
        return graph
    from akas.graph import Graph
    from akas.ops.convolution.definition import commit_histories

    commit = Graph.from_metadata(graph.metadata())
    commit.calls, commit.request = [], None
    commit_histories(graph, commit)
    commit(
        "advance",
        dict(r=rows, chunk=chunk),
        commit.buffers["positions"],
        commit.buffers["active"],
    )
    return graph, commit
