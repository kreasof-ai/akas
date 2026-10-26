"""Contracts and implementation choices for attention."""

from akas.kernels.contracts import OutputABI
from akas.ops.attention.validation import validation
from akas.ops.attention.validation import extent_validation
from akas.ops.attention.validation import pad_validation
from akas.kernels.contracts import Implementation
from akas.quantization.kv import is_quantized
from akas.ops.attention.validation import pipeline_validation as pipeline_validation


def cache_storage(p):
    from akas.memory.state import BufferSpec
    from akas.quantization.kv import get
    from akas.quantization.kv.codec import Storage

    padding = p.get("cache_padding", 0)
    if type(padding) is not int or not 0 <= padding <= 4096 or padding % 32:
        raise ValueError("cache padding must be a bounded multiple of 32 tokens")
    if padding and p["kv_dtype"] not in ("float16", "float32"):
        raise ValueError("physical cache padding requires native floating storage")
    shape = (p["r"] // p.get("chunk", 1), p["kh"], p["capacity"] + padding, p["d"])
    if p["kv_dtype"] not in ("float16", "float32"):
        return get(p["kv_dtype"]).storage(shape, provider=p["provider"])
    return Storage(
        BufferSpec(shape, p["kv_dtype"]),
        (BufferSpec((*shape[:-1], 1), "float32"),),
        ("scales",),
    )


def fp8_decode_attention(ctx, block):
    from akas.quantization.kv import is_quantized

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
    from akas.quantization.kv import is_quantized

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


def search_policies(graph, commit, *, provider, target):
    if graph.contract == "reference":
        if any(
            call.kind in ("attention", "attention_partial")
            and call.parameters.get("optimized")
            for call in graph.calls
        ):
            choices = [{"splits": count} for count in (32, 64, 128)]
            if provider == "cuda" and any(
                _can_stage_prefill(call) for call in graph.calls
            ):
                choices += [dict(choice, decoded_kv=True) for choice in choices]
            if provider == "cuda" and target == "sm_89" and any(
                _can_pad_prefill(call) for call in graph.calls
            ):
                choices += [dict(choice, native_tail=True) for choice in choices
                            if not choice.get("decoded_kv")]
            return choices
        return []
    if (
        graph.contract != "block-fp8"
        or provider != "cuda"
        or not any(
            call.kind == "attention" and call.parameters.get("kv_dtype") == "fp8"
            for call in graph.calls
        )
    ):
        return []
    choices = [{"splits": count} for count in (2, 4, 8)]
    if target == "sm_90a":
        choices.append({"hopper": True, "decoded_kv": True, "splits": 2})
    return choices


def _can_pad_prefill(call):
    p = call.parameters
    return (
        call.kind == "attention" and p.get("optimized", False)
        and p.get("kv_dtype") == "float16" and p.get("unit_scales", False)
        and p.get("chunk", 1) >= 256 and p.get("d") == 64
        and p.get("capacity", 0) >= 4096 and p["capacity"] % 256 == 0
    )


def _pad_prefill(graph, call, workspaces):
    from dataclasses import replace
    from akas.graph import Call

    p = call.parameters
    slots, heads, capacity, depth = p["r"] // p["chunk"], p["kh"], p["capacity"], p["d"]
    fields = (dict(call.bindings) if call.bindings else dict(zip(
        ("q", "kc", "vc", "ks", "vs", "out", "positions", "enabled"), call.arguments)))
    shape = (slots, heads, capacity, depth)
    if shape not in workspaces:
        workspaces[shape] = graph.buffer(f"pipeline.attention.tail.{len(workspaces)}.lengths",
                                        (slots,), "int32")
    lengths = workspaces[shape]
    extent = Call("attention_cache_extent",
                  dict(slots=slots, chunk=p["chunk"], capacity=capacity),
                  (fields["positions"], fields["enabled"], lengths.name))
    pad = Call("attention_cache_pad", dict(slots=slots, heads=heads, capacity=capacity, d=depth),
               (fields["kc"], fields["vc"], lengths.name))
    return extent, pad, replace(call, parameters={**p, "staged_kv": True})


def _can_stage_prefill(call):
    """Limit full-cache materialization to compatible, amortized MMA chunks."""
    p = call.parameters
    eligible = (
        call.kind == "attention"
        and p.get("optimized", False)
        and p.get("kv_dtype") not in (None, "float16", "float32", "bfloat16")
        and p.get("chunk", 1) >= 256
        and p.get("d") == 64
        and p.get("capacity", 0) >= 4096
        and p["capacity"] % 512 == 0
    )
    if not eligible:
        return False
    from akas.quantization.kv import get

    try:
        return get(p["kv_dtype"]).supports(
            "stage",
            (p["r"] // p["chunk"], p["kh"], p["capacity"], p["d"]),
            provider="cuda",
        )
    except ValueError:
        return False


def _stage_prefill(graph, call, workspaces):
    """Expose conversion to qualification/search and share temporary allocations."""
    from akas.graph import Call

    p = call.parameters
    slots, heads, capacity, depth = (
        p["r"] // p["chunk"],
        p["kh"],
        p["capacity"],
        p["d"],
    )
    names = (
        dict(call.bindings)
        if call.bindings
        else dict(
            zip(
                ("q", "kc", "vc", "ks", "vs", "out", "positions", "enabled"),
                call.arguments,
            )
        )
    )
    if set(names) != {"q", "kc", "vc", "ks", "vs", "out", "positions", "enabled"}:
        raise ValueError("cache staging requires the floating attention ABI")
    shape = (slots, heads, capacity, depth)
    if shape not in workspaces:
        prefix = f"pipeline.attention.stage.{len(workspaces)}."
        workspaces[shape] = (
            graph.buffer(prefix + "lengths", (slots,), "int32"),
            graph.buffer(prefix + "keys", shape, "float16"),
            graph.buffer(prefix + "values", shape, "float16"),
            graph.buffer(prefix + "scales", (*shape[:-1], 1), "float32"),
        )
    lengths, keys, values, scales = workspaces[shape]
    extent = Call(
        "attention_cache_extent",
        dict(slots=slots, chunk=p["chunk"], capacity=capacity),
        (names["positions"], names["enabled"], lengths.name),
    )
    stage = Call(
        "kv_stage",
        dict(
            slots=slots,
            kv_heads=heads,
            capacity=capacity,
            d=depth,
            kv_dtype=p["kv_dtype"],
        ),
        tuple(names[name] for name in ("kc", "vc", "ks", "vs"))
        + (lengths.name, keys.name, values.name, scales.name),
    )
    attention = Call(
        "attention",
        {**p, "kv_dtype": "float16", "unit_scales": True, "staged_kv": True},
        (
            names["q"],
            keys.name,
            values.name,
            scales.name,
            scales.name,
            names["out"],
            names["positions"],
            names["enabled"],
        ),
        label=call.label,
    )
    return extent, stage, attention


def lower(graph, commit, settings, *, provider, target):
    from akas.graph import Call

    allowed = {
        "splits",
        "decoded_kv",
        "hopper",
        "key_rows",
        "query_tokens",
        "pad_queries",
        "query_rows",
        "threads",
        "value_splits",
        "packed_loads",
        "native_tail",
        "adaptive_splits",
    }
    if set(settings) - allowed:
        raise ValueError("unknown attention pipeline policy")
    if "adaptive_splits" in settings and graph.contract != "reference":
        raise ValueError("adaptive splits require the floating attention contract")
    if graph.contract == "reference":
        from dataclasses import replace

        if not settings or set(settings) - {"splits", "decoded_kv", "native_tail", "adaptive_splits"}:
            raise ValueError("unknown floating attention policy")
        if "splits" in settings and (
            type(settings["splits"]) is not int
            or settings["splits"] not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
        ):
            raise ValueError(
                "floating attention policy requires a supported split count"
            )
        if "decoded_kv" in settings and type(settings["decoded_kv"]) is not bool:
            raise ValueError("decoded KV policy must be a boolean")
        if settings.get("decoded_kv") and provider != "cuda":
            raise ValueError("staged floating attention is currently qualified on CUDA")
        if "native_tail" in settings and type(settings["native_tail"]) is not bool:
            raise ValueError("native tail preparation must be a boolean")
        if settings.get("native_tail") and (provider != "cuda" or target != "sm_89"):
            raise ValueError("native tail attention currently requires CUDA sm_89")
        if "adaptive_splits" in settings and type(settings["adaptive_splits"]) is not bool:
            raise ValueError("adaptive attention splits must be a boolean")
        if settings.get("adaptive_splits") and provider != "webgpu":
            raise ValueError("adaptive floating attention currently requires WebGPU")
        calls, workspaces, tails = [], {}, {}
        for call in graph.calls:
            if (
                "splits" in settings
                and call.kind
                in (
                    "attention_partial",
                    "attention_adaptive_partial",
                    "attention_merge",
                )
                and call.parameters.get("optimized")
            ):
                p = {**call.parameters, "splits": settings["splits"]}
                if call.kind in ("attention_partial", "attention_adaptive_partial"):
                    name = (
                        call.bindings.get("out") if call.bindings else call.arguments[5]
                    )
                    buffer = graph.buffers[name]
                    if buffer.alias_of is not None:
                        raise ValueError(
                            "attention partition scratch cannot be an alias"
                        )
                    buffer = replace(
                        buffer, shape=(p["r"] * p["h"] * p["splits"] * 66,)
                    )
                    object.__setattr__(buffer, "_graph", graph)
                    graph.buffers[name] = buffer
                call = replace(call, parameters=p)
            if settings.get("adaptive_splits") and call.kind == "attention_partial":
                if not (call.parameters.get("unit_scales") and call.parameters["kv_dtype"] == "float16"):
                    raise ValueError("adaptive floating attention requires unit-scale F16 KV")
                call = replace(call, kind="attention_adaptive_partial")
            if settings.get("decoded_kv") and _can_stage_prefill(call):
                calls.extend(_stage_prefill(graph, call, workspaces))
                continue
            if settings.get("native_tail") and _can_pad_prefill(call):
                calls.extend(_pad_prefill(graph, call, tails))
                continue
            calls.append(call)
        graph.calls = calls
        return graph, commit
    if graph.contract != "block-fp8":
        return (graph, commit)
    if provider != "cuda":
        raise ValueError("native attention pipelines require CUDA")
    if settings.get("hopper") and target != "sm_90a":
        raise ValueError("Hopper attention requires sm_90a")
    if settings.get("decoded_kv") and (not settings.get("hopper")):
        raise ValueError("decoded KV requires a compatible Hopper attention algorithm")
    splits = settings.get("splits", 1)
    if type(splits) is not int or splits <= 0:
        raise ValueError("attention splits must be positive")
    result = []
    for index, call in enumerate(graph.calls):
        if call.kind != "attention" or "slots" not in call.parameters:
            result.append(call)
            continue
        if call.parameters.get("kv_dtype") != "fp8":
            raise ValueError("this attention pipeline requires FP8 KV storage")
        names = (
            dict(call.bindings)
            if call.bindings
            else dict(
                zip(
                    (
                        "q",
                        "kc",
                        "vc",
                        "ks",
                        "vs",
                        "projection",
                        "positions",
                        "lengths",
                        "out",
                    ),
                    call.arguments,
                )
            )
        )
        required = {
            "q",
            "kc",
            "vc",
            "ks",
            "vs",
            "projection",
            "positions",
            "lengths",
            "out",
        }
        if names.keys() != required:
            raise ValueError("attention call does not match the native chunk ABI")
        slots, chunk, capacity = (
            call.parameters[name] for name in ("slots", "chunk", "capacity")
        )
        query = graph.buffers[names["q"]]
        cache = graph.buffers[names["kc"]]
        if query.shape != (slots * chunk, 16, 256) or cache.shape[1:] != (
            2,
            capacity,
            256,
        ):
            raise ValueError(
                "selected native attention requires 16/2 heads and width 256"
            )
        p = {**call.parameters, **settings, "splits": splits}
        prefix = f"pipeline.attention.{index}."
        if settings.get("decoded_kv"):
            if capacity % 128:
                raise ValueError("decoded KV capacity must be divisible by 128")
            decoded = [
                graph.buffer(prefix + field, (slots, 2, capacity, 256), "bfloat16")
                for field in ("decoded_k", "decoded_v")
            ]
            result.append(
                Call(
                    "attention_decode_workspace",
                    p,
                    tuple(
                        (
                            names[name]
                            for name in ("kc", "vc", "ks", "vs", "positions", "lengths")
                        )
                    )
                    + tuple((value.name for value in decoded)),
                )
            )
            names.update(kc=decoded[0].name, vc=decoded[1].name)
        if splits > 1:
            partial = graph.buffer(
                prefix + "parts", (slots, 2, splits, chunk * 8, 256), "float32"
            )
            stats = graph.buffer(
                prefix + "stats", (slots, 2, splits, chunk * 8, 2), "float32"
            )
            args = tuple((names[name] for name in ("q", "kc", "vc")))
            if not settings.get("decoded_kv"):
                args += tuple((names[name] for name in ("ks", "vs")))
            args += tuple((names[name] for name in ("positions", "lengths"))) + (
                partial.name,
                stats.name,
            )
            result.append(Call("attention_chunk_partial", p, args))
            active = graph.buffer(prefix + "active", (slots * chunk,), "int32")
            result.append(
                Call("attention_chunk_active", p, (names["lengths"], active.name))
            )
            result.append(
                Call(
                    "attention_chunk_merge",
                    p,
                    (
                        partial.name,
                        stats.name,
                        names["projection"],
                        active.name,
                        names["out"],
                    ),
                    label=call.label,
                )
            )
        elif settings.get("hopper"):
            args = tuple((names[name] for name in ("q", "kc", "vc")))
            if not settings.get("decoded_kv"):
                args += tuple((names[name] for name in ("ks", "vs")))
            args += tuple(
                (names[name] for name in ("projection", "positions", "lengths", "out"))
            )
            result.append(Call("attention_chunk_hopper", p, args, label=call.label))
        else:
            result.append(call)
    graph.calls = result
    return (graph, commit)


def projected_workspace(graph, rows, config, *, prefix="", splits=None, gated=True):
    """Scratch belongs to attention, independent of checkpoint weight storage."""
    heads, kv_heads, depth = (config.heads, config.kv_heads, config.head_dim)
    specs = {
        "attq": (rows, (2 if gated else 1) * heads * depth),
        "attk": (rows, kv_heads * depth),
        "attv": (rows, kv_heads * depth),
        "attq_normal": (rows, heads, depth),
        "att_out": (rows, heads * depth),
    }
    if splits is not None:
        specs.update(
            att_parts=(rows, kv_heads, splits, heads // kv_heads, depth),
            att_stats=(rows, kv_heads, splits, heads // kv_heads, 2),
        )
    return {
        name: graph.buffer(
            prefix + name,
            shape,
            "bfloat16" if name in ("attq_normal", "att_out") else "float32",
        )
        for name, shape in specs.items()
    }


def attend(
    graph, x, weights, states, controls, *, config, prefix, context, dtype="bfloat16"
):
    """Compose packed Q/K/V projection and mapped causal attention."""
    from akas.ops.linear.definition import project

    rows, width = x.shape
    common = controls["parameters"]
    projected = project(
        graph,
        x,
        weights[prefix + "input.weight"],
        width=width,
        name=prefix + "projection",
    )
    query = graph.buffer(prefix + "query", (rows, config.heads, config.head_dim), dtype)
    attended = graph.buffer(
        prefix + "attended", (rows, config.heads * config.head_dim), dtype
    )
    parameters = dict(
        common,
        heads=config.heads,
        kv_heads=config.kv_heads,
        d=config.head_dim,
        capacity=context,
        eps=config.epsilon,
        **({"kv_dtype": "fp8"} if len(states) == 4 else {}),
    )
    graph(
        "attention_qkv",
        dict(parameters, theta=config.theta),
        projected,
        weights[prefix + "q_norm.weight"],
        weights[prefix + "k_norm.weight"],
        controls["mapping"],
        controls["flat_positions"],
        controls["active"],
        *states,
        query,
    )
    graph(
        "attention",
        parameters,
        query,
        *states,
        projected,
        controls["mapping"],
        controls["positions"],
        controls["lengths"],
        attended,
    )
    return project(
        graph,
        attended,
        weights[prefix + "o_proj.weight"],
        width=width,
        name=prefix + "output",
    )


OUTPUT_ARGUMENTS = {
    "reference": {
        "rope": (1,),
        "cache_write": (2, 3, 4, 5),
        "attention": (5,),
        "attention_partial": (5,),
        "attention_adaptive_partial": (5,),
        "attention_merge": (1,),
        "attention_cache_extent": OutputABI((-1,), ("lengths",)),
        "attention_cache_pad": OutputABI((0, 1), ("kc", "vc")),
    }
}
OUTPUT_ARGUMENTS["block-fp8"] = {
    "attention_decode_workspace": (-2, -1),
    "attention_chunk_active": (-1,),
    "attention_chunk_partial": (-2, -1),
    "attention_chunk_merge": (-1,),
    "attention_chunk_hopper": (-1,),
}
for _contract in ("bf16", "block-fp8"):
    OUTPUT_ARGUMENTS.setdefault(_contract, {}).update(
        {
            "attention_chunk_active": OutputABI((-1,), ("out",)),
            "attention_chunk_merge": OutputABI((-1,), ("out",)),
            "attention_chunk_hopper": OutputABI((-1,), ("out",)),
            "attention_chunk_partial": OutputABI((-2, -1), ("out", "stats")),
            "attention_decode_workspace": OutputABI(
                (-2, -1), ("decoded_k", "decoded_v")
            ),
        }
    )
OUTPUT_ARGUMENTS["bf16"].update(
    {
        "attention_qkv": lambda call: OutputABI(
            (6, 7, 8, 9, 10)
            if call.parameters.get("kv_dtype") == "fp8" or len(call.arguments) == 11
            else (6, 7, 8),
            ("kc", "vc", "ks", "vs", "q")
            if call.parameters.get("kv_dtype") == "fp8" or len(call.arguments) == 11
            else ("kc", "vc", "q"),
        ),
        "attention": OutputABI((-1,), ("out",)),
    }
)
OUTPUT_ARGUMENTS["block-fp8"].update(
    {
        "attention_q": OutputABI((-1,), ("q",)),
        "attention_partial": OutputABI((-2, -1), ("out", "stats")),
        "attention_merge": OutputABI((-1,), ("out",)),
        "attention": OutputABI((-1,), ("out",)),
    }
)
VALIDATIONS = {
    name: validation
    for name in (
        "rope",
        "cache_write",
        "attention",
        "attention_partial",
        "attention_adaptive_partial",
        "attention_merge",
        "attention_qkv",
        "attention_q",
        "attention_kv",
    )
}
VALIDATIONS["attention_cache_extent"] = extent_validation
VALIDATIONS["attention_cache_pad"] = pad_validation
VALIDATIONS.update(
    {
        name: pipeline_validation
        for name in (
            "attention_decode_workspace",
            "attention_chunk_active",
            "attention_chunk_partial",
            "attention_chunk_merge",
            "attention_chunk_hopper",
        )
    }
)
IMPLEMENTATIONS = (
    Implementation(
        "attention",
        "akas.ops.attention.cuda",
        "block-fp8",
        phase="prefill",
        priority=30,
        accepts=lambda p: is_quantized(p.get("kv_dtype", "bfloat16")),
        factory="quantized",
        name="attention.cuda.quantized",
    ),
    Implementation(
        "attention_kv",
        "akas.ops.attention.cuda",
        "block-fp8",
        phase="prefill",
        priority=30,
        accepts=lambda p: is_quantized(p.get("kv_dtype", "bfloat16")),
        factory="kv_quantized",
        name="attention.cuda.quantized",
    ),
    Implementation(
        "attention_partial",
        "akas.ops.attention.cuda",
        "block-fp8",
        phase="decode",
        priority=30,
        accepts=lambda p: is_quantized(p.get("kv_dtype", "bfloat16")),
        factory="partial_quantized",
        name="attention.cuda.quantized",
    ),
    Implementation(
        "attention_kv",
        "akas.ops.attention.cuda",
        "block-fp8",
        phase="decode",
        priority=30,
        accepts=lambda p: is_quantized(p.get("kv_dtype", "bfloat16")),
        factory="kv_quantized",
        name="attention.cuda.quantized",
    ),
    Implementation(
        "attention_partial",
        "akas.ops.attention.cuda",
        "packed",
        priority=20,
        factory="partitioned_flat",
        name="attention.cuda.packed",
    ),
    Implementation(
        "attention_grouped",
        "akas.ops.attention.cuda",
        "packed",
        priority=20,
        factory="grouped",
        name="attention.cuda.packed",
    ),
    Implementation(
        "attention_merge",
        "akas.ops.attention.cuda",
        "packed",
        priority=20,
        factory="merge_partitioned_flat",
        name="attention.cuda.packed",
    ),
    Implementation(
        "qkv",
        "akas.ops.attention.kernel",
        "packed-baseline",
        factory="prepare_flat",
        name="attention.cuda.packed_baseline",
    ),
    Implementation(
        "attention",
        "akas.ops.attention.kernel",
        "packed-baseline",
        factory="attention_flat",
        name="attention.cuda.packed_baseline",
    ),
    Implementation(
        "qkv",
        "akas.ops.attention.kernel",
        "packed",
        priority=0,
        factory="prepare_flat",
        name="attention.cuda.packed_baseline",
    ),
    Implementation(
        "attention",
        "akas.ops.attention.kernel",
        "packed",
        priority=0,
        factory="attention_flat",
        name="attention.cuda.packed_baseline",
    ),
    Implementation(
        "qkv",
        "akas.ops.attention.kernel",
        "packed",
        provider="webgpu",
        priority=0,
        factory="prepare_flat",
        name="attention.cuda.packed_baseline",
    ),
    Implementation(
        "attention",
        "akas.ops.attention.kernel",
        "packed",
        provider="webgpu",
        priority=0,
        factory="attention_flat",
        name="attention.cuda.packed_baseline",
    ),
    Implementation(
        "attention_scores",
        "akas.ops.attention.webgpu",
        "packed",
        provider="webgpu",
        priority=20,
        factory="attention_scores",
        name="attention.webgpu.packed",
    ),
    Implementation(
        "attention",
        "akas.ops.attention.webgpu",
        "packed",
        provider="webgpu",
        priority=20,
        factory="attention",
        name="attention.webgpu.packed",
    ),
    Implementation(
        "qkv",
        "akas.ops.attention.webgpu",
        "packed",
        provider="webgpu",
        priority=20,
        factory="qkv",
        name="attention.webgpu.packed",
    ),
    Implementation(
        "attention_qkv",
        "akas.ops.attention.cuda",
        "bf16",
        accepts=lambda p: not is_quantized(p.get("kv_dtype", "bfloat16")),
        factory="qkv_mapped_bf16",
        name="attention.cuda.bf16_batch",
    ),
    Implementation(
        "attention_qkv",
        "akas.ops.attention.cuda",
        "bf16",
        priority=20,
        accepts=lambda p: is_quantized(p.get("kv_dtype", "bfloat16")),
        factory="qkv_mapped_fp8",
        name="attention.cuda.fp8",
    ),
    Implementation(
        "attention",
        "akas.ops.attention.cuda",
        "bf16",
        factory="mapped_bf16",
        name="attention.cuda.bf16_batch",
    ),
    Implementation(
        "attention_qkv",
        "akas.ops.attention.webgpu",
        "bf16",
        provider="webgpu",
        factory="qkv_mapped_webgpu",
        name="attention.webgpu.bf16_batch",
    ),
    Implementation(
        "attention",
        "akas.ops.attention.webgpu",
        "bf16",
        provider="webgpu",
        factory="mapped_webgpu",
        name="attention.webgpu.bf16_batch",
    ),
    Implementation(
        "attention_kv",
        "akas.ops.attention.cuda",
        "block-fp8",
        phase="decode",
        factory="kv",
        name="attention.cuda.decode",
    ),
    Implementation(
        "attention_partial",
        "akas.ops.attention.cuda",
        "block-fp8",
        phase="decode",
        factory="partial_bf16",
        name="attention.cuda.decode",
    ),
    Implementation(
        "attention_q",
        "akas.ops.attention.cuda",
        "block-fp8",
        factory="query",
        name="attention.cuda.decode",
    ),
    Implementation(
        "attention_merge",
        "akas.ops.attention.cuda",
        "block-fp8",
        factory="merge_native",
        name="attention.cuda.decode",
    ),
    Implementation(
        "attention_kv",
        "akas.ops.attention.cuda",
        "block-fp8",
        phase="prefill",
        required=("slots", "chunk"),
        factory="kv_chunk",
        name="attention.cuda.prefill",
    ),
    Implementation(
        "attention",
        "akas.ops.attention.cuda",
        "block-fp8",
        phase="prefill",
        required=("slots", "chunk"),
        factory="native",
        name="attention.cuda.prefill",
    ),
    Implementation(
        "rope",
        "akas.ops.attention.kernel",
        "reference",
        factory="rope",
        name="attention.reference",
    ),
    Implementation(
        "cache_write",
        "akas.ops.attention.kernel",
        "reference",
        factory="cache_write",
        name="attention.reference",
    ),
    Implementation(
        "attention",
        "akas.ops.attention.kernel",
        "reference",
        factory="attention",
        name="attention.reference",
    ),
    Implementation(
        "rope",
        "akas.ops.attention.kernel",
        "reference",
        provider="webgpu",
        factory="rope",
        name="attention.reference",
    ),
    Implementation(
        "cache_write",
        "akas.ops.attention.kernel",
        "reference",
        provider="webgpu",
        factory="cache_write",
        name="attention.reference",
    ),
    Implementation(
        "attention",
        "akas.ops.attention.kernel",
        "reference",
        provider="webgpu",
        factory="attention",
        name="attention.reference",
    ),
    Implementation(
        "attention",
        "akas.ops.attention.cuda",
        "reference",
        priority=30,
        defaults=(("prefill_precision", "float16"),),
        accepts=lambda p: p.get("optimized", False),
        factory="floating",
        name="attention.cuda.floating",
    ),
    Implementation(
        "attention_partial",
        "akas.ops.attention.kernel",
        "reference",
        priority=30,
        defaults=(("prefill_precision", "float16"),),
        accepts=lambda p: p.get("optimized", False),
        factory="partial_floating",
        name="attention.cuda.floating",
    ),
    Implementation(
        "attention_merge",
        "akas.ops.attention.kernel",
        "reference",
        priority=30,
        defaults=(("prefill_precision", "float16"),),
        accepts=lambda p: p.get("optimized", False),
        factory="merge_floating",
        name="attention.cuda.floating",
    ),
    Implementation(
        "attention_partial",
        "akas.ops.attention.kernel",
        "reference",
        provider="webgpu",
        priority=30,
        accepts=lambda p: p.get("optimized", False),
        factory="partial_floating",
        name="attention.floating",
    ),
    Implementation(
        "attention_merge",
        "akas.ops.attention.kernel",
        "reference",
        provider="webgpu",
        priority=30,
        accepts=lambda p: p.get("optimized", False),
        factory="merge_floating",
        name="attention.floating",
    ),
    Implementation(
        "attention_decode_workspace",
        "akas.ops.attention.cuda",
        "block-fp8",
        factory="decode",
        name="attention.cuda.workspace",
    ),
    Implementation(
        "attention_chunk_active",
        "akas.ops.attention.kernel",
        "block-fp8",
        factory="chunk_active",
        name="attention.pipeline",
    ),
    Implementation(
        "attention_chunk_partial",
        "akas.ops.attention.cuda",
        "block-fp8",
        space=(("key_rows", (64, 32, 128)), ("packed_loads", (False, True))),
        semantic_keys=("slots", "chunk", "capacity", "splits"),
        accepts=lambda p: not p.get("hopper"),
        factory="partial",
        name="attention.cuda.split",
    ),
    Implementation(
        "attention_chunk_partial",
        "akas.ops.attention.cuda",
        "block-fp8",
        targets=("sm_90a",),
        space=(("query_tokens", (8, 4, 16)), ("key_rows", (64, 32, 128))),
        semantic_keys=("slots", "chunk", "capacity", "splits"),
        accepts=lambda p: p.get("hopper", False),
        factory="hopper_prefill_split_partial",
        name="attention.hardware.hopper.attention",
    ),
    Implementation(
        "attention_chunk_merge",
        "akas.ops.attention.cuda",
        "block-fp8",
        factory="chunk",
        name="attention.cuda.split",
    ),
    Implementation(
        "attention_chunk_hopper",
        "akas.ops.attention.cuda",
        "block-fp8",
        targets=("sm_90a",),
        space=(("query_rows", (128, 64, 256)), ("threads", (256, 128, 512))),
        semantic_keys=("slots", "chunk", "capacity"),
        factory="attention",
        name="attention.hardware.hopper.prefill",
    ),
)

IMPLEMENTATIONS += tuple(
    Implementation(
        "attention_cache_pad",
        f"akas.ops.attention.{provider}",
        "reference",
        provider=provider,
        factory="cache_pad",
        name="attention.cache_pad",
    )
    for provider in ("cuda", "webgpu")
)

IMPLEMENTATIONS += tuple(
    Implementation(
        "attention_cache_extent",
        f"akas.ops.attention.{provider}",
        "reference",
        provider=provider,
        factory="cache_extent",
        name="attention.cache_extent",
    )
    for provider in ("cuda", "webgpu")
)

IMPLEMENTATIONS += (
    Implementation(
        "attention",
        "akas.ops.attention.cuda",
        "reference",
        priority=35,
        defaults=(
            ("prefill_precision", "float16"),
            ("async_loads", True),
            ("query_rows", 32),
            ("key_rows", 64),
            ("stages", 2),
            ("threads", 128),
        ),
        space=(
            ("async_loads", (True, False)),
            ("query_rows", (32, 64, 16)),
            ("key_rows", (64, 128, 32)),
            ("stages", (2, 1, 3)),
            ("threads", (128, 256)),
        ),
        accepts=lambda p: (
            p.get("optimized", False)
            and p.get("chunk", 1) > 1
            and p.get("unit_scales", False)
            and p["kv_dtype"] == "float16"
        ),
        factory="floating",
        name="attention.cuda.pipelined",
        reference_keys=("key_rows",),
    ),
)

IMPLEMENTATIONS += (
    Implementation(
        "attention",
        "akas.ops.attention.webgpu",
        "reference",
        provider="webgpu",
        priority=40,
        factory="tiled_prefill",
        name="attention.webgpu.tiled_prefill",
        accepts=lambda p: (
            p.get("optimized", False)
            and p.get("chunk", 1) >= 4
            and p["chunk"] % 4 == 0
            and p["d"] == 64
            and p["h"] % p["kh"] == 0
            and p["h"] // p["kh"] in (1, 2, 4, 8)
        ),
        space=(
            ("block_q", (4, 8, 16)),
            ("block_k", (16, 32)),
            ("key_pad", (0, 1)),
            ("shared_dtype", ("float16", "float32")),
            ("dot_width", (4, 1)),
            ("packed_shared", (False, True)),
        ),
    ),
)

IMPLEMENTATIONS += tuple(
    Implementation(
        "attention_partial",
        "akas.ops.attention.kernel",
        "reference",
        provider=provider,
        priority=35 if provider == "webgpu" else 25,
        accepts=lambda p: (
            p.get("optimized", False)
            and p["d"] == 64
            and p["h"] % p["kh"] == 0
            and p["h"] // p["kh"] in (1, 2, 4, 8)
        ),
        factory="tiled_partial_floating",
        name="attention.grouped_decode",
        space=(("shared_dtype", ("float32", "float16")), ("dot_width", (1, 4)))
        if provider == "webgpu" else (),
    )
    for provider in ("cuda", "webgpu")
)


def _mma_cache_supported(p):
    if p.get("kv_dtype") == "float16":
        return p.get("unit_scales", False)
    from akas.quantization.formats import get

    try:
        return get("kv", p.get("kv_dtype")).supports(
            "read_packed", (p["r"], p["kh"], p["capacity"], p["d"]), provider="cuda"
        )
    except ValueError:
        return False


IMPLEMENTATIONS += (
    Implementation(
        "attention_partial", "akas.ops.attention.webgpu", "reference",
        provider="webgpu", priority=34, factory="blocked_partial",
        defaults=(("key_unroll", 4),), space=(("key_unroll", (4, 8, 2)),),
        semantic_keys=("r", "h", "kh", "d", "capacity", "splits", "kv_dtype"),
        accepts=lambda p: p.get("optimized", False) and p["d"] == 64
        and p["kv_dtype"] in ("float16", "float32"),
        name="attention.webgpu.blocked_decode",
    ),
    Implementation(
        "attention_partial", "akas.ops.attention.webgpu", "reference",
        provider="webgpu", priority=33, factory="channel_partial",
        defaults=(("key_rows", 16),), space=(("key_rows", (16, 32)),),
        semantic_keys=("r", "h", "kh", "d", "capacity", "splits", "kv_dtype"),
        accepts=lambda p: p.get("optimized", False) and p["d"] == 64
        and p["h"] % p["kh"] == 0 and p["h"] // p["kh"] in (1, 2, 4, 8)
        and p["kv_dtype"] == "float16" and p.get("unit_scales", False),
        name="attention.webgpu.channel_decode",
    ),
    Implementation(
        "attention_adaptive_partial", "akas.ops.attention.webgpu", "reference",
        provider="webgpu", priority=40, factory="channel_partial",
        defaults=(("key_rows", 32), ("granule", 512)),
        space=(("key_rows", (32, 16)), ("granule", (512, 128, 64))),
        reference_keys=("granule",),
        semantic_keys=("r", "h", "kh", "d", "capacity", "splits", "kv_dtype"),
        accepts=lambda p: p.get("optimized", False) and p["d"] == 64
        and p["h"] % p["kh"] == 0 and p["h"] // p["kh"] in (1, 2, 4, 8)
        and p["kv_dtype"] == "float16" and p.get("unit_scales", False),
        name="attention.webgpu.adaptive_decode",
    ),
    Implementation(
        "attention_adaptive_partial", "akas.ops.attention.webgpu", "reference",
        provider="webgpu", priority=39, factory="packed_channel_partial",
        defaults=(("key_rows", 32), ("granule", 128), ("dot_chains", 1)),
        space=(("key_rows", (32, 64, 128, 16)), ("granule", (128, 512, 64)),
               ("dot_chains", (1, 4))),
        reference_keys=("granule",),
        semantic_keys=("r", "h", "kh", "d", "capacity", "splits", "kv_dtype"),
        accepts=lambda p: p.get("optimized", False) and p["d"] == 64
        and p["h"] % p["kh"] == 0 and p["h"] // p["kh"] in (1, 2, 4, 8)
        and p["kv_dtype"] == "float16" and p.get("unit_scales", False),
        name="attention.webgpu.packed_decode",
    ),
    Implementation(
        "attention_adaptive_partial", "akas.ops.attention.webgpu", "reference",
        provider="webgpu", priority=39, factory="two_pass_partial",
        defaults=(("granule", 128), ("key_lanes", 8), ("value_unroll", 8), ("value_chains", 4)),
        space=(("key_lanes", (8, 16, 4)), ("value_unroll", (8, 16, 4)),
               ("value_chains", (4, 1))), reference_keys=("granule",),
        semantic_keys=("r", "h", "kh", "d", "capacity", "splits", "kv_dtype"),
        accepts=lambda p: p.get("optimized", False) and p["d"] == 64
        and p["h"] % p["kh"] == 0 and p["h"] // p["kh"] in (1, 2, 4, 8)
        and p["kv_dtype"] == "float16" and p.get("unit_scales", False)
        and (p["capacity"] <= 16384 or p["capacity"] % 128 == 0)
        and ((p["capacity"] + p.get("splits", 16) * 128 - 1)
             // (p.get("splits", 16) * 128) * 128 * (p["h"] // p["kh"])
             + (p["h"] // p["kh"])
             + 2 * (p["h"] // p["kh"])) * 4 <= 32768,
        name="attention.webgpu.two_pass_decode",
    ),
    Implementation(
        "attention_adaptive_partial", "akas.ops.attention.webgpu", "reference",
        provider="webgpu", priority=38, factory="buffered_partial",
        defaults=(("granule", 128), ("key_lanes", 8), ("value_unroll", 8),
                  ("value_chains", 4), ("cache_rows", 64), ("cache_dtype", "float16")),
        space=(("key_lanes", (8, 16, 4)), ("cache_dtype", ("float16", "float32")),
               ("cache_rows", (64, 32)), ("value_unroll", (8, 16))),
        reference_keys=("granule",),
        semantic_keys=("r", "h", "kh", "d", "capacity", "splits", "kv_dtype"),
        accepts=lambda p: p.get("optimized", False) and p["d"] == 64
        and p["h"] % p["kh"] == 0 and p["h"] // p["kh"] in (1, 2, 4)
        and p.get("splits", 16) >= 256 and p["kv_dtype"] == "float16"
        and p.get("unit_scales", False)
        and (p["capacity"] <= 16384 or p["capacity"] % 128 == 0),
        name="attention.webgpu.buffered_decode",
    ),
    Implementation(
        "attention_merge", "akas.ops.attention.webgpu", "reference",
        provider="webgpu", priority=31, factory="stream_merge",
        accepts=lambda p: p.get("optimized", False) and p["d"] == 64 and p["splits"] <= 512,
        name="attention.webgpu.stream_merge",
    ),
)


IMPLEMENTATIONS += (
    Implementation(
        "attention_partial",
        "akas.ops.attention.cuda",
        "reference",
        factory="mma_partial",
        name="attention.cuda.mma_decode",
        reference_keys=("key_rows",),
        priority=40,
        defaults=(("key_rows", 64), ("stages", 2), ("decode_precision", "float16")),
        space=(("key_rows", (64, 128)), ("stages", (2, 1, 3))),
        accepts=lambda p: (
            p.get("optimized", False)
            and p.get("d") == 64
            and _mma_cache_supported(p)
            and p["h"] % p["kh"] == 0
            and p["h"] // p["kh"] in (2, 4, 8, 16)
            and p["capacity"] >= 4096
        ),
    ),
)


IMPLEMENTATIONS += (
    Implementation(
        "attention",
        "akas.ops.attention.cuda",
        "reference",
        factory="packed_prefill",
        name="attention.cuda.packed_prefill",
        priority=40,
        defaults=(
            ("prefill_precision", "float16"),
            ("query_rows", 64),
            ("key_rows", 64),
            ("stages", 1),
            ("threads", 128),
            ("pipeline_group", 8),
            ("group_queries", True),
        ),
        space=(("query_rows", (64, 128)), ("stages", (1, 2)), ("threads", (128, 256))),
        reference_keys=("key_rows",),
        accepts=lambda p: (
            p.get("optimized", False)
            and p.get("chunk", 1) > 1
            and p.get("d") == 64
            and p.get("kv_dtype") not in ("float16", "float32")
            and p["h"] % p["kh"] == 0
            and p["h"] // p["kh"] in (1, 2, 4, 8)
            and p["capacity"] >= 4096
            and _mma_cache_supported(p)
        ),
    ),
)


# Row-local MMA removes cross-warp barriers while retaining the established
# two-half, per-lane F32 reduction order. Staging owns finite zero padding.
IMPLEMENTATIONS += (
    Implementation(
        "attention",
        "akas.ops.attention.cuda",
        "reference",
        factory="floating_ordered",
        name="attention.cuda.ordered",
        priority=40,
        targets=("sm_89",),
        defaults=(
            ("prefill_precision", "float16"),
            ("async_loads", True),
            ("query_rows", 64),
            ("key_rows", 64),
            ("stages", 1),
            ("threads", 128),
        ),
        space=(
            ("query_rows", (64,)),
            ("key_rows", (64,)),
            ("stages", (1,)),
            ("threads", (128,)),
        ),
        reference_keys=("key_rows",),
        accepts=lambda p: (
            p.get("optimized", False)
            and p.get("chunk", 1) >= 256
            and p.get("d") == 64
            and p.get("unit_scales", False)
            and p.get("staged_kv", False)
            and p["kv_dtype"] == "float16"
            and p["capacity"] % 256 == 0
        ),
    ),
)


from akas.ops.attention.validation import window_validation

for _kind, _positions, _names, _module in (
    ("rotary_partial", (1,), ("out",), "kernel"),
    ("qkv_split", (-3, -2, -1), ("q", "k", "v"), "kernel"),
    ("window_cache", (2, 3), ("kc", "vc"), "kernel"),
    ("window_attention", (-1,), ("out",), "cuda"),
    ("window_merge", (1,), ("out",), "kernel"),
):
    OUTPUT_ARGUMENTS["reference"][_kind] = OutputABI(_positions, _names)
    VALIDATIONS[_kind] = window_validation
    IMPLEMENTATIONS += (
        Implementation(
            _kind,
            "akas.ops.attention." + _module,
            "reference",
            factory=_kind,
            name="attention." + _kind,
        ),
    )


from akas.ops.attention.validation import rotary_table_validation

OUTPUT_ARGUMENTS["reference"]["rotary_table"] = OutputABI((1,), ("out",))
OUTPUT_ARGUMENTS["reference"]["window_grouped"] = OutputABI((-1,), ("out",))
VALIDATIONS["rotary_table"] = rotary_table_validation
VALIDATIONS["window_grouped"] = window_validation
IMPLEMENTATIONS += (
    Implementation(
        "rotary_table",
        "akas.ops.attention.kernel",
        "reference",
        factory="rotary_table",
        name="attention.cuda.rotary_table",
    ),
    Implementation(
        "window_grouped",
        "akas.ops.attention.cuda",
        "reference",
        factory="window_grouped",
        name="attention.cuda.window_grouped",
        space=(("key_rows", (64, 32, 128)),),
    ),
)


from akas.ops.attention.validation import rotary_compact_validation

OUTPUT_ARGUMENTS["reference"]["rotary_compact"] = OutputABI((1,), ("out",))
VALIDATIONS["rotary_compact"] = rotary_compact_validation
IMPLEMENTATIONS += (
    Implementation(
        "rotary_compact",
        "akas.ops.attention.kernel",
        "reference",
        factory="rotary_compact",
        name="attention.cuda.rotary_compact",
    ),
)

from akas.ops.attention.validation import qkv_rotary_cache_validation

OUTPUT_ARGUMENTS["reference"]["qkv_rotary_cache"] = OutputABI(
    (1, 2, 3), ("q", "kc", "vc")
)
VALIDATIONS["qkv_rotary_cache"] = qkv_rotary_cache_validation
IMPLEMENTATIONS += (
    Implementation(
        "qkv_rotary_cache",
        "akas.ops.attention.kernel",
        "reference",
        factory="qkv_rotary_cache",
        name="attention.cuda.qkv_rotary_cache",
    ),
)
