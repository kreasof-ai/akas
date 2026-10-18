"""Contracts and implementation choices for attention."""

from akas.kernels.contracts import OutputABI
from akas.ops.attention.validation import validation
from akas.kernels.contracts import Implementation
from akas.quantization.kv import is_quantized
from akas.ops.attention.validation import pipeline_validation as pipeline_validation


def cache_storage(p):
    from akas.memory.state import BufferSpec
    from akas.quantization.kv import get
    from akas.quantization.kv.codec import Storage

    shape = (p["r"] // p.get("chunk", 1), p["kh"], p["capacity"], p["d"])
    if p["kv_dtype"] not in ("float16", "float32"):
        return get(p["kv_dtype"]).storage(shape, provider=p["provider"])
    dtype = "float32" if p["provider"] == "webgpu" else p["kv_dtype"]
    return Storage(
        BufferSpec(shape, dtype),
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
    }
    if set(settings) - allowed:
        raise ValueError("unknown attention pipeline policy")
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
    query = graph.buffer(
        prefix + "query", (rows, config.heads, config.head_dim), dtype
    )
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
        "attention_merge": (1,),
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
            if call.parameters.get("kv_dtype") == "fp8"
            or len(call.arguments) == 11
            else (6, 7, 8),
            ("kc", "vc", "ks", "vs", "q")
            if call.parameters.get("kv_dtype") == "fp8"
            or len(call.arguments) == 11
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
        "attention_merge",
        "attention_qkv",
        "attention_q",
        "attention_kv",
    )
}
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
