"""Operation-owned split-context and decoded-cache attention pipelines."""


def lower(graph, commit, settings, *, provider, target):
    from akas.graph import Call

    allowed = {"splits", "decoded_kv", "hopper", "key_rows", "query_tokens", "pad_queries",
               "query_rows", "threads", "value_splits", "packed_loads"}
    if set(settings) - allowed:
        raise ValueError("unknown attention pipeline policy")
    if graph.contract != "block-fp8":
        return graph, commit
    if provider != "cuda":
        raise ValueError("native attention pipelines require CUDA")
    if settings.get("hopper") and target != "sm_90a":
        raise ValueError("Hopper attention requires sm_90a")
    if settings.get("decoded_kv") and not settings.get("hopper"):
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
        names = (dict(call.bindings) if call.bindings else
                 dict(zip(("q", "kc", "vc", "ks", "vs", "projection", "positions", "lengths", "out"), call.arguments)))
        required = {"q", "kc", "vc", "ks", "vs", "projection", "positions", "lengths", "out"}
        if names.keys() != required:
            raise ValueError("attention call does not match the native chunk ABI")
        slots, chunk, capacity = (call.parameters[name] for name in ("slots", "chunk", "capacity"))
        query = graph.buffers[names["q"]]
        cache = graph.buffers[names["kc"]]
        if query.shape != (slots * chunk, 16, 256) or cache.shape[1:] != (2, capacity, 256):
            raise ValueError("selected native attention requires 16/2 heads and width 256")
        p = {**call.parameters, **settings, "splits": splits}
        prefix = f"pipeline.attention.{index}."
        if settings.get("decoded_kv"):
            if capacity % 128:
                raise ValueError("decoded KV capacity must be divisible by 128")
            decoded = [graph.buffer(prefix + field, (slots, 2, capacity, 256), "bfloat16") for field in ("decoded_k", "decoded_v")]
            result.append(Call("attention_decode_workspace", p, tuple(names[name] for name in ("kc", "vc", "ks", "vs", "positions", "lengths")) + tuple(value.name for value in decoded)))
            names.update(kc=decoded[0].name, vc=decoded[1].name)
        if splits > 1:
            partial = graph.buffer(prefix + "parts", (slots, 2, splits, chunk * 8, 256), "float32")
            stats = graph.buffer(prefix + "stats", (slots, 2, splits, chunk * 8, 2), "float32")
            args = tuple(names[name] for name in ("q", "kc", "vc"))
            if not settings.get("decoded_kv"):
                args += tuple(names[name] for name in ("ks", "vs"))
            args += tuple(names[name] for name in ("positions", "lengths")) + (partial.name, stats.name)
            result.append(Call("attention_chunk_partial", p, args))
            active = graph.buffer(prefix + "active", (slots * chunk,), "int32")
            result.append(Call("attention_chunk_active", p, (names["lengths"], active.name)))
            result.append(Call("attention_chunk_merge", p,
                               (partial.name, stats.name, names["projection"], active.name, names["out"]), label=call.label))
        elif settings.get("hopper"):
            args = tuple(names[name] for name in ("q", "kc", "vc"))
            if not settings.get("decoded_kv"):
                args += tuple(names[name] for name in ("ks", "vs"))
            args += tuple(names[name] for name in ("projection", "positions", "lengths", "out"))
            result.append(Call("attention_chunk_hopper", p, args, label=call.label))
        else:
            result.append(call)
    graph.calls = result
    return graph, commit
