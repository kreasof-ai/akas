"""DSpark sidecar phases follow the same graph composition contract as targets."""

from dataclasses import replace
from akas.graph import Graph
from akas.models.definition import Composition, Definition
from akas.models.dspark.weights import DSparkDraftCheckpoint


def compose(
    checkpoint,
    *,
    rows,
    chunk,
    max_requests,
    max_context,
    provider="cuda",
    phase=None,
    kv_dtype="float16",
    optimized=True,
):
    """Inject target context features or propose one Markov-corrected draft block."""
    from akas.ops.attention.definition import cache_storage, floating_attention

    config = checkpoint.config
    phase = "proposal" if phase is None else phase
    if phase not in ("context", "proposal") or provider not in ("cuda", "webgpu"):
        raise ValueError(
            "DSpark requires a context or proposal phase and a GPU provider"
        )
    if kv_dtype != "float16":
        raise ValueError("DSpark currently requires native F16 KV")
    if (
        type(rows) is not int
        or type(chunk) is not int
        or not 1 <= rows <= max_requests
        or not 1 <= chunk <= max_context <= config.max_context
    ):
        raise ValueError(
            "DSpark geometry exceeds the compiled request or context capacity"
        )
    if phase == "proposal" and chunk > config.block_size:
        raise ValueError("DSpark proposal exceeds the trained block size")
    graph = Graph(contract="reference", phase=phase)
    weights = checkpoint.graph_weights(graph, provider=provider)
    count, width = rows * chunk, config.width
    tokens = graph.buffer("tokens", (rows,), "int32", role="input")
    source = (
        tokens
        if chunk == 1
        else graph.buffer("prefill_tokens", (count,), "int32", role="input")
    )
    positions = graph.buffer("positions", (rows,), "int32", role="state")
    active = graph.buffer("active", (rows,), "int32", role="input")
    mapping = graph.buffer("mapping", (rows,), "int32", role="input")
    row_positions, row_active = positions, active
    if chunk > 1:
        row_positions = graph.buffer("row_positions", (count,), "int32")
        row_active = graph.buffer("row_active", (count,), "int32")
        graph(
            "chunk_controls",
            dict(r=rows, chunk=chunk),
            positions,
            active,
            row_positions,
            row_active,
        )

    def scratch(name, channels, *, n=count, role="scratch", dtype="float32"):
        return graph.buffer(name, (n * channels,), dtype, role=role)

    def project(name, x, weight, depth, channels, *, n=count):
        out = scratch(name, channels, n=n)
        graph(
            "linear",
            dict(
                r=n,
                k=depth,
                o=channels,
                optimized=optimized,
                **weights.parameters(weight),
            ),
            x,
            weights[weight],
            out,
        )
        return out

    def norm(name, x, weight, *, channels=width, n=count):
        out = scratch(name, channels, n=n)
        graph("rms", dict(r=n, c=channels, eps=config.epsilon), x, weights[weight], out)
        return out

    def add(name, left, right, *, channels=width, n=count):
        out = scratch(name, channels, n=n)
        graph("add", dict(r=n, c=channels), left, right, out)
        return out

    def rotate(name, x, heads):
        out = graph.buffer(name, (count, heads * config.head_dim), "float32")
        graph(
            "rope",
            dict(
                r=count,
                h=heads,
                d=config.head_dim,
                theta=config.theta,
                provider=provider,
                position_limit=max_context,
            ),
            x,
            out,
            row_positions,
        )
        return out

    parameters = dict(
        r=count,
        h=config.heads,
        kh=config.kv_heads,
        d=config.head_dim,
        capacity=max_context,
        kv_dtype=kv_dtype,
        unit_scales=True,
        provider=provider,
        optimized=optimized,
        **({"chunk": chunk} if chunk > 1 else {}),
    )
    storage = cache_storage(parameters)
    caches = []
    for layer in range(config.depth):
        prefix = f"blk.{layer}."
        caches.append(
            tuple(
                graph.buffer(
                    prefix + name,
                    storage.values.shape,
                    storage.values.dtype,
                    role="state",
                )
                if name in ("kc", "vc")
                else graph.buffer(
                    prefix + name, storage.metadata[0].shape, "float32", role="state"
                )
                for name in ("kc", "vc", "ks", "vs")
            )
        )

    def write_cache(layer, hidden):
        prefix = f"blk.{layer}."
        key = project(
            prefix + "key",
            hidden,
            prefix + "attn_k.weight",
            width,
            config.kv_heads * config.head_dim,
        )
        key = norm(
            prefix + "key_norm",
            key,
            prefix + "attn_k_norm.weight",
            channels=config.head_dim,
            n=count * config.kv_heads,
        )
        key = rotate(prefix + "key_rotary", key, config.kv_heads)
        value = project(
            prefix + "value",
            hidden,
            prefix + "attn_v.weight",
            width,
            config.kv_heads * config.head_dim,
        )
        value = value.view((count, config.kv_heads * config.head_dim))
        graph(
            "cache_write",
            parameters,
            key,
            value,
            *caches[layer],
            row_positions,
            row_active,
        )

    features = {}
    if phase == "context":
        target_hidden = scratch("target_hidden", config.feature_width, role="input")
        hidden = project(
            "context_projection",
            target_hidden,
            "fc.weight",
            config.feature_width,
            width,
        )
        hidden = norm("context_normalized", hidden, "enc.output_norm.weight")
        for layer in range(config.depth):
            write_cache(layer, hidden)
        graph("advance", dict(r=rows, chunk=chunk), positions, active)
        features["hidden_input"] = target_hidden.name
        prediction = tokens.name
    else:
        hidden = scratch("embedding", width)
        graph(
            "embedding",
            dict(
                r=count,
                c=width,
                v=config.vocab,
                start=0,
                **weights.parameters("embedding"),
            ),
            source,
            weights["embedding"],
            hidden,
        )
        limits = graph.buffer("block_limits", (count,), "int32")
        graph("block_limits", dict(r=rows, chunk=chunk), positions, active, limits)
        for layer in range(config.depth):
            prefix = f"blk.{layer}."
            normal = norm(prefix + "normalized", hidden, prefix + "attn_norm.weight")
            query = project(
                prefix + "query", normal, prefix + "attn_q.weight", width, width
            )
            query = norm(
                prefix + "query_norm",
                query,
                prefix + "attn_q_norm.weight",
                channels=config.head_dim,
                n=count * config.heads,
            )
            query = rotate(prefix + "query_rotary", query, config.heads)
            write_cache(layer, normal)
            attended = scratch(prefix + "attended", width)
            floating_attention(
                graph, parameters, query, *caches[layer], attended, limits, row_active
            )
            projected = project(
                prefix + "attention_output",
                attended,
                prefix + "attn_output.weight",
                width,
                width,
            )
            hidden = add(prefix + "attention_residual", hidden, projected)
            normal = norm(prefix + "ffn_normalized", hidden, prefix + "ffn_norm.weight")
            gate = project(
                prefix + "gate", normal, prefix + "ffn_gate.weight", width, config.ff
            )
            up = project(
                prefix + "up", normal, prefix + "ffn_up.weight", width, config.ff
            )
            activated = scratch(prefix + "activated", config.ff)
            graph("swiglu", dict(r=count, c=config.ff), gate, up, activated)
            down = project(
                prefix + "down", activated, prefix + "ffn_down.weight", config.ff, width
            )
            hidden = add(prefix + "output", hidden, down)
        hidden = norm("normalized_output", hidden, "output_norm.weight")
        base = project("base_logits", hidden, "head", width, config.vocab)
        previous = scratch("anchor", 1, n=rows, dtype="int32", role="input")
        # Each block starts with a known anchor; the coordinator supplies it.
        features["anchor_input"] = previous.name
        predicted = source
        confidence = None
        for index in range(chunk):
            prefix = f"sample.{index}."
            latent = scratch(prefix + "markov", config.markov_rank, n=rows)
            graph(
                "embedding",
                dict(
                    r=rows,
                    c=config.markov_rank,
                    v=config.vocab,
                    start=0,
                    **weights.parameters("markov_w1.weight"),
                ),
                previous,
                weights["markov_w1.weight"],
                latent,
            )
            bias = project(
                prefix + "bias",
                latent,
                "markov_w2.weight",
                config.markov_rank,
                config.vocab,
                n=rows,
            )
            step_base = scratch(prefix + "base", config.vocab, n=rows)
            graph(
                "select_token",
                dict(r=rows, chunk=chunk, c=config.vocab, index=index),
                base,
                step_base,
            )
            logits = add(
                prefix + "logits", step_base, bias, channels=config.vocab, n=rows
            )
            sampled = scratch(prefix + "token", 1, n=rows, dtype="int32")
            graph("argmax", dict(r=rows, c=config.vocab), logits, sampled, active)
            out = scratch(
                prefix + "predictions",
                chunk,
                n=rows,
                dtype="int32",
                role="output" if index == chunk - 1 else "scratch",
            )
            graph(
                "replace_token",
                dict(r=rows, chunk=chunk, index=index),
                predicted,
                sampled,
                out,
            )
            predicted = out
            step_hidden = scratch(prefix + "hidden", width, n=rows)
            graph(
                "select_token",
                dict(r=rows, chunk=chunk, c=width, index=index),
                hidden,
                step_hidden,
            )
            joined = scratch(
                prefix + "confidence_input", width + config.markov_rank, n=rows
            )
            graph(
                "concatenate",
                dict(r=rows, left=width, right=config.markov_rank),
                step_hidden,
                latent,
                joined,
            )
            score = project(
                prefix + "confidence_projection",
                joined,
                "conf_proj.weight",
                width + config.markov_rank,
                1,
                n=rows,
            )
            probability = scratch(prefix + "confidence", 1, n=rows)
            graph(
                "sigmoid_bias",
                dict(r=rows),
                score,
                weights["conf_proj.bias"],
                probability,
            )
            if confidence is None:
                confidence = probability
            else:
                packed = scratch(prefix + "confidences", index + 1, n=rows)
                graph(
                    "concatenate",
                    dict(r=rows, left=index, right=1),
                    confidence,
                    probability,
                    packed,
                )
                confidence = packed
            previous = sampled
        prediction = predicted.name
        features.update(
            tokens=predicted.name, confidence=confidence.name, logits=base.name
        )
    for name, spec in tuple(graph.buffers.items()):
        if spec.role == "state":
            lifted = replace(spec, shape=(max_requests, *spec.shape[1:]))
            object.__setattr__(lifted, "_graph", graph)
            graph.buffers[name] = lifted
    graph.bind_requests(
        inputs=dict(tokens=source.name, mapping=mapping.name, lengths=active.name),
        prediction=prediction,
        reset=["positions"],
        features=features,
    )
    return Composition(graph)


class DSparkDraft(Definition):
    checkpoint_type = DSparkDraftCheckpoint
    model_defaults = dict(kv_dtype="float16", optimized=True)
