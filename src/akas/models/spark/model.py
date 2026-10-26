"""Spark composition over shared projections, rotary, window attention and GELU."""

from akas.models.definition import Composition, Definition
from akas.models.spark.weights import SparkCheckpoint


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
    prefill_chunk=256,
    rotary_cache=False,
    optimized_attention=False,
    embedding_location="device",
    fused_ffn=False,
    weight_arena=False,
    half_activations=False,
    compact_swa=False,
):
    from akas.graph import Graph

    c = checkpoint.config
    if provider != "cuda" or kv_dtype != "float16":
        raise ValueError("Spark currently requires CUDA with float16 KV")
    if embedding_location not in ("device", "host") or (
        embedding_location == "host" and (checkpoint.gguf is None or c.tied)
    ):
        raise ValueError("host embeddings require a GGUF with a separate output head")
    if (
        phase not in (None, "decode", "prefill")
        or not 1 <= rows <= max_requests
        or not 1 <= chunk <= max_context <= c.max_context
        or chunk > prefill_chunk
    ):
        raise ValueError("invalid Spark request geometry or phase")
    if compact_swa and not rotary_cache:
        raise ValueError("compact sliding cache requires compact rotary tables")
    g = Graph(contract="reference", phase="forward")
    b, w = rows * chunk, c.width
    weights = checkpoint.graph_weights(g, provider=provider)
    rotary_tables = {}
    rotary_stride = min(1024, max_context)
    rotary_blocks = (max_context + rotary_stride - 1) // rotary_stride
    if rotary_cache:
        from akas.models.spark.weights import rotary_parameters

        for i, (theta, rotary) in enumerate(rotary_parameters(c)):
            rotary_tables[theta, rotary] = (
                g.buffer(
                    f"rope.{i}.fine", (rotary_stride, rotary), "float32", role="weight"
                ),
                g.buffer(
                    f"rope.{i}.coarse",
                    (rotary_blocks, rotary),
                    "float32",
                    role="weight",
                ),
            )
    tokens = g.buffer("tokens", (rows,), "int32", role="input")
    pos = g.buffer("positions", (max_requests,), "int32", role="state")
    active = g.buffer("active", (rows,), "int32", role="input")
    mapping = g.buffer("mapping", (rows,), "int32", role="input")
    row_pos, row_active, source = pos, active, tokens
    if chunk > 1:
        source = g.buffer("prefill_tokens", (b,), "int32", role="input")
        row_pos = g.buffer("row_positions", (b,), "int32")
        row_active = g.buffer("row_active", (b,), "int32")
        g("chunk_controls", dict(r=rows, chunk=chunk), pos, active, row_pos, row_active)

    narrow = half_activations and checkpoint.gguf is not None and b >= 128

    def scratch(name, width, count=b, dtype="float32"):
        return g.buffer(name, (count * width,), dtype)

    def params(name):
        return (
            weights.parameters(name)
            if hasattr(weights, "parameters")
            else {"weight_dtype": "float16"}
        )

    def project(name, x, weight, k, o, count=b):
        out = scratch(name, o, count)
        if weight in getattr(weights, "scales", {}):
            lhs = g.view(name + ".input_matrix", x, (count, k))
            matrix = g.view(name + ".output_matrix", out, (count, o))
            g(
                "quantized_linear",
                dict(
                    r=count,
                    k=k,
                    o=o,
                    weight_format="fp8",
                    weight_options={"scale_layout": "channel"},
                ),
                lhs,
                weights[weight],
                weights.scales[weight],
                matrix,
            )
            return out
        for weight_buffer, start, n in weights.matrix(weight):
            g(
                "linear",
                dict(
                    r=count,
                    k=k,
                    o=n,
                    optimized=True,
                    **({"input_dtype": x.dtype} if x.dtype == "float16" else {}),
                    **(
                        dict(vector_q8=True, short_packed=True, tiled_packed=True)
                        if optimized_attention
                        else {}
                    ),
                    **params(weight),
                    **(
                        {}
                        if hasattr(weights, "parameters")
                        else dict(output_stride=o, output_offset=start)
                    ),
                ),
                x,
                weight_buffer,
                out,
            )
        return out

    def norm(name, x, weight, count=b):
        half = narrow and count >= 128
        out = scratch(name, w, count, dtype="float16" if half else "float32")
        parameters = dict(
            r=count,
            c=w,
            eps=c.epsilon,
            **(
                {"output_dtype": "float16", "algorithm": "fragment", "flat": True}
                if half
                else {}
            ),
        )
        if (
            half
            and g.calls
            and g.calls[-1].kind == "add"
            and g.calls[-1].arguments[-1] == x.name
        ):
            addition = g.calls.pop()
            g("add_rms", parameters, *addition.arguments, weights[weight], out)
        else:
            g("rms", parameters, x, weights[weight], out)
        return out

    def add(name, x, y):
        out = scratch(name, w)
        g("add", dict(r=b, c=w), x, y, out)
        return out

    hidden = scratch("hidden", w)
    for weight, start, n in weights.matrix("token_embd.weight"):
        g(
            "embedding",
            dict(r=b, c=w, v=n, start=start, **params("token_embd.weight")),
            source,
            weight,
            hidden,
        )
    for i, kind in enumerate(c.layers):
        p = f"blk.{i}."
        normal = norm(p + "normalized", hidden, p + "attn_norm.weight")
        qkv = project(
            p + "qkv",
            normal,
            p + "attn_qkv.weight",
            w,
            (c.heads + 2 * c.kv_heads) * c.head_dim,
        )
        window = c.window if kind == "sliding_attention" else 0
        # The whole chunk writes before attention reads. Retain its earliest query's window.
        capacity = (
            min(max_context, window if compact_swa else window + prefill_chunk)
            if window
            else max_context
        )
        state_shape = (max_requests, c.kv_heads, capacity, c.head_dim)
        kc, vc = [
            g.buffer(p + n, state_shape, "float16", role="state") for n in ("kc", "vc")
        ]
        ap = dict(
            r=b,
            h=c.heads,
            kh=c.kv_heads,
            d=c.head_dim,
            capacity=capacity,
            chunk=chunk,
            window=window,
        )
        current_window = bool(compact_swa and window and chunk > 1)
        if (narrow and rotary_cache) or current_window:
            query_storage = scratch(
                p + "query_storage", c.heads * c.head_dim, dtype="float16"
            )
            qr = g.view(p + "qr", query_storage, (b, c.heads * c.head_dim))
            if current_window:
                current_shape = (rows, c.kv_heads, chunk, c.head_dim)
                current_k, current_v = [
                    g.buffer(p + n, current_shape, "float16")
                    for n in ("current_k", "current_v")
                ]
            else:
                current_k, current_v = kc, vc
            g(
                "qkv_rotary_cache",
                dict(
                    ap,
                    **(
                        {"capacity": chunk, "current_chunk": True}
                        if current_window
                        else {}
                    ),
                    theta=c.theta[i],
                    rotary=c.rotary[i],
                    context=max_context,
                    stride=rotary_stride,
                    blocks=rotary_blocks,
                ),
                qkv,
                qr,
                current_k,
                current_v,
                row_pos,
                row_active,
                *rotary_tables[c.theta[i], c.rotary[i]],
            )
            ap["query_dtype"] = "float16"
        else:
            q, k, v = (
                scratch(p + "q", c.heads * c.head_dim),
                scratch(p + "k", c.kv_heads * c.head_dim),
                g.buffer(p + "v", (b, c.kv_heads * c.head_dim), "float32"),
            )
            g(
                "qkv_split",
                dict(r=b, h=c.heads, kh=c.kv_heads, d=c.head_dim),
                qkv,
                q,
                k,
                v,
            )
            qr, kr = (
                g.buffer(p + "qr", (b, c.heads * c.head_dim), "float32"),
                g.buffer(p + "kr", (b, c.kv_heads * c.head_dim), "float32"),
            )
            for x, out, h in ((q, qr, c.heads), (k, kr, c.kv_heads)):
                rp = dict(r=b, h=h, d=c.head_dim, theta=c.theta[i], rotary=c.rotary[i])
                if rotary_cache:
                    g(
                        "rotary_compact",
                        dict(
                            rp,
                            capacity=max_context,
                            stride=rotary_stride,
                            blocks=rotary_blocks,
                        ),
                        x,
                        out,
                        row_pos,
                        *rotary_tables[c.theta[i], c.rotary[i]],
                    )
                else:
                    g("rotary_partial", rp, x, out, row_pos)
            g("window_cache", ap, kr, v, kc, vc, row_pos, row_active)
        splits = (
            (4 if window else 32)
            if optimized_attention and chunk == 1
            else (16 if chunk == 1 else 1)
        )
        ap["splits"] = splits
        if optimized_attention and chunk > 1:
            ap.update(query_rows=64, key_rows=32, contiguous=not window)
        parts = scratch(p + "attention_parts", c.heads * splits * (c.head_dim + 2))
        attended = scratch(p + "attended", c.heads * c.head_dim)
        g(
            "window_grouped"
            if optimized_attention and chunk == 1
            else "window_attention_current"
            if current_window
            else "window_attention",
            ap,
            qr,
            kc,
            vc,
            *((current_k, current_v) if current_window else ()),
            row_pos,
            row_active,
            parts,
        )
        if current_window:
            g(
                "window_cache_tail",
                ap,
                current_k,
                current_v,
                kc,
                vc,
                row_pos,
                row_active,
            )
        g("window_merge", ap, parts, attended, row_active)
        gate = project(p + "attention_gate", normal, p + "attn_gate.weight", w, c.heads)
        gated = scratch(
            p + "gated", c.heads * c.head_dim, dtype="float16" if narrow else "float32"
        )
        g(
            "head_gate_half" if narrow else "head_gate",
            dict(r=b, h=c.heads, d=c.head_dim),
            attended,
            gate,
            gated,
        )
        mixed = project(
            p + "attention_output",
            gated,
            p + "attn_output.weight",
            c.heads * c.head_dim,
            w,
        )
        hidden = add(p + "residual", hidden, mixed)
        normal = norm(p + "ffn_normalized", hidden, p + "ffn_norm.weight")
        if (
            fused_ffn
            and b >= 128
            and checkpoint.gguf is not None
            and w % 128 == 0
            and params(p + "ffn_gate.weight") == params(p + "ffn_up.weight")
            and params(p + "ffn_gate.weight").get("type") in (8, 12, 14, 30)
        ):
            activated = scratch(
                p + "activated", c.ff, dtype="float16" if narrow else "float32"
            )
            g(
                "geglu_linear",
                dict(
                    r=b,
                    k=w,
                    o=c.ff,
                    input_dtype=normal.dtype,
                    output_dtype=activated.dtype,
                    **params(p + "ffn_gate.weight"),
                ),
                normal,
                weights[p + "ffn_gate.weight"],
                weights[p + "ffn_up.weight"],
                activated,
            )
        else:
            gate = project(p + "gate", normal, p + "ffn_gate.weight", w, c.ff)
            up = project(p + "up", normal, p + "ffn_up.weight", w, c.ff)
            activated = scratch(p + "activated", c.ff)
            g("geglu", dict(r=b, c=c.ff), gate, up, activated)
        mixed = project(p + "ffn_output", activated, p + "ffn_down.weight", c.ff, w)
        hidden = add(p + "output", hidden, mixed)
    if chunk > 1:
        last = scratch("last_hidden", w, rows)
        g("chunk_last", dict(r=rows, c=w, chunk=chunk), hidden, last, active)
        hidden = last
    hidden = norm("normalized_output", hidden, "output_norm.weight", rows)
    head = "token_embd.weight" if c.tied else "output.weight"
    logits = project("logits", hidden, head, w, c.vocab, rows)
    from dataclasses import replace

    g.buffers["logits"] = replace(logits, role="output")
    sampled = g.buffer("sampled", (rows,), "int32", role="output")
    g("argmax", dict(r=rows, c=c.vocab), logits, sampled, active)
    g("next_tokens", dict(r=rows), sampled, tokens)
    g("advance", dict(r=rows, **({"chunk": chunk} if chunk > 1 else {})), pos, active)
    g.bind_requests(
        inputs={"tokens": source.name, "mapping": mapping.name, "lengths": active.name},
        prediction="sampled",
        features={"logits": "logits"},
        reset=["positions"],
    )
    return Composition(g)


class Spark(Definition):
    checkpoint_architectures = ("spark2_5",)
    checkpoint_type = SparkCheckpoint
    model_defaults = {"kv_dtype": "float16", "prefill_chunk": 256}
    optimization_chunks = (256, 128, 32, 1)
