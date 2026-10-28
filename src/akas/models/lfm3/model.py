"""LFM3 composition over shared operations and the common request executor."""

from akas.models.definition import Composition, Definition
from akas.models.lfm3.weights import LFM3Checkpoint, rotary_parameters


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
    prefill_chunk=512,
    optimized_attention=True,
    weight_arena=True,
    half_activations=False,
    q4_layout="raw",
    q8_ffn=False,
    q8_projections=False,
    fused_sandwich=False,
    fused_q8_ffn=False,
    ffn_tile_m=128,
    ffn_tile_k=64,
):
    from dataclasses import replace
    from akas.graph import Graph

    c = checkpoint.config
    if q8_ffn and (checkpoint.gguf is None or not half_activations):
        raise ValueError("Q8 FFN prefill requires GGUF and half_activations")
    if q8_projections and not q8_ffn:
        raise ValueError("Q8 projections require Q8 FFN prefill")
    if fused_q8_ffn and not q8_ffn:
        raise ValueError("Fused Q8 FFN requires Q8 FFN prefill")
    if provider != "cuda" or kv_dtype != "float16":
        raise ValueError("LFM3 currently requires CUDA and float16 KV")
    if (
        phase not in (None, "decode", "prefill")
        or not 1 <= rows <= max_requests
        or not 1 <= chunk <= max_context <= c.max_context
        or chunk > prefill_chunk
    ):
        raise ValueError("invalid LFM3 request geometry or phase")
    if ffn_tile_m not in (32, 64, 128, 256) or ffn_tile_k not in (32, 64, 128, 256):
        raise ValueError("invalid LFM3 paired projection tile")
    g = Graph(contract="reference", phase="forward")
    b, w = rows * chunk, c.width
    weights = checkpoint.graph_weights(g, provider=provider, q4_layout=q4_layout)
    stride = min(1024, max_context)
    blocks = (max_context + stride - 1) // stride
    tables = {}
    for i, key in enumerate(rotary_parameters(c)):
        tables[key] = (
            g.buffer(f"rope.{i}.fine", (stride, key[1]), "float32", role="weight"),
            g.buffer(f"rope.{i}.coarse", (blocks, key[1]), "float32", role="weight"),
        )
    tokens = g.buffer("tokens", (rows,), "int32", role="input")
    pos = g.buffer("positions", (max_requests,), "int32", role="state")
    active = g.buffer("active", (rows,), "int32", role="input")
    mapping = g.buffer("mapping", (rows,), "int32", role="input")
    source, row_pos, row_active = tokens, pos, active
    if chunk > 1:
        source = g.buffer("prefill_tokens", (b,), "int32", role="input")
        row_pos = g.buffer("row_positions", (b,), "int32")
        row_active = g.buffer("row_active", (b,), "int32")
        g("chunk_controls", dict(r=rows, chunk=chunk), pos, active, row_pos, row_active)

    narrow = (
        half_activations
        and checkpoint.gguf is not None
        and b >= 128
        and all(k % 512 == 0 for k in (w, c.ff, c.heads * c.head_dim))
    )

    def scratch(name, width, count=b, dtype="float32"):
        return g.buffer(name, (count * width,), dtype)

    half_inputs = {}
    quantized_inputs = {}
    terminal = f"blk.{len(c.layers) - 1}."

    def quantize_input(name, x, depth, count):
        from akas.quantization import get

        if x.name not in quantized_inputs:
            storage = get("activations", "q8").storage((count, depth), provider="cuda")
            packed = g.buffer(name + ".q8", storage.values.shape, storage.values.dtype)
            scale, total = [
                g.buffer(name + suffix, spec.shape, spec.dtype)
                for suffix, spec in zip((".q8_scales", ".q8_sums"), storage.metadata)
            ]
            g(
                "quantize_q8",
                dict(r=count, k=depth, input_dtype=x.dtype),
                x,
                packed,
                scale,
                total,
            )
            quantized_inputs[x.name] = packed, scale
        return quantized_inputs[x.name]

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
                    weight_options={"scale_layout": "block"},
                ),
                lhs,
                weights[weight],
                weights.scales[weight],
                matrix,
            )
        else:
            # Preserve final-row pruning after the last convolution state write.
            tail = (
                c.layers[-1] == "conv"
                and name.startswith(terminal)
                and name != terminal + "projection"
            )
            if narrow and not tail and count >= 128 and x.dtype == "float32":
                if x.name not in half_inputs:
                    half = scratch(name + ".half_input", k, count, "float16")
                    g("linear_half_input", dict(r=count, k=k), x, half)
                    half_inputs[x.name] = half
                x = half_inputs[x.name]
            if (
                q8_ffn
                and narrow
                and not tail
                and count >= 128
                and (
                    weight.endswith("ffn_down.weight") or (q8_projections and o >= 512)
                )
                and weights.parameters(weight).get("type") == 2
            ):
                packed, scale = quantize_input(name, x, k, count)
                g(
                    "q4_q8_linear",
                    dict(r=count, k=k, o=o, paired=False, **weights.parameters(weight)),
                    packed,
                    scale,
                    weights[weight],
                    out,
                )
                return out
            tiling = {}
            if (
                count > 1
                and checkpoint.gguf is not None
                and weights.parameters(weight).get("type") == 2
            ):
                if weight.endswith(("ffn_gate.weight", "ffn_up.weight")):
                    tiling = dict(
                        prefill_tile_m=min(count, ffn_tile_m),
                        prefill_tile_n=64,
                        prefill_tile_k=min(k, ffn_tile_k),
                        prefill_tile_threads=128,
                    )
                elif count >= 128 and k > w:
                    tiling = dict(
                        prefill_tile_m=64,
                        prefill_tile_n=128,
                        prefill_tile_k=64,
                        prefill_tile_stages=2,
                        prefill_tile_threads=128,
                    )
                elif count >= 128 and 32 < o <= 2048:
                    # Small output dimensions otherwise leave most SMs idle.
                    tiling = dict(
                        prefill_tile_m=32 if o < 1024 else 128,
                        prefill_tile_n=64,
                        prefill_tile_k=128,
                    )
            for wb, start, n in weights.matrix(weight):
                g(
                    "linear",
                    dict(
                        r=count,
                        k=k,
                        o=n,
                        optimized=True,
                        short_packed=True,
                        tiled_packed=True,
                        vector_q8=True,
                        **({"input_dtype": "float16"} if x.dtype == "float16" else {}),
                        **weights.parameters(weight),
                        **tiling,
                    ),
                    x,
                    wb,
                    out,
                )
        return out

    def norm(name, x, weight, count=b, width=w):
        half = (
            narrow
            and width == w
            and name.endswith((".normalized", ".ffn_normalized"))
            and name != terminal + "ffn_normalized"
        )
        out = scratch(name, width, count, "float16" if half else "float32")
        parameters = dict(
            r=count,
            c=width,
            eps=c.epsilon,
            **(
                dict(output_dtype="float16", algorithm="fragment", flat=True)
                if half
                else {}
            ),
        )
        if (
            not name.startswith(f"blk.{len(c.layers) - 1}.ffn_")
            and g.calls
            and g.calls[-1].kind == "add"
            and g.calls[-1].arguments[-1] == x.name
            and g.calls[-1].parameters == dict(r=count, c=width)
        ):
            addition = g.calls.pop()
            if (
                fused_sandwich
                and g.calls
                and g.calls[-1].kind == "rms"
                and g.calls[-1].arguments[-1] == addition.arguments[1]
                and g.calls[-1].parameters == dict(r=count, c=width, eps=c.epsilon)
            ):
                post = g.calls.pop()
                g(
                    "post_add_rms",
                    parameters,
                    post.arguments[0],
                    post.arguments[1],
                    addition.arguments[0],
                    weights[weight],
                    addition.arguments[2],
                    out,
                )
            else:
                g("add_rms", parameters, *addition.arguments, weights[weight], out)
        else:
            g("rms", parameters, x, weights[weight], out)
        return out

    def add(name, x, y):
        out = scratch(name, w)
        g("add", dict(r=b, c=w), x, y, out)
        return out

    hidden = scratch("hidden", w)
    for wb, start, n in weights.matrix("token_embd.weight"):
        g(
            "embedding",
            dict(r=b, c=w, v=n, start=start, **weights.parameters("token_embd.weight")),
            source,
            wb,
            hidden,
        )
    resets = ["positions"]
    for i, kind in enumerate(c.layers):
        p = f"blk.{i}."
        normal = norm(p + "normalized", hidden, p + "attn_norm.weight")
        if kind == "conv":
            projection = project(
                p + "projection", normal, p + "shortconv.in_proj.weight", w, 3 * w
            )
            state = g.buffer(
                p + "conv_state", (max_requests, c.conv - 1, w), "float32", role="state"
            )
            resets.append(state.name)
            mixed = scratch(p + "mixed", w)
            g(
                "conv",
                dict(r=rows, c=w, length=c.conv, chunk=chunk),
                projection,
                weights[p + "shortconv.conv.weight"],
                state,
                mixed,
                active,
            )
            mixed = project(
                p + "operator_output", mixed, p + "shortconv.out_proj.weight", w, w
            )
        else:
            d, h, kh = c.head_dim, c.heads, c.kv_heads
            q = project(p + "q", normal, p + "attn_q.weight", w, h * d)
            k = project(p + "k", normal, p + "attn_k.weight", w, kh * d)
            v = project(p + "v_flat", normal, p + "attn_v.weight", w, kh * d)
            v = g.view(p + "v", v, (b, kh * d))
            q = norm(p + "qn", q, p + "attn_q_norm.weight", b * h, d)
            k = norm(p + "kn", k, p + "attn_k_norm.weight", b * kh, d)
            qr = g.buffer(p + "qr", (b, h * d), "float32")
            kr = g.buffer(p + "kr", (b, kh * d), "float32")
            for x, out, heads in ((q, qr, h), (k, kr, kh)):
                g(
                    "rotary_compact",
                    dict(
                        r=b,
                        h=heads,
                        d=d,
                        theta=c.theta[i],
                        rotary=c.rotary[i],
                        full_head_pairs=True,
                        capacity=max_context,
                        stride=stride,
                        blocks=blocks,
                    ),
                    x,
                    out,
                    row_pos,
                    *tables[c.theta[i], c.rotary[i]],
                )
            window = c.window if kind == "sliding_attention" else 0
            # Cache writes precede attention. Preserve every query's full window.
            capacity = (
                min(max_context, window + prefill_chunk) if window else max_context
            )
            shape = (max_requests, kh, capacity, d)
            kc, vc = [
                g.buffer(p + n, shape, "float16", role="state") for n in ("kc", "vc")
            ]
            ap = dict(
                r=b, h=h, kh=kh, d=d, capacity=capacity, chunk=chunk, window=window
            )
            g("window_cache", ap, kr, v, kc, vc, row_pos, row_active)
            splits = (4 if window else 32) if chunk == 1 else 1
            ap["splits"] = splits
            if optimized_attention and chunk > 1:
                ap.update(query_rows=64, key_rows=32, contiguous=not window)
            parts = scratch(p + "attention_parts", h * splits * (d + 2))
            attended = scratch(p + "attended", h * d)
            g(
                "window_grouped"
                if optimized_attention and chunk == 1
                else "window_attention",
                ap,
                qr,
                kc,
                vc,
                row_pos,
                row_active,
                parts,
            )
            g("window_merge", ap, parts, attended, row_active)
            gate = project(p + "attention_gate", normal, p + "attn_gate.weight", w, h)
            gated = scratch(p + "gated", h * d)
            g("head_gate", dict(r=b, h=h, d=d), attended, gate, gated)
            mixed = project(
                p + "operator_output", gated, p + "attn_output.weight", h * d, w
            )
        if c.sandwich:
            mixed = norm(p + "post_operator", mixed, p + "post_attention_norm.weight")
        hidden = add(p + "residual", hidden, mixed)
        normal = norm(p + "ffn_normalized", hidden, p + "ffn_norm.weight")
        if (
            normal.dtype == "float16"
            and weights.parameters(p + "ffn_gate.weight").get("type") == 2
            and weights.parameters(p + "ffn_gate.weight")
            == weights.parameters(p + "ffn_up.weight")
        ):
            activated = scratch(p + "activated", c.ff, dtype="float16")
            parameters = dict(
                r=b,
                k=w,
                o=c.ff,
                optimized=True,
                tiled_packed=True,
                input_dtype="float16",
                output_dtype="float16",
                prefill_tile_m=min(b, ffn_tile_m),
                prefill_tile_n=64,
                prefill_tile_k=min(w, ffn_tile_k),
                prefill_tile_threads=128,
                **weights.parameters(p + "ffn_gate.weight"),
            )
            if q8_ffn:
                packed, scale = quantize_input(p + "ffn", normal, w, b)
                g(
                    "q4_q8_linear",
                    {**parameters, "paired": True},
                    packed,
                    scale,
                    weights[p + "ffn_gate.weight"],
                    weights[p + "ffn_up.weight"],
                    activated,
                )
            else:
                g(
                    "ffn",
                    parameters,
                    normal,
                    weights[p + "ffn_gate.weight"],
                    weights[p + "ffn_up.weight"],
                    activated,
                )
        else:
            gate = project(p + "gate", normal, p + "ffn_gate.weight", w, c.ff)
            up = project(p + "up", normal, p + "ffn_up.weight", w, c.ff)
            activated = scratch(p + "activated", c.ff)
            g("swiglu", dict(r=b, c=c.ff), gate, up, activated)
        mixed = project(p + "ffn_output", activated, p + "ffn_down.weight", c.ff, w)
        if c.sandwich:
            mixed = norm(p + "post_ffn", mixed, p + "post_ffw_norm.weight")
        hidden = add(p + "output", hidden, mixed)
    if chunk > 1:
        last = scratch("last_hidden", w, rows)
        g("chunk_last", dict(r=rows, c=w, chunk=chunk), hidden, last, active)
        hidden = last
    hidden = norm("normalized_output", hidden, "output_norm.weight", rows)
    head = "token_embd.weight" if c.tied else "output.weight"
    logits = project("logits", hidden, head, w, c.vocab, rows)
    g.buffers["logits"] = replace(logits, role="output")
    sampled = g.buffer("sampled", (rows,), "int32", role="output")
    g("argmax", dict(r=rows, c=c.vocab), logits, sampled, active)
    g("next_tokens", dict(r=rows), sampled, tokens)
    g("advance", dict(r=rows, **({"chunk": chunk} if chunk > 1 else {})), pos, active)
    g.bind_requests(
        inputs={"tokens": source.name, "mapping": mapping.name, "lengths": active.name},
        prediction="sampled",
        features={"logits": "logits"},
        reset=resets,
    )
    from akas.compilation.rewrites import optimize

    g = optimize(g)
    if fused_q8_ffn:
        from akas.graph import Call

        calls, i = [], 0
        while i < len(g.calls):
            call = g.calls[i]
            if (
                call.kind == "q4_q8_linear"
                and call.parameters.get("paired")
                and i + 1 < len(g.calls)
            ):
                following = g.calls[i + 1]
                if (
                    following.kind == "quantize_q8"
                    and following.arguments[0] == call.arguments[-1]
                ):
                    calls.append(
                        Call(
                            "q4_q8_ffn",
                            call.parameters,
                            (*call.arguments[:-1], *following.arguments[1:3]),
                        )
                    )
                    i += 2
                    continue
            calls.append(call)
            i += 1
        g.calls = calls
    return Composition(g)


class LFM3(Definition):
    checkpoint_architectures = ("lfm3",)
    checkpoint_type = LFM3Checkpoint
    model_defaults = {"kv_dtype": "float16", "prefill_chunk": 512}
    optimization_chunks = (512, 1024, 2048, 256, 128, 32, 1)
