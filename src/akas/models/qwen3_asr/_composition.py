"""Forward graphs for Qwen3 ASR audio embedding and autoregressive decoding."""

from math import prod

from akas.graph import Graph
from akas.models.qwen3_asr.audio import CONV_FRAMES, CONV_ROWS
from akas.models.qwen3_asr.weights import CONV_PROJECTION

DECODE_BLOCK = 4
PARTITIONED_DECODE = "decode_partitioned"
PARTITION_MIN_PREFIX = 128
DECODE_FEEDBACK_BUFFERS = ("token", "previous_token", "positions", "active")


def feedback():
    graph = Graph(contract="reference")
    sampled, tokens, positions, active = (
        graph.buffer(name, (1,), "int32", role="input")
        for name in DECODE_FEEDBACK_BUFFERS
    )
    stops = graph.buffer("decode_stop_ids", (2,), "int32", role="weight")
    journal = graph.buffer(
        "decode_journal", (1, DECODE_BLOCK + 1), "int32", role="state"
    )
    graph(
        "decode_feedback",
        dict(r=1, capacity=DECODE_BLOCK),
        sampled,
        stops,
        tokens,
        positions,
        active,
        journal,
    )
    return graph


def _linear(
    graph,
    checkpoint,
    x,
    weight,
    name,
    *,
    role="scratch",
    up=None,
    residual=None,
    tap_major=False,
    integer=False,
    signed_decode=False,
):
    rows, depth = x.shape
    info = checkpoint.tensors[weight]
    columns, *inner = info.shape
    if prod(inner) != depth:
        raise ValueError(f"projection input does not match {weight}")
    one_component = integer == "q8" and depth % 64 == 0
    integer = bool(integer) and rows > 1 and info.type == 8 and depth % 32 == 0
    signed_decode = signed_decode and rows == 1 and info.type == 8 and depth % 128 == 0
    half = rows > 1 and info.type == 8 and not integer
    aligned = integer or signed_decode
    packed = checkpoint.buffer(
        graph,
        weight,
        provider="webgpu",
        half=half,
        tap_major=tap_major,
        aligned=aligned,
    )
    if depth % 32:
        from akas.ops.linear.definition import project
        from akas.quantization.weights import Weight

        matrix = Weight(
            (columns, depth),
            (packed,),
            (dict(format="ggml", rows=columns, options={"encoding": info.type}),),
        )
        return project(graph, x, matrix, width=depth, name=name, role=role)
    out = graph.buffer(name, (rows, columns), "float32", role=role)
    parameters = dict(
        r=rows, k=depth, o=columns, ggml=True, type=1 if half else info.type
    )
    if rows > 1 and parameters["type"] == 1:
        parameters.update(half_accum=True, input_dtype=x.dtype)
        if tap_major and depth % 32 == 0:
            parameters["half_k"] = 32
    if depth % 64:
        parameters["tile"] = (16, 32, 32)
    if integer or signed_decode:
        two_component = integer and not one_component
        activation = _packed_activation(graph, x, two_component=two_component)
        parameters["q16" if two_component else "signed_q8_input"] = True
        parameters["q8_aligned"] = True
    else:
        activation = (x.view((rows * depth,)),)
    if rows == 1 and info.type == 8 and not signed_decode:
        parameters["vector_q8_webgpu"] = True
    kind, inputs = "linear", ()
    if up is not None:
        kind = "ffn"
        inputs = (
            checkpoint.buffer(graph, up, provider="webgpu", half=half, aligned=aligned),
        )
    elif residual is not None:
        kind = "linear_add"
        inputs = (residual.view((rows * columns,)),)
    graph(
        kind,
        parameters,
        *activation,
        packed,
        *inputs,
        out.view((rows * columns,)),
    )
    return out


def _packed_activation(graph, x, *, two_component):
    from akas.quantization.activations.q8 import FORMAT as q8
    from akas.quantization.activations.q16 import FORMAT as q16

    rows, depth = x.shape
    codec = q16 if two_component else q8
    name = f"{graph.storage_owner(x.name)}.{codec.name}"
    names = tuple(f"{name}.{part}" for part in ("packed", "scales", "sums"))
    if names[0] not in graph.buffers:
        storage = codec.storage(x.shape, provider="webgpu")
        buffers = tuple(
            graph.buffer(key, layout.shape, layout.dtype)
            for key, layout in zip(names, storage.buffers)
        )
        graph(
            f"quantize_{codec.name}",
            dict(r=rows, k=depth),
            x.view((rows * depth,)),
            *buffers,
        )
    return tuple(graph.buffers[key] for key in (names if two_component else names[:2]))


def _add(graph, x, y, name, *, role="scratch"):
    rows, width = x.shape
    flat = (rows * width,)
    out = graph.buffer(name, x.shape, "float32", role=role)
    graph("add", dict(r=rows, c=width), x.view(flat), y.view(flat), out.view(flat))
    return out


def _activated(graph, checkpoint, x, name, *, bias, gelu=False, half=False):
    rows, width = x.shape
    weights = checkpoint.buffer(graph, bias, provider="webgpu")
    out = graph.buffer(name, x.shape, "float16" if half else "float32")
    parameters = dict(r=rows, c=width)
    if half:
        parameters["output_dtype"] = "float16"
    graph(
        "gelu_bias" if gelu else "column_bias",
        parameters,
        x,
        weights.view((width,)),
        out,
    )
    return out


def frontend(checkpoint):
    graph = Graph(contract="reference")
    c = checkpoint.config
    x = graph.buffer("mel", (CONV_FRAMES, c.mels, 1), "float32", role="input")
    for layer in range(1, 4):
        width, height, channels = x.shape
        prefix = f"a.conv2d.{layer}"
        weight = prefix + ".weight"
        columns, in_channels, kh, kw = checkpoint.tensors[weight].shape
        if (channels, 3, 3) != (in_channels, kh, kw):
            raise ValueError("Qwen3 ASR convolution expects 3x3 filters")
        ow, oh = (width + 1) // 2, (height + 1) // 2
        params = dict(
            width=width, height=height, channels=channels, kernel=3, stride=2, padding=1
        )
        if checkpoint.tensors[weight].type == 1 and channels == 1:
            out = graph.buffer(prefix + ".projected", (ow * oh, columns), "float32")
            graph(
                "single_channel_conv2d",
                dict(params, columns=columns),
                x,
                checkpoint.buffer(graph, weight, provider="webgpu"),
                out,
            )
            x = out
        elif x.dtype == "float16":
            out = graph.buffer(prefix + ".projected", (ow * oh, columns), "float32")
            graph(
                "implicit_half_conv2d",
                dict(params, columns=columns),
                x,
                checkpoint.buffer(graph, weight, provider="webgpu", tap_major=True),
                out.view((ow * oh * columns,)),
            )
            x = out
        else:
            tap_major = checkpoint.tensors[weight].type == 1 and channels % 4 == 0
            patches = graph.buffer(
                prefix + ".patches",
                (ow * oh, channels * 9),
                "float16" if tap_major else "float32",
            )
            image = x.view((width, height, channels // 4, 4)) if tap_major else x
            graph(
                "image_patches_half" if tap_major else "image_patches",
                params,
                image,
                patches,
            )
            x = _linear(
                graph,
                checkpoint,
                patches,
                weight,
                prefix + ".projected",
                tap_major=tap_major,
            )
        implicit_next = (
            layer < 3
            and columns % 16 == 0
            and checkpoint.tensors[f"a.conv2d.{layer + 1}.weight"].type == 1
        )
        x = _activated(
            graph,
            checkpoint,
            x,
            prefix + ".gelu",
            bias=prefix + ".bias",
            gelu=True,
            half=implicit_next,
        )
        x = x.view((ow, oh, columns))
    projected = _linear(
        graph,
        checkpoint,
        x.view((CONV_ROWS, oh * columns)),
        CONV_PROJECTION,
        "projected",
    )
    position = graph.buffer("sinusoid", projected.shape, "float32", role="weight")
    _add(graph, projected, position, "features", role="output")
    return graph


def _attention(
    graph,
    q,
    k,
    v,
    positions,
    limits,
    active,
    prefix,
    *,
    heads,
    kv_heads,
    context,
    persistent,
    rotary=None,
    splits=1,
):
    rows, width = q.shape
    dim = width // heads
    params = dict(
        r=rows, h=heads, kh=kv_heads, d=dim, capacity=context, chunk=rows, window=0
    )
    cache = [
        graph.buffer(
            prefix + suffix,
            (1, kv_heads, context, dim),
            "float16",
            role="state" if persistent else "scratch",
        )
        for suffix in ("key_cache", "value_cache")
    ]
    if rotary is None:
        graph("window_cache", params, k, v, *cache, positions, active)
    else:
        wq, wk, rope, origin, theta, epsilon = rotary
        rotated = graph.buffer(prefix + "q_rot", q.shape, "float32")
        graph(
            "qk_norm_rotary_cache",
            dict(
                params,
                cache_capacity=context,
                rotary=dim,
                theta=theta,
                eps=epsilon,
                stride=context,
                blocks=1,
            ),
            q.view((rows * width,)),
            k.view((rows * kv_heads * dim,)),
            v,
            wq,
            wk,
            rotated,
            *cache,
            positions,
            active,
            rope,
            origin,
        )
        q = rotated
    out = graph.buffer(prefix + "attended", (rows * width,), "float32")
    params["splits"] = splits
    parts = graph.buffer(
        prefix + "attn_parts", (rows * heads * splits * (dim + 2),), "float32"
    )
    graph(
        "window_grouped" if rows == 1 else "window_attention",
        dict(params, query_rows=1 if rows == 1 else 4, key_rows=32),
        q,
        *cache,
        limits,
        active,
        parts,
    )
    graph("window_merge", params, parts, out, active)
    return out.view((rows, width))


def encoder(checkpoint, rows, *, integer=False):
    graph = Graph(contract="reference")
    c = checkpoint.config
    x = graph.buffer("features", (rows, c.audio_width), "float32", role="input")
    positions = graph.buffer("positions", (rows,), "int32", role="input")
    active = graph.buffer("active", (rows,), "int32", role="input")
    limits = graph.buffer("limits", (rows,), "int32", role="input")

    def dense(x, name, *, gelu=False):
        y = _linear(
            graph,
            checkpoint,
            x,
            name + ".weight",
            name + ".projected",
            integer=integer,
        )
        return _activated(
            graph, checkpoint, y, name + ".activated", bias=name + ".bias", gelu=gelu
        )

    def norm(x, name):
        out = graph.buffer(name + ".normalized", x.shape, "float32")
        w, b = (
            checkpoint.buffer(graph, name + suffix, provider="webgpu")
            for suffix in (".weight", ".bias")
        )
        graph(
            "affine_layer_norm",
            dict(r=rows, c=c.audio_width, eps=c.audio_epsilon),
            x,
            w,
            b,
            out,
        )
        return out

    for layer in range(c.audio_layers):
        prefix = f"a.blk.{layer}."
        normal = norm(x, prefix + "ln1")
        q, k, v = (dense(normal, prefix + "attn_" + tag) for tag in ("q", "k", "v"))
        attended = _attention(
            graph,
            q,
            k,
            v,
            positions,
            limits,
            active,
            prefix,
            heads=c.audio_heads,
            kv_heads=c.audio_heads,
            context=rows,
            persistent=False,
        )
        x = _add(graph, x, dense(attended, prefix + "attn_out"), prefix + "attention")
        normal = norm(x, prefix + "ln2")
        hidden = dense(normal, prefix + "ffn_up", gelu=True)
        x = _add(graph, x, dense(hidden, prefix + "ffn_down"), prefix + "ffn")
    x = dense(norm(x, "a.post_ln"), "mm.a.mlp.1", gelu=True)
    projected = _linear(
        graph, checkpoint, x, "mm.a.mlp.2.weight", "projected", integer=integer
    )
    out = graph.buffer("audio_embedding", (rows, c.width), "float32", role="output")
    bias = checkpoint.buffer(graph, "mm.a.mlp.2.bias", provider="webgpu")
    graph("column_bias", dict(r=rows, c=c.width), projected, bias, out)
    return graph


def decoder(
    checkpoint,
    rows,
    context,
    *,
    decode=False,
    integer=False,
    signed_decode=False,
    attention_splits=1,
):
    graph = Graph(contract="reference")
    c = checkpoint.config
    x = graph.buffer(
        "embedding", (rows, c.width), "float32", role="scratch" if decode else "input"
    )
    if decode:
        from akas.models.qwen3_asr.weights import TOKEN_EMBEDDING

        token = graph.buffer("previous_token", (1,), "int32", role="input")
        weight = checkpoint.buffer(graph, TOKEN_EMBEDDING, provider="webgpu")
        graph(
            "embedding",
            dict(
                r=1,
                c=c.width,
                v=c.vocab,
                ggml=True,
                type=checkpoint.tensors[TOKEN_EMBEDDING].type,
            ),
            token,
            weight,
            x.view((c.width,)),
        )
    positions = graph.buffer("positions", (rows,), "int32", role="input")
    active = graph.buffer("active", (rows,), "int32", role="input")
    rope = graph.buffer("rope", (context, c.head_dim), "float32", role="weight")
    origin = graph.buffer("rope_origin", (1, c.head_dim), "float32", role="weight")

    def residual_projection(hidden, matrix, residual, name):
        if rows == 1 or (
            integer == "q8"
            and hidden.shape[1] % 64 == 0
            and checkpoint.tensors[matrix].type == 8
        ):
            return _linear(
                graph,
                checkpoint,
                hidden,
                matrix,
                name,
                residual=residual,
                integer=integer,
                signed_decode=signed_decode,
            )
        y = _linear(
            graph, checkpoint, hidden, matrix, name + ".projected", integer=integer
        )
        return _add(graph, residual, y, name)

    def norm(x, weight, name, *, role="scratch"):
        flat = (rows * c.width,)
        out = graph.buffer(name, x.shape, "float32", role=role)
        scale = checkpoint.buffer(graph, weight, provider="webgpu")
        graph(
            "rms",
            dict(r=rows, c=c.width, eps=c.epsilon),
            x.view(flat),
            scale,
            out.view(flat),
        )
        return out

    def rotate(x, name, heads):
        flat = (rows * heads * c.head_dim,)
        normal = graph.buffer(name + ".normalized", flat, "float32")
        out = graph.buffer(name + ".rotary", x.shape, "float32")
        scale = checkpoint.buffer(graph, name + "_norm.weight", provider="webgpu")
        graph(
            "rms",
            dict(r=rows * heads, c=c.head_dim, eps=c.epsilon),
            x.view(flat),
            scale,
            normal,
        )
        graph(
            "rotary_compact",
            dict(
                r=rows,
                h=heads,
                d=c.head_dim,
                rotary=c.head_dim,
                capacity=context,
                stride=context,
                blocks=1,
                theta=c.rope_theta,
            ),
            normal,
            out,
            positions,
            rope,
            origin,
        )
        return out

    for layer in range(c.layers):
        prefix = f"blk.{layer}."
        normal = norm(x, prefix + "attn_norm.weight", prefix + "normalized")
        q_name, k_name = prefix + "attn_q", prefix + "attn_k"
        q, k, v = (
            _linear(
                graph,
                checkpoint,
                normal,
                name + ".weight",
                name + ".projected",
                integer=integer,
                signed_decode=signed_decode,
            )
            for name in (q_name, k_name, prefix + "attn_v")
        )
        rotary = None
        if rows == 1:
            wq, wk = (
                checkpoint.buffer(graph, name + "_norm.weight", provider="webgpu")
                for name in (q_name, k_name)
            )
            rotary = (wq, wk, rope, origin, c.rope_theta, c.epsilon)
        else:
            q, k = rotate(q, q_name, c.heads), rotate(k, k_name, c.kv_heads)
        attended = _attention(
            graph,
            q,
            k,
            v,
            positions,
            positions,
            active,
            prefix,
            heads=c.heads,
            kv_heads=c.kv_heads,
            context=context,
            persistent=True,
            rotary=rotary,
            splits=attention_splits,
        )
        x = residual_projection(
            attended, prefix + "attn_output.weight", x, prefix + "attention"
        )
        normal = norm(x, prefix + "ffn_norm.weight", prefix + "ffn_normalized")
        activated = _linear(
            graph,
            checkpoint,
            normal,
            prefix + "ffn_gate.weight",
            prefix + "activated",
            up=prefix + "ffn_up.weight",
            integer=integer,
            signed_decode=signed_decode,
        )
        x = residual_projection(
            activated, prefix + "ffn_down.weight", x, prefix + "ffn"
        )
    hidden = norm(
        x, "output_norm.weight", "hidden", role="scratch" if decode else "output"
    )
    if decode:
        _head(graph, checkpoint, hidden, signed_decode=signed_decode)
    return graph


def head(checkpoint, *, signed_decode=False):
    graph = Graph(contract="reference")
    x = graph.buffer("hidden", (1, checkpoint.config.width), "float32", role="input")
    _head(graph, checkpoint, x, signed_decode=signed_decode)
    return graph


def _head(graph, checkpoint, x, *, signed_decode):
    c = checkpoint.config
    active = graph.buffer("head_active", (1,), "int32", role="weight")
    logits = _linear(
        graph, checkpoint, x, checkpoint.head, "logits", signed_decode=signed_decode
    )
    token = graph.buffer("token", (1,), "int32", role="output")
    block = 1024
    parts = (c.vocab + block - 1) // block
    values = graph.buffer("argmax.values", (1, parts), "float32")
    indices = graph.buffer("argmax.indices", (1, parts), "int32")
    params = dict(r=1, c=c.vocab, block=block)
    graph("argmax_partial", params, logits.view((c.vocab,)), values, indices)
    graph("argmax_merge", params, values, indices, active, token)
