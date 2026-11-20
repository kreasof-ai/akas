"""Pure graph composition for the fixed-shape offline Voice Isolator."""

from math import prod

from akas.graph import Graph


def _buffer(graph, name, shape, dtype="float32", role="scratch"):
    return graph.buffer(name, shape, dtype, role=role)


def _constant(graph, name, shape, dtype="float32"):
    if name not in graph.buffers:
        _buffer(graph, name, shape, dtype, role="weight")
    return graph.buffers[name]


def _weight(graph, checkpoint, name):
    return checkpoint.buffer(graph, name, provider="webgpu")


def _flat(buffer):
    return buffer.view((prod(buffer.shape),))


def _add(graph, x, y, name, *, role="scratch"):
    out = _buffer(graph, name, x.shape, role=role)
    graph(
        "add", dict(r=prod(x.shape[:-1]), c=x.shape[-1]), _flat(x), _flat(y), _flat(out)
    )
    return out


def _binary(graph, operation, x, y, name):
    out = _buffer(graph, name, x.shape)
    graph(
        "elementwise_binary",
        dict(n=prod(x.shape), operation=operation),
        _flat(x),
        _flat(y),
        _flat(out),
    )
    return out


def _linear(graph, checkpoint, x, weight, name, *, bias=False, activation="identity"):
    rows, depth = x.shape
    columns, expected = checkpoint.tensors[weight].shape
    if depth != expected:
        raise ValueError(f"Voice Isolator projection depth mismatch: {weight}")
    weights = _weight(graph, checkpoint, weight)
    if bias:
        offset = _weight(graph, checkpoint, name + ".bias" if bias is True else bias)
        out = _buffer(graph, name, (rows, columns))
        parameters = dict(r=rows, k=depth, o=columns, activation=activation)
        if weights.dtype == "float16":
            parameters["weight_dtype"] = weights.dtype
        graph(
            "linear_bias_activation", parameters, _flat(x), weights, offset, _flat(out)
        )
        return out
    out = _buffer(graph, name + ".projected", (rows, columns))
    if weights.dtype == "float16":
        graph(
            "linear_w16a16",
            dict(r=rows, k=depth, o=columns, weight_dtype=weights.dtype),
            _flat(x),
            weights,
            _flat(out),
        )
        return out
    graph(
        "linear",
        dict(r=rows, k=depth, o=columns, ggml=True, type=0, tiled_f32=True),
        _flat(x),
        weights,
        _flat(out),
    )
    return out


def _norm(graph, checkpoint, x, prefix):
    rows, width = x.shape
    out = _buffer(graph, prefix + ".normalized", x.shape)
    graph(
        "affine_layer_norm",
        dict(r=rows, c=width, eps=1e-5),
        x,
        _weight(graph, checkpoint, prefix + ".weight"),
        _weight(graph, checkpoint, prefix + ".bias"),
        out,
    )
    return out


def _activation(graph, kind, x, name):
    out = _buffer(graph, name, x.shape)
    graph(kind, dict(r=x.shape[0], c=x.shape[1]), x, out)
    return out


def _transpose(graph, x, name):
    bands, frames, width = x.shape
    out = _buffer(graph, name, (frames, bands, width))
    graph("token_transpose", dict(bands=bands, frames=frames, width=width), x, out)
    return out


def _attention(graph, checkpoint, x, prefix, *, heads, frames, rotary):
    rows, width = x.shape
    depth = width // heads
    weight = prefix + (".qkv.weight" if rotary else ".in_proj_weight")
    bias = prefix + (".qkv.bias" if rotary else ".in_proj_bias")
    qkv = _linear(graph, checkpoint, x, weight, prefix + ".qkv", bias=bias)
    query, key, value = (
        _buffer(graph, prefix + "." + name, (rows, width)) for name in ("q", "k", "v")
    )
    graph(
        "qkv_split",
        dict(r=rows, h=heads, kh=heads, d=depth),
        _flat(qkv),
        _flat(query),
        _flat(key),
        value,
    )
    if rotary:
        positions = _constant(
            graph, "position." + prefix + f"@{graph.sample_count}", (rows,), "int32"
        )
        for name, tensor in (("q", query), ("k", key)):
            rotated = _buffer(graph, prefix + "." + name + ".rotary", tensor.shape)
            graph(
                "rotary_compact",
                dict(
                    r=rows,
                    h=heads,
                    d=depth,
                    rotary=depth,
                    capacity=frames,
                    stride=512,
                    blocks=2,
                    theta=10000.0,
                ),
                _flat(tensor),
                rotated,
                positions,
                _constant(graph, "rope_fine", (512, depth)),
                _constant(graph, "rope_coarse", (2, depth)),
            )
            if name == "q":
                query = rotated
            else:
                key = rotated
    attended = _buffer(graph, prefix + ".attended", (rows, width))
    graph(
        "full_attention",
        dict(r=rows, h=heads, d=depth, chunk=frames),
        query,
        key,
        value,
        attended,
    )
    return attended, value


def _transformer_axis(graph, checkpoint, x, prefix, *, frames, rotary, exclusive):
    rows, width = x.shape
    residual = x
    normal = _norm(graph, checkpoint, x, prefix + ".norm1")
    attention_prefix = prefix + (".attn" if rotary else ".self_attn")
    attended, values = _attention(
        graph,
        checkpoint,
        normal,
        attention_prefix,
        heads=checkpoint.config.num_heads,
        frames=frames,
        rotary=rotary,
    )
    if exclusive:
        head_width = width // checkpoint.config.num_heads
        out = _buffer(
            graph,
            attention_prefix + ".exclusive",
            (rows, checkpoint.config.num_heads, head_width),
        )
        graph(
            "exclusive_attention",
            dict(r=rows, h=checkpoint.config.num_heads, d=head_width),
            attended.view(out.shape),
            values.view(out.shape),
            out,
        )
        attended = out.view((rows, width))
    projected = _linear(
        graph,
        checkpoint,
        attended,
        attention_prefix + (".proj.weight" if rotary else ".out_proj.weight"),
        attention_prefix + (".proj" if rotary else ".out_proj"),
        bias=True,
    )
    x = _add(graph, residual, projected, prefix + ".attention_residual")
    normal = _norm(graph, checkpoint, x, prefix + ".norm2")
    ffn_prefix = prefix + (".ffn" if rotary else "")
    first = ffn_prefix + (".0" if rotary else ".linear1")
    second = ffn_prefix + (".3" if rotary else ".linear2")
    activated = _linear(
        graph,
        checkpoint,
        normal,
        first + ".weight",
        first,
        bias=True,
        activation="gelu",
    )
    activated = _linear(
        graph, checkpoint, activated, second + ".weight", second, bias=True
    )
    return _add(graph, x, activated, prefix + ".ffn_residual")


def _transformer(graph, checkpoint, x, index):
    c = checkpoint.config
    bands, frames, width = x.shape
    prefix = f"blocks.{index}"
    time = _transformer_axis(
        graph,
        checkpoint,
        x.view((bands * frames, width)),
        prefix + ".time_transformer",
        frames=frames,
        rotary=True,
        exclusive=True,
    )
    frequency = _transpose(graph, time.view(x.shape), prefix + ".time_to_band")
    frequency = _transformer_axis(
        graph,
        checkpoint,
        frequency.view((frames * bands, width)),
        prefix + ".band_transformer",
        frames=bands,
        rotary=False,
        exclusive=False,
    )
    return _transpose(
        graph, frequency.view((frames, bands, width)), prefix + ".band_to_time"
    )


def _mamba_branch(graph, checkpoint, x, prefix, *, frames):
    c = checkpoint.config
    rows, width = x.shape
    projected = _linear(
        graph, checkpoint, x, prefix + ".in_proj.weight", prefix + ".in_proj"
    )
    raw = _buffer(graph, prefix + ".u_raw", (rows, c.inner))
    gate = _buffer(graph, prefix + ".gate", raw.shape)
    graph(
        "channel_slice",
        dict(r=rows, c=2 * c.inner, start=0, width=c.inner),
        projected,
        raw,
    )
    graph(
        "channel_slice",
        dict(r=rows, c=2 * c.inner, start=c.inner, width=c.inner),
        projected,
        gate,
    )
    convolved = _buffer(graph, prefix + ".convolved", raw.shape)
    sequence_shape = (rows // frames, frames, c.inner)
    graph(
        "causal_depthwise_sequences",
        dict(sequences=rows // frames, frames=frames, c=c.inner, taps=c.mamba_conv),
        raw.view(sequence_shape),
        _weight(graph, checkpoint, prefix + ".conv1d.weight").view(
            (c.inner, c.mamba_conv)
        ),
        _weight(graph, checkpoint, prefix + ".conv1d.bias"),
        convolved.view(sequence_shape),
    )
    activated = _activation(graph, "silu", convolved, prefix + ".silu")
    coefficients = _linear(
        graph, checkpoint, activated, prefix + ".x_proj.weight", prefix + ".x_proj"
    )
    dt = _buffer(graph, prefix + ".dt_raw", (rows, c.dt_rank))
    b = _buffer(graph, prefix + ".b", (rows, c.mamba_state))
    cc = _buffer(graph, prefix + ".c", b.shape)
    graph(
        "channel_slice",
        dict(r=rows, c=c.dt_rank + 2 * c.mamba_state, start=0, width=c.dt_rank),
        coefficients,
        dt,
    )
    graph(
        "channel_slice",
        dict(
            r=rows,
            c=c.dt_rank + 2 * c.mamba_state,
            start=c.dt_rank,
            width=c.mamba_state,
        ),
        coefficients,
        b,
    )
    graph(
        "channel_slice",
        dict(
            r=rows,
            c=c.dt_rank + 2 * c.mamba_state,
            start=c.dt_rank + c.mamba_state,
            width=c.mamba_state,
        ),
        coefficients,
        cc,
    )
    delta = _linear(
        graph,
        checkpoint,
        dt,
        prefix + ".dt_proj.weight",
        prefix + ".dt_proj",
        bias=True,
        activation="softplus",
    )
    out = _buffer(graph, prefix + ".scan", activated.shape)
    graph(
        "selective_scan",
        dict(sequences=rows // frames, frames=frames, c=c.inner, s=c.mamba_state),
        activated,
        delta,
        _constant(graph, "decay." + prefix + ".A_log", (c.inner, c.mamba_state)),
        b,
        cc,
        _weight(graph, checkpoint, prefix + ".D"),
        gate,
        out,
    )
    return _linear(
        graph, checkpoint, out, prefix + ".out_proj.weight", prefix + ".out_proj"
    )


def _mamba_direction(graph, checkpoint, x, prefix, direction, *, frames):
    if direction == "forward_blocks":
        return _mamba_branch(
            graph, checkpoint, x, prefix + ".forward_blocks", frames=frames
        )
    sequences, frames = x.shape[0] // frames, frames
    reverse = _buffer(
        graph, prefix + ".input_reversed", (sequences, frames, x.shape[1])
    )
    graph(
        "mamba_reverse",
        dict(sequences=sequences, frames=frames, c=x.shape[1]),
        x.view(reverse.shape),
        reverse,
    )
    out = _mamba_branch(
        graph,
        checkpoint,
        reverse.view(x.shape),
        prefix + ".backward_blocks",
        frames=frames,
    )
    restored = _buffer(graph, prefix + ".backward", reverse.shape)
    graph(
        "mamba_reverse",
        dict(sequences=sequences, frames=frames, c=x.shape[1]),
        out.view(reverse.shape),
        restored,
    )
    return restored.view(x.shape)


def _mamba_axis(graph, checkpoint, x, prefix, *, frames):
    forward = _mamba_direction(
        graph, checkpoint, x, prefix, "forward_blocks", frames=frames
    )
    backward = _mamba_direction(
        graph, checkpoint, x, prefix, "backward_blocks", frames=frames
    )
    joined = _buffer(graph, prefix + ".joined", (x.shape[0], 2 * x.shape[1]))
    graph("mamba_join", dict(r=x.shape[0], c=x.shape[1]), x, forward, backward, joined)
    projected = _linear(
        graph,
        checkpoint,
        joined,
        prefix + ".output_proj.weight",
        prefix + ".output_proj",
        bias=True,
    )
    normalized = _norm(graph, checkpoint, projected, prefix + ".norm")
    return _add(graph, x, normalized, prefix + ".residual")


def _mamba(graph, checkpoint, x, index):
    bands, frames, width = x.shape
    prefix = f"blocks.{index}"
    time = _mamba_axis(
        graph,
        checkpoint,
        x.view((bands * frames, width)),
        prefix + ".time_mamba",
        frames=frames,
    )
    frequency = _transpose(graph, time.view(x.shape), prefix + ".time_to_band")
    frequency = _mamba_axis(
        graph,
        checkpoint,
        frequency.view((frames * bands, width)),
        prefix + ".freq_mamba",
        frames=bands,
    )
    return _transpose(
        graph, frequency.view((frames, bands, width)), prefix + ".band_to_time"
    )


def _encode(graph, checkpoint, spectrum, features):
    c = checkpoint.config
    frames = spectrum.shape[0]
    for index, (start, end) in enumerate(c.edges):
        band = _buffer(graph, f"encoder.{index}.bins", (frames, 2 * (end - start)))
        graph(
            "spectrum_band",
            dict(frames=frames, bins=c.bins, start=start, end=end),
            spectrum,
            band,
        )
        normal = _norm(graph, checkpoint, band, f"encoders.{index}.0")
        projected = _linear(
            graph,
            checkpoint,
            normal,
            f"encoders.{index}.1.weight",
            f"encoders.{index}.1",
            bias=True,
        )
        embedding = _constant(graph, f"band_embedding.{index}", (c.width,))
        graph(
            "token_scatter",
            dict(bands=len(c.edges), frames=frames, width=c.width, band=index),
            projected,
            embedding,
            features,
        )


def _decode(graph, checkpoint, features):
    c = checkpoint.config
    bands, frames, width = features.shape
    scale = _constant(graph, "decoder_condition.scale", (width,))
    shift = _constant(graph, "decoder_condition.shift", (width,))
    conditioned = _buffer(graph, "decoder.conditioned", features.shape)
    graph(
        "channel_affine",
        dict(r=frames * bands, c=width),
        features.view((frames * bands, width)),
        scale,
        shift,
        conditioned.view((frames * bands, width)),
    )
    raw = _buffer(graph, "decoder.raw", (frames, c.bins, 4))
    for index, (start, end) in enumerate(c.edges):
        band = _buffer(graph, f"decoder.{index}.input", (frames, width))
        graph(
            "band_gather",
            dict(bands=bands, frames=frames, width=width, band=index),
            conditioned,
            band,
        )
        prefix = f"denoise_decoder.heads.{index}"
        normal = _norm(graph, checkpoint, band, prefix + ".0")
        hidden = _linear(
            graph,
            checkpoint,
            normal,
            prefix + ".1.weight",
            prefix + ".1",
            bias=True,
            activation="gelu",
        )
        head = _linear(
            graph, checkpoint, hidden, prefix + ".3.weight", prefix + ".3", bias=True
        )
        graph(
            "band_scatter",
            dict(frames=frames, bins=c.bins, start=start, end=end),
            head.view((frames, end - start, 4)),
            raw,
        )
    return raw


def enhance(checkpoint, samples):
    if type(samples) is not int or not 1 <= samples <= checkpoint.config.chunk_samples:
        raise ValueError("Voice Isolator enhancement needs 1 to 8 seconds per graph")
    c = checkpoint.config
    frames = c.frames(samples)
    graph = Graph(contract="reference")
    graph.sample_count = samples
    audio = _buffer(graph, "audio", (samples,), role="input")
    window = _constant(graph, "window", (c.n_fft,))
    spectrum = _buffer(graph, "spectrum", (frames, c.bins, 2))
    graph(
        "stft",
        dict(
            samples=samples,
            fft=c.n_fft,
            hop=c.hop_length,
            frames=frames,
            bins=c.bins,
            reflect=samples > c.n_fft // 2,
        ),
        audio,
        window,
        spectrum,
    )
    features = _buffer(graph, "features", (len(c.edges), frames, c.width))
    _encode(graph, checkpoint, spectrum, features)
    for index, kind in enumerate(c.block_types):
        features = (
            _transformer(graph, checkpoint, features, index)
            if kind == "transformer"
            else _mamba(graph, checkpoint, features, index)
        )
    raw = _decode(graph, checkpoint, features)
    corrected = _buffer(graph, "correction", spectrum.shape)
    graph(
        "complex_correction",
        dict(frames=frames, bins=c.bins, bound=c.correction_bound),
        raw,
        spectrum,
        corrected,
    )
    inverse = _buffer(graph, "inverse_frames", (frames, c.n_fft))
    graph(
        "istft_matrix",
        dict(frames=frames, bins=c.bins, fft=c.n_fft),
        corrected,
        _constant(graph, f"inverse_basis@{samples}", (c.n_fft, 2 * c.bins)),
        inverse,
    )
    output = _buffer(graph, "enhanced_audio", (samples,), role="output")
    graph(
        "istft_overlap",
        dict(samples=samples, fft=c.n_fft, hop=c.hop_length, frames=frames),
        inverse,
        audio,
        window,
        output,
    )
    return graph
