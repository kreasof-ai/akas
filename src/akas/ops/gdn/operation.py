"""Compatible kernels and search spaces owned by this operation."""

from akas.ops.registry import variants

IMPLEMENTATIONS = (
    *variants(
        (
            "conv_commit",
            "defer_accept",
            "defer_clear",
            "gdn_deferred",
            "gdn_materialize",
        ),
        "gdn.verification",
        "bf16",
    ),
    *variants(
        ("gdn_conv", "gdn_prepare", "gdn_norm"),
        "gdn.cuda.bf16_batch",
        "bf16",
    ),
    *variants(
        ("gdn_scan",),
        "gdn.cuda.bf16_batch",
        "bf16",
        semantic_keys=("slots", "chunk", "pool", "heads", "d", "commit"),
        space=(("tile", (32, 16, 64)),),
    ),
    *variants(
        ("gdn_conv", "gdn_recurrent"),
        "gdn.cuda.decode",
        "block-fp8",
        phase="decode",
    ),
    *variants(("gdn_prepare", "gdn_norm"), "gdn.cuda.decode", "block-fp8"),
    *variants(
        ("gdn_conv", "gdn_scan"),
        "gdn.cuda.prefill",
        "block-fp8",
        phase="prefill",
        required=("slots", "chunk"),
    ),
    *variants(
        ("gdn_conv", "gdn_scan", "restore"),
        "gdn.cuda.verification",
        "block-fp8",
        phase="verification",
        required=("slots", "chunk"),
    ),
)


from akas.ops.gdn.validation import VALIDATIONS


def separate_workspace(graph, rows, config, *, prefix=""):
    """Declare scratch for separately projected recurrent operands."""
    heads, key, value = config.value_heads, config.key_dim, config.value_dim
    channels = 2 * config.key_heads * key + heads * value
    specs = {"qkv": (rows, channels), "z": (rows, heads * value),
             "a": (rows, heads), "b": (rows, heads), "conv": (rows, channels),
             "gdn_q": (rows, heads, key), "gdn_k": (rows, heads, key),
             "gdn_v": (rows, heads, value), "gdn_out": (rows, heads, value),
             "gdn_decay": (rows, heads), "gdn_beta": (rows, heads),
             "gdn_normal": (rows, heads * value)}
    return {name: graph.buffer(prefix + name, shape,
                              "bfloat16" if name == "gdn_normal" else "float32")
            for name, shape in specs.items()}


def gated_delta(
    graph, x, weights, states, controls, *, config, prefix, verify=False, commit=None
):
    """Compose mapped recurrent mixing and its accepted-prefix replay journal."""
    from akas.ops.linear.operation import project

    rows, width = x.shape
    common = controls["parameters"]
    channels = config.qkv_width
    projection = channels + config.value_width + 2 * config.value_heads
    projected = project(
        graph,
        x,
        weights[prefix + "input.weight"],
        width=width,
        name=prefix + "projection",
        role="journal" if verify else "scratch",
    )
    conv = graph.buffer(prefix + "convolved", (rows, channels), "float32")
    saved = {}
    for name in ("q", "k", "v", "g", "beta"):
        shape = (
            (rows, config.value_heads, config.key_dim)
            if name in ("q", "k", "v")
            else (rows, config.value_heads)
        )
        saved[name] = graph.buffer(
            prefix + name, shape, "float32", role="journal" if verify else "scratch"
        )
    recurrent = graph.buffer(
        prefix + "recurrent", (rows, config.value_heads, config.key_dim), "float32"
    )
    normalized = graph.buffer(
        prefix + "normalized", (rows, config.value_width), "bfloat16"
    )
    matrix, history = states
    conv_args = (
        projected,
        weights[prefix + "conv1d.weight"],
        controls["mapping"],
        controls["lengths"],
        history,
        conv,
    )
    scan_args = (
        *(saved[name] for name in ("q", "k", "v", "g", "beta")),
        controls["mapping"],
        controls["lengths"],
        matrix,
        recurrent,
    )
    graph(
        "gdn_conv",
        dict(
            common,
            channels=channels,
            projection=projection,
            **({"commit": False} if verify else {}),
        ),
        *conv_args,
    )
    graph(
        "gdn_prepare",
        dict(
            common,
            heads=config.value_heads,
            key_heads=config.key_heads,
            d=config.key_dim,
            channels=channels,
            projection=projection,
        ),
        conv,
        projected,
        weights[prefix + "dt_bias"],
        weights[prefix + "A_log"],
        *(saved[name] for name in ("q", "k", "v", "g", "beta")),
    )
    graph(
        "gdn_scan",
        dict(
            common,
            heads=config.value_heads,
            d=config.key_dim,
            tile=32,
            **({"commit": False} if verify else {}),
        ),
        *scan_args,
    )
    if verify:
        if commit is None:
            raise ValueError("verification needs a commit graph")
        commit.buffers = graph.buffers
        commit(
            "gdn_conv",
            dict(common, channels=channels, projection=projection),
            *conv_args,
        )
        commit(
            "gdn_scan",
            dict(common, heads=config.value_heads, d=config.key_dim, tile=32),
            *scan_args,
        )
    graph(
        "gdn_norm",
        dict(
            common,
            heads=config.value_heads,
            d=config.value_dim,
            channels=channels,
            projection=projection,
            eps=config.epsilon,
        ),
        recurrent,
        projected,
        weights[prefix + "norm.weight"],
        normalized,
    )
    return project(
        graph,
        normalized,
        weights[prefix + "out_proj.weight"],
        width=width,
        name=prefix + "output",
    )
