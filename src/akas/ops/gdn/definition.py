"""Contracts and implementation choices for gdn."""

from akas.kernels.contracts import OutputABI
from akas.kernels.contracts import Implementation
from akas.ops.gdn.validation import VALIDATIONS as VALIDATIONS


def _native_conv_outputs(call):
    if (
        call.parameters.get("full_checkpoints")
        or "checkpoints" in call.bindings
        or len(call.arguments) == 6
    ):
        indexes = (4, 5) if call.parameters.get("commit") is False else (3, 4, 5)
        names = (
            ("out", "checkpoints")
            if call.parameters.get("commit") is False
            else ("state", "out", "checkpoints")
        )
    else:
        indexes = (4,) if call.parameters.get("commit") is False else (3, 4)
        names = ("out",) if call.parameters.get("commit") is False else ("state", "out")
    return OutputABI(indexes, names)


def _native_scan_outputs(call):
    indexes, names = ((7,), ("out",))
    if call.parameters.get("commit", True):
        indexes, names = ((6, 7), ("state", "out"))
    if (
        call.parameters.get("full_checkpoints")
        or "checkpoints" in call.bindings
        or len(call.arguments) == 9
    ):
        indexes, names = ((*indexes, 8), (*names, "checkpoints"))
    return OutputABI(indexes, names)


def fp8_decode_gdn(ctx, block):
    prefix = block.mixer_prefix
    for name, dest in (("in_proj_qkv", "qkv"), ("in_proj_z", "z")):
        ctx.fp8(prefix + name, ctx.buffers["normal"], ctx.buffers[dest])
    for name in ("a", "b"):
        ctx.bf16(prefix + "in_proj_" + name, ctx.buffers["normal"], ctx.buffers[name])
    state, history = ctx.executor.states[block.index]
    ctx.emit(
        "gdn_conv",
        dict(r=ctx.rows),
        ctx.buffers["qkv"],
        ctx.weights[prefix + "conv1d.weight"],
        ctx.buffers["active"],
        history,
        ctx.buffers["conv"],
    )
    ctx.emit(
        "gdn_prepare",
        dict(r=ctx.rows),
        ctx.buffers["conv"],
        ctx.buffers["a"],
        ctx.buffers["b"],
        ctx.weights[prefix + "dt_bias"],
        ctx.weights[prefix + "A_log"],
        ctx.buffers["gdn_q"],
        ctx.buffers["gdn_k"],
        ctx.buffers["gdn_v"],
        ctx.buffers["gdn_decay"],
        ctx.buffers["gdn_beta"],
    )
    ctx.emit(
        "gdn_recurrent",
        dict(slots=ctx.rows, heads=32, key=128, value=128),
        ctx.buffers["gdn_q"],
        ctx.buffers["gdn_k"],
        ctx.buffers["gdn_v"],
        ctx.buffers["gdn_decay"],
        ctx.buffers["gdn_beta"],
        ctx.buffers["active"],
        state,
        ctx.buffers["gdn_out"],
    )
    ctx.emit(
        "gdn_norm",
        dict(r=ctx.rows, eps=ctx.executor.config.epsilon),
        ctx.buffers["gdn_out"],
        ctx.buffers["z"],
        ctx.weights[prefix + "norm.weight"],
        ctx.buffers["gdn_normal"],
    )
    ctx.fp8(prefix + "out_proj", ctx.buffers["gdn_normal"], ctx.buffers["mixed"])


def fp8_prefill_gdn(ctx, block):
    prefix = block.mixer_prefix
    for name, dest in (("in_proj_qkv", "qkv"), ("in_proj_z", "z")):
        ctx.fp8(prefix + name, "normal", dest)
    for name in ("a", "b"):
        ctx.bf16(prefix + "in_proj_" + name, "normal", name)
    state, history = ctx.model.states[block.index]
    ctx.small(
        "gdn_conv",
        x=ctx.buffers["qkv"],
        w=ctx.weights[prefix + "conv1d.weight"],
        lengths=ctx.buffers["lengths"],
        state=history,
        out=ctx.buffers["conv"],
    )
    ctx.emit(
        "gdn_prepare",
        dict(r=ctx.rows),
        qkv=ctx.buffers["conv"],
        a=ctx.buffers["a"],
        b=ctx.buffers["b"],
        dt=ctx.weights[prefix + "dt_bias"],
        A=ctx.weights[prefix + "A_log"],
        q=ctx.buffers["gdn_q"],
        k=ctx.buffers["gdn_k"],
        v=ctx.buffers["gdn_v"],
        g=ctx.buffers["gdn_decay"],
        beta=ctx.buffers["gdn_beta"],
    )
    ctx.small(
        "gdn_scan",
        q=ctx.buffers["gdn_q"],
        k=ctx.buffers["gdn_k"],
        v=ctx.buffers["gdn_v"],
        g=ctx.buffers["gdn_decay"],
        beta=ctx.buffers["gdn_beta"],
        lengths=ctx.buffers["lengths"],
        state=state,
        out=ctx.buffers["gdn_out"],
    )
    ctx.emit(
        "gdn_norm",
        dict(r=ctx.rows, eps=ctx.model.config.epsilon),
        x=ctx.buffers["gdn_out"],
        z=ctx.buffers["z"],
        w=ctx.weights[prefix + "norm.weight"],
        out=ctx.buffers["gdn_normal"],
    )
    ctx.fp8(prefix + "out_proj", "gdn_normal", "mixed")


def search_policies(graph, commit, *, provider, target):
    if (
        graph.contract == "block-fp8"
        and commit is not None
        and any(
            call.kind == "gdn_scan" and call.parameters.get("full_checkpoints")
            for call in graph.calls
        )
    ):
        return [{"commit": "recompute"}]
    return []


def lower(graph, commit, settings, *, provider, target):
    from dataclasses import replace
    from akas.ops.gdn.definition import journal_prefixes

    if set(settings) - {"commit"}:
        raise ValueError("unknown recurrent policy key")
    mode = settings.get("commit", "snapshots")
    if mode not in ("snapshots", "recompute"):
        raise ValueError("recurrent commit must be snapshots or recompute")
    if mode == "snapshots" or graph.contract != "block-fp8" or commit is None:
        return (graph, commit)
    scans = [call for call in graph.calls if call.kind in ("gdn_conv", "gdn_scan")]
    if not scans:
        return (graph, commit)
    if not all((call.parameters.get("full_checkpoints") for call in scans)):
        raise ValueError("recompute policy requires a non-mutating verification graph")
    journals = {call.bindings["checkpoints"] for call in scans}
    graph.calls = [
        replace(
            call,
            parameters={
                key: value
                for key, value in call.parameters.items()
                if key != "full_checkpoints"
            },
            bindings={
                key: value
                for key, value in call.bindings.items()
                if key != "checkpoints"
            },
        )
        if call in scans
        else call
        for call in graph.calls
    ]
    commit.calls = [call for call in commit.calls if call.kind != "restore"]
    later = commit.calls
    commit.calls = []
    parameters = scans[0].parameters
    lengths = graph.buffers[graph.request["inputs"]["lengths"]]
    journal_prefixes(
        graph,
        commit,
        slots=parameters["slots"],
        chunk=parameters["chunk"],
        lengths=lengths,
        mode="recompute",
    )
    commit.calls = [
        replace(
            call,
            parameters={
                key: value for key, value in call.parameters.items() if key != "commit"
            },
        )
        for call in commit.calls
    ] + later
    graph.buffers = {
        name: spec for name, spec in graph.buffers.items() if name not in journals
    }
    commit.buffers = graph.buffers
    return (graph, commit)


def separate_workspace(graph, rows, config, *, prefix=""):
    """Declare scratch for separately projected recurrent operands."""
    heads, key, value = (config.value_heads, config.key_dim, config.value_dim)
    channels = 2 * config.key_heads * key + heads * value
    specs = {
        "qkv": (rows, channels),
        "z": (rows, heads * value),
        "a": (rows, heads),
        "b": (rows, heads),
        "conv": (rows, channels),
        "gdn_q": (rows, heads, key),
        "gdn_k": (rows, heads, key),
        "gdn_v": (rows, heads, value),
        "gdn_out": (rows, heads, value),
        "gdn_decay": (rows, heads),
        "gdn_beta": (rows, heads),
        "gdn_normal": (rows, heads * value),
    }
    return {
        name: graph.buffer(
            prefix + name, shape, "bfloat16" if name == "gdn_normal" else "float32"
        )
        for name, shape in specs.items()
    }


def gated_delta(
    graph, x, weights, states, controls, *, config, prefix, verify=False, commit=None
):
    """Compose mapped recurrent mixing and its accepted-prefix replay journal."""
    from akas.ops.linear.definition import project

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
            **{"commit": False} if verify else {},
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
            **{"commit": False} if verify else {},
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


def journal_prefixes(graph, commit, *, slots, chunk, lengths, mode="snapshots"):
    """Keep native scan snapshots without mutating persistent state until accept.

    Every live prefix has a checkpoint, including the complete window. Zero
    accepted tokens and discarded verification leave recurrent state unchanged.
    """
    from dataclasses import replace

    if mode not in ("snapshots", "recompute"):
        raise ValueError("unknown recurrent commit policy")
    commit.buffers = graph.buffers
    calls = []
    for call in graph.calls:
        if call.kind not in ("gdn_conv", "gdn_scan"):
            calls.append(call)
            continue
        if mode == "recompute":
            from math import prod
            from akas.graph import Graph

            copies = Graph(contract=graph.contract, phase=graph.phase)
            copies.buffers = graph.buffers
            bindings = dict(call.bindings)
            for name in ("x", "q", "k", "v", "g", "beta"):
                if name not in bindings:
                    continue
                source = graph.buffers[bindings[name]]
                saved = graph.buffer(
                    "verify.operands." + bindings["state"] + "." + name,
                    source.shape,
                    source.dtype,
                    role="journal",
                )
                shape = (prod(source.shape),)
                source_view = graph.view(saved.name + ".source", source, shape)
                saved_view = graph.view(saved.name + ".flat", saved, shape)
                copies(
                    "copy_buffer",
                    dict(n=shape[0], dtype=source.dtype),
                    x=source_view,
                    out=saved_view,
                )
                bindings[name] = saved.name
            calls.extend(copies.calls)
            calls.append(
                replace(
                    call,
                    parameters={**call.parameters, "commit": False},
                    bindings=bindings,
                )
            )
            commit(
                call.kind,
                call.parameters,
                **{name: graph.buffers[value] for name, value in bindings.items()},
            )
            continue
        state = graph.buffers[call.bindings["state"]]
        saved = graph.buffer(
            "verify.saved." + state.name,
            (slots, chunk, *state.shape[1:]),
            state.dtype,
            role="journal",
        )
        parameters = {**call.parameters, "full_checkpoints": True, "commit": False}
        calls.append(
            replace(
                call,
                parameters=parameters,
                bindings={**call.bindings, "checkpoints": saved.name},
            )
        )
        commit(
            "restore",
            dict(
                slots=slots,
                chunk=chunk,
                shape=list(state.shape[1:]),
                full_checkpoints=True,
            ),
            checkpoints=saved,
            state=state,
            accepted=lengths,
            lengths=lengths,
        )
    graph.calls = calls


OUTPUT_ARGUMENTS = {
    "bf16": {
        "gdn_conv": lambda call: OutputABI(
            (5,) if call.parameters.get("commit") is False else (4, 5),
            ("out",) if call.parameters.get("commit") is False else ("state", "out"),
        ),
        "gdn_prepare": OutputABI((4, 5, 6, 7, 8), ("q", "k", "v", "g", "beta")),
        "gdn_scan": lambda call: OutputABI(
            (8,) if call.parameters.get("commit") is False else (7, 8),
            ("out",) if call.parameters.get("commit") is False else ("state", "out"),
        ),
        "gdn_norm": OutputABI((-1,), ("out",)),
    },
    "block-fp8": {
        "gdn_conv": _native_conv_outputs,
        "gdn_prepare": OutputABI((5, 6, 7, 8, 9), ("q", "k", "v", "g", "beta")),
        "gdn_scan": _native_scan_outputs,
        "gdn_recurrent": OutputABI((6, 7), ("state", "out")),
        "gdn_norm": OutputABI((-1,), ("out",)),
        "restore": OutputABI((1,), ("state",)),
    },
}
IMPLEMENTATIONS = (
    Implementation(
        "conv_commit",
        "akas.ops.gdn.kernel",
        "bf16",
        factory="conv_commit",
        name="gdn.verification",
    ),
    Implementation(
        "defer_accept",
        "akas.ops.gdn.kernel",
        "bf16",
        factory="defer_accept",
        name="gdn.verification",
    ),
    Implementation(
        "defer_clear",
        "akas.ops.gdn.kernel",
        "bf16",
        factory="defer_clear",
        name="gdn.verification",
    ),
    Implementation(
        "gdn_deferred",
        "akas.ops.gdn.kernel",
        "bf16",
        factory="gdn_deferred",
        name="gdn.verification",
    ),
    Implementation(
        "gdn_materialize",
        "akas.ops.gdn.kernel",
        "bf16",
        factory="gdn_materialize",
        name="gdn.verification",
    ),
    Implementation(
        "gdn_conv",
        "akas.ops.gdn.kernel",
        "bf16",
        factory="mapped_conv",
        name="gdn.cuda.bf16_batch",
    ),
    Implementation(
        "gdn_prepare",
        "akas.ops.gdn.kernel",
        "bf16",
        factory="mapped_prepare",
        name="gdn.cuda.bf16_batch",
    ),
    Implementation(
        "gdn_norm",
        "akas.ops.gdn.kernel",
        "bf16",
        factory="mapped_norm",
        name="gdn.cuda.bf16_batch",
    ),
    Implementation(
        "gdn_scan",
        "akas.ops.gdn.kernel",
        "bf16",
        space=(("tile", (32, 16, 64)),),
        semantic_keys=("slots", "chunk", "pool", "heads", "d", "commit"),
        factory="mapped_scan",
        name="gdn.cuda.bf16_batch",
    ),
    Implementation(
        "gdn_conv",
        "akas.ops.gdn.kernel",
        "block-fp8",
        phase="decode",
        factory="decode_conv",
        name="gdn.cuda.decode",
    ),
    Implementation(
        "gdn_recurrent",
        "akas.ops.gdn.kernel",
        "block-fp8",
        phase="decode",
        factory="decode_recurrent",
        name="gdn.cuda.decode",
    ),
    Implementation(
        "gdn_prepare",
        "akas.ops.gdn.kernel",
        "block-fp8",
        factory="decode_prepare",
        name="gdn.cuda.decode",
    ),
    Implementation(
        "gdn_norm",
        "akas.ops.gdn.kernel",
        "block-fp8",
        factory="decode_norm",
        name="gdn.cuda.decode",
    ),
    Implementation(
        "gdn_conv",
        "akas.ops.gdn.kernel",
        "block-fp8",
        phase="prefill",
        required=("slots", "chunk"),
        factory="prefill_conv",
        name="gdn.cuda.prefill",
    ),
    Implementation(
        "gdn_scan",
        "akas.ops.gdn.kernel",
        "block-fp8",
        phase="prefill",
        required=("slots", "chunk"),
        factory="prefill_scan",
        name="gdn.cuda.prefill",
    ),
    Implementation(
        "gdn_conv",
        "akas.ops.gdn.kernel",
        "block-fp8",
        phase="verification",
        required=("slots", "chunk"),
        factory="journal_conv",
        name="gdn.cuda.verification",
    ),
    Implementation(
        "gdn_scan",
        "akas.ops.gdn.kernel",
        "block-fp8",
        phase="verification",
        required=("slots", "chunk"),
        factory="journal_scan",
        name="gdn.cuda.verification",
    ),
    Implementation(
        "restore",
        "akas.ops.gdn.kernel",
        "block-fp8",
        phase="verification",
        required=("slots", "chunk"),
        factory="journal_restore",
        name="gdn.cuda.verification",
    ),
    Implementation(
        "gdn_conv",
        "akas.ops.gdn.kernel",
        "block-fp8",
        priority=30,
        required=("full_checkpoints",),
        factory="journal_conv",
        name="gdn.cuda.verification",
    ),
    Implementation(
        "gdn_scan",
        "akas.ops.gdn.kernel",
        "block-fp8",
        priority=30,
        required=("full_checkpoints",),
        factory="journal_scan",
        name="gdn.cuda.verification",
    ),
    Implementation(
        "gdn_conv",
        "akas.ops.gdn.kernel",
        "block-fp8",
        phase="commit",
        required=("slots", "chunk"),
        factory="prefill_conv",
        name="gdn.cuda.prefill",
    ),
    Implementation(
        "gdn_scan",
        "akas.ops.gdn.kernel",
        "block-fp8",
        phase="commit",
        required=("slots", "chunk"),
        factory="prefill_scan",
        name="gdn.cuda.prefill",
    ),
    Implementation(
        "restore",
        "akas.ops.gdn.kernel",
        "block-fp8",
        priority=30,
        required=("full_checkpoints",),
        factory="restore",
        name="gdn.restore",
    ),
)
