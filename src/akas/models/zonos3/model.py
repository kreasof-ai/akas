"""Compose and execute the ZONOS3-Mini speech phases."""

from contextlib import ExitStack
from dataclasses import dataclass
from math import prod
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from akas.artifacts.read import read_forward, read_forward_set
from akas.execution.forward import Forward
from akas.graph import Graph
from akas.memory.transfer import write_many
from akas.models.zonos3.config import SILENCE, SILENCE_FRAMES
from akas.models.zonos3.weights import (
    CODEBOOK_EMBEDDING,
    CODEBOOK_HEAD,
    TOKEN_EMBEDDING,
    Zonos3MiniCheckpoint,
)


def compose(checkpoint, *, phase, rows=1, provider="webgpu", context=1024):
    """Declare a compiled forward stage using the GGUF pair's native storage."""
    if type(rows) is not int or rows < 1 or provider not in ("cuda", "webgpu"):
        raise ValueError("ZONOS3 requires positive row counts and CUDA or WebGPU")
    if phase == "backbone":
        return compose_backbone(
            checkpoint, rows=rows, context=context, provider=provider
        )
    if phase == "codec":
        return compose_codec(checkpoint, frames=rows, provider=provider)
    config = checkpoint.config
    width = config.width
    code_count = config.codebooks * config.code_vocab
    graph = Graph(contract="reference")

    if phase in ("text_embedding", "audio_feedback"):
        is_audio = phase == "audio_feedback"
        count = rows * config.codebooks if is_audio else rows
        table = CODEBOOK_EMBEDDING if is_audio else TOKEN_EMBEDDING
        ids = graph.buffer("ids", (count,), "int32", role="input")
        weight = checkpoint.graph_weight(graph, table, provider=provider)
        embedded = graph.buffer("embedding", (count * width,), "float32", role="output")
        params = dict(r=count, c=width, v=code_count if is_audio else config.vocab)
        if not is_audio:
            params.update(ggml=True, type=8)
        graph("embedding", params, ids, weight, embedded)
    elif phase in ("gdn_input", "audio_head"):
        is_audio = phase == "audio_head"
        if not is_audio and not config.recurrent[0]:
            raise ValueError("ZONOS3's first layer is not recurrent")
        name = CODEBOOK_HEAD if is_audio else "blk.0.attn_qkv.weight"
        columns = code_count if is_audio else 3 * config.delta_width
        x = graph.buffer("hidden", (rows * width,), "float32", role="input")
        weight = checkpoint.graph_weight(graph, name, provider=provider)
        out = graph.buffer("projection", (rows * columns,), "float32", role="output")
        params = dict(r=rows, k=width, o=columns)
        if is_audio:
            params["optimized"] = True
        else:
            params.update(ggml=True, type=8, optimized=True)
        graph("linear", params, x, weight, out)
    else:
        raise ValueError(f"unknown ZONOS3 forward phase: {phase}")
    return graph


def compose_backbone(
    checkpoint,
    *,
    rows=None,
    slots=None,
    chunk=None,
    pool=None,
    context=1024,
    provider="webgpu",
    include_head=False,
    embed_frames=False,
    sample=False,
):
    from akas.ops.gdn.definition import gated_delta
    from akas.ops.linear.definition import project
    from akas.ops.moe.definition import routed_moe
    from akas.quantization.weights import Weight

    c = checkpoint.config
    if slots is None:
        if type(rows) is not int or rows < 1:
            raise ValueError("ZONOS3 backbone requires positive row counts")
        slots, chunk, pool = 1, rows, 1 if pool is None else pool
    else:
        chunk = 1 if chunk is None else chunk
        pool = slots if pool is None else pool
        if (
            any(type(v) is not int or v < 1 for v in (slots, chunk, pool))
            or slots > pool
        ):
            raise ValueError("invalid ZONOS3 multi-slot backbone geometry")
        rows = slots * chunk
    if (
        provider not in ("cuda", "webgpu")
        or type(context) is not int
        or not 1 <= chunk <= context <= c.max_context
    ):
        raise ValueError(
            "ZONOS3 backbone requires CUDA or WebGPU and a valid request context"
        )
    if c.conv_width != 4 or c.delta_dim % 32:
        raise ValueError("unsupported ZONOS3 recurrent geometry")
    if sample and not include_head:
        raise ValueError("ZONOS3 codebook sampling requires include_head=True")
    g = Graph(contract="reference")
    x = g.buffer(
        "embedding",
        (rows, c.width),
        "float32",
        role="scratch" if embed_frames else "input",
    )
    positions = g.buffer("positions", (rows,), "int32", role="input")
    lengths = g.buffer("lengths", (slots,), "int32", role="input")
    mapping = g.buffer("mapping", (slots,), "int32", role="input")
    active = g.buffer("active", (rows,), "int32", role="input")
    rotary = g.buffer("rope", (context, c.head_dim), "float32", role="weight")
    unit = g.buffer("unit.weight", (c.width,), "float32", role="weight")
    head_unit = g.buffer("head_unit.weight", (c.head_dim,), "float32", role="weight")
    sources = 2 * len(c.recurrent) + 1
    bank = g.buffer("residual_bank", (sources, rows, c.width), "float32")

    def weight(name):
        if name not in g.buffers:
            checkpoint.graph_weight(g, name, provider=provider)
        return g.buffer_at(name)

    if embed_frames:
        cond_rows = sum(b + 1 for b in c.condition_buckets)
        frames = g.buffer("frames", (rows, c.codebooks + 1), "int32", role="input")
        g(
            "zonos3_frame_embed",
            dict(
                r=rows,
                c=c.width,
                codebooks=c.codebooks,
                code_vocab=c.code_vocab,
                text_vocab=c.vocab,
                cond_rows=cond_rows,
            ),
            frames,
            weight("a.gen.code.out_embd.bf16"),
            weight(TOKEN_EMBEDDING),
            weight("cond_embd.weight").view((cond_rows, c.width)),
            x,
        )

    def encoded(names, shape, *, store=None):
        tensors = (checkpoint.backbone if store is None else store).tensors
        return Weight(
            shape,
            tuple(weight(name) for name in names),
            tuple(
                dict(
                    format="ggml",
                    rows=prod(tensors[name].shape[:-1]),
                    options={"encoding": tensors[name].type},
                )
                for name in names
            ),
        )

    def norm(x, weights, name, *, role="scratch"):
        out = g.buffer(name, x.shape, "float32", role=role)
        flat = (rows * c.width,)
        g(
            "rms",
            dict(r=rows, c=c.width, eps=c.epsilon),
            x.view(flat),
            weights,
            out.view(flat),
        )
        return out

    def residual(source, index, name, query):
        out = g.buffer(name, source.shape, "float32")
        g(
            "attention_residual",
            dict(r=rows, c=c.width, capacity=sources, index=index, eps=c.epsilon),
            bank,
            source,
            query,
            out,
        )
        return out

    def linear(x, name, out):
        shape = checkpoint.backbone.tensors[name].shape
        return project(g, x, encoded((name,), shape), width=c.width, name=out)

    def delta(x, prefix):
        heads, d, width = c.delta_heads, c.delta_dim, c.delta_width
        channels = 3 * width
        shape = (channels + width + 2 * heads, c.width)
        joined = (
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_alpha.weight",
            "ssm_beta.weight",
        )
        input_names = (
            (prefix + "input.weight",)
            if (prefix + "input.weight") in checkpoint.backbone.tensors
            else tuple(prefix + name for name in joined)
        )
        conv = g.view(
            prefix + "conv", weight(prefix + "ssm_conv1d.weight"), (channels, 1, 4)
        )
        log_decay = g.buffer(prefix + "ssm_a_log", (heads,), "float32", role="weight")
        w = {
            prefix + "input.weight": encoded(input_names, shape),
            prefix + "conv1d.weight": conv,
            prefix + "dt_bias": weight(prefix + "ssm_dt.bias"),
            prefix + "A_log": log_decay,
            prefix + "norm.weight": weight(prefix + "ssm_norm.weight"),
            prefix + "out_proj.weight": encoded(
                (prefix + "ssm_out.weight",), (c.width, width)
            ),
        }
        cfg = SimpleNamespace(
            qkv_width=channels,
            value_width=width,
            value_heads=heads,
            key_heads=heads,
            value_dim=d,
            key_dim=d,
            epsilon=c.epsilon,
        )
        state = g.buffer(prefix + "delta", (pool, heads, d, d), "float32", role="state")
        history = g.buffer(
            prefix + "conv_state", (pool, channels, 3), "float32", role="state"
        )
        controls = dict(
            mapping=mapping,
            lengths=lengths,
            parameters=dict(slots=slots, chunk=chunk, pool=pool, dtype="float32"),
        )
        return gated_delta(
            g,
            x,
            w,
            (state, history),
            controls,
            config=cfg,
            prefix=prefix,
            dtype="float32",
        )

    def attention(x, prefix):
        h, kh, d = c.heads, c.kv_heads, c.head_dim
        scale = weight(prefix + "attn_q_norm.weight")
        q, k = (
            g.buffer(prefix + tag + "_rot", (rows, heads * d), "float32")
            for tag, heads in (("q", h), ("k", kh))
        )
        for tag, heads, out in (("q", h, q), ("k", kh, k)):
            projected = linear(x, prefix + f"attn_{tag}.weight", prefix + tag)
            g(
                "head_rms_rotary",
                dict(
                    r=rows,
                    h=heads,
                    d=d,
                    capacity=context,
                    scale_heads=h if tag == "q" else 1,
                    eps=1e-6,
                ),
                projected,
                scale if tag == "q" else head_unit,
                rotary,
                positions,
                out,
            )
        v = linear(x, prefix + "attn_v.weight", prefix + "v")
        shape = (pool, kh, context, d)
        kc, vc = (
            g.buffer(prefix + name, shape, "float16", role="state")
            for name in ("kc", "vc")
        )
        params = dict(r=rows, h=h, kh=kh, d=d, capacity=context, chunk=chunk, window=0)
        g("window_cache", params, k, v, kc, vc, positions, active)
        attended = g.buffer(prefix + "attended", (rows * h * d,), "float32")
        g("causal_grouped", params, q, kc, vc, positions, active, attended)
        gated = g.buffer(prefix + "gated", (rows * h * d,), "float32")
        gate = linear(x, prefix + "attn_gate.weight", prefix + "gate")
        g("head_gate", dict(r=rows, h=h, d=d), attended, gate.view((rows * h,)), gated)
        return linear(
            gated.view((rows, h * d)),
            prefix + "attn_output.weight",
            prefix + "attention",
        )

    def feed_forward(x, prefix, index):
        if index < c.dense_layers:
            gate, up = (
                linear(x, prefix + f"ffn_{tag}.weight", prefix + f"ffn_{tag}")
                for tag in ("gate", "up")
            )
            inner = g.buffer(prefix + "swiglu", (rows, c.dense_width), "float32")
            flat = (rows * c.dense_width,)
            g(
                "swiglu",
                dict(r=rows, c=c.dense_width),
                gate.view(flat),
                up.view(flat),
                inner.view(flat),
            )
            return linear(inner, prefix + "ffn_down.weight", prefix + "dense")
        route_scale = (
            weight(prefix + "ffn_route_norm.weight") if c.router_norm else unit
        )
        route_input = norm(x, route_scale, prefix + "route_norm")
        logits = linear(route_input, prefix + "ffn_gate_inp.weight", prefix + "logits")
        latent = linear(x, prefix + "ffn_latent_down.weight", prefix + "latent")
        names = {
            tag: prefix + f"ffn_{tag}_exps.weight" for tag in ("gate", "up", "down")
        }
        experts = {
            tag: encoded((name,), checkpoint.backbone.tensors[name].shape)
            for tag, name in names.items()
        }
        mixed = routed_moe(
            g,
            latent,
            logits,
            weight(prefix + "exp_probs_b.bias"),
            experts,
            top=c.experts_per_row,
            active=active,
            prefix=prefix + "experts.",
            provider=provider,
            router="biased_softmax_router",
        )
        return linear(mixed, prefix + "ffn_latent_up.weight", prefix + "ffn")

    current = norm(x, unit, "input_norm")
    for index, recurrent in enumerate(c.recurrent):
        prefix = f"blk.{index}."
        routed = residual(
            current,
            2 * index,
            prefix + "attn_route",
            weight(prefix + "attn_res_score.weight"),
        )
        normal = norm(
            routed, weight(prefix + "attn_norm.weight"), prefix + "attn_normal"
        )
        mixed = delta(normal, prefix) if recurrent else attention(normal, prefix)
        routed = residual(
            mixed,
            2 * index + 1,
            prefix + "ffn_route",
            weight(prefix + "ffn_res_score.weight"),
        )
        normal = norm(routed, weight(prefix + "ffn_norm.weight"), prefix + "ffn_normal")
        current = feed_forward(normal, prefix, index)
    routed = residual(
        current, sources - 1, "output_route", weight("output_res_score.weight")
    )
    hidden = norm(
        routed,
        weight("output_norm.weight"),
        "hidden",
        role="scratch" if include_head else "output",
    )
    if include_head:
        head_in = hidden
        if chunk > 1:
            head_in = g.buffer("head_hidden", (slots, c.width), "float32")
            g(
                "last_rows",
                dict(slots=slots, chunk=chunk, pool=pool, c=c.width, dtype="float32"),
                hidden,
                lengths,
                head_in,
            )
        logits = project(
            g,
            head_in,
            encoded(
                (CODEBOOK_HEAD,),
                (c.codebooks * c.code_vocab, c.width),
                store=checkpoint.projector,
            ),
            width=c.width,
            name="logits",
            role="output",
        )
        if sample:
            history = g.buffer(
                "sample_history", (pool, 96, c.codebooks), "int32", role="state"
            )
            steps = g.buffer("sample_steps", (slots,), "int32", role="input")
            temps = g.buffer("temperatures", (slots,), "float32", role="input")
            unifs = g.buffer("uniforms", (slots, c.codebooks), "float32", role="input")
            sampled = g.buffer(
                "sampled_frames", (slots, c.codebooks + 1), "int32", role="output"
            )
            g(
                "codebook_sample",
                dict(
                    slots=slots,
                    codebooks=c.codebooks,
                    vocab=c.code_vocab,
                    window=96,
                    rep_codebooks=6,
                    top_k=80,
                    top_p=0.69,
                    penalty=2.4,
                    softcap=c.logit_softcap,
                    silence_frames=c.codebooks,
                    text_pad=c.vocab - 1,
                ),
                logits.view((slots, c.codebooks * c.code_vocab)),
                history,
                steps,
                temps,
                unifs,
                sampled,
            )
    return g


HEAD_DIM = 64
STAGE_WINDOWS = (150, 300, 480, 480, 480)
SPECTRUM_ROWS_PER_FRAME = 1 << (len(STAGE_WINDOWS) - 1)


def compose_codec(checkpoint, *, frames, provider):
    if provider != "webgpu":
        raise ValueError("ZONOS3 codec requires WebGPU")
    config = checkpoint.config
    graph = Graph(contract="reference")
    table = checkpoint.projector.tensors
    root = "a.gen.wav."
    head_unit = graph.buffer("codec_unit", (HEAD_DIM,), "float32", role="input")

    def weight(name):
        if name not in graph.buffers:
            if table[name].type != 0:
                raise ValueError(f"ZONOS3 codec weight must be F32: {name}")
            checkpoint.graph_weight(graph, name, provider=provider)
        return graph.buffer_at(name)

    def dense(x, name):
        tensor_name = name + ".weight"
        columns, depth = table[tensor_name].shape
        rows = x.shape[0]
        out = graph.buffer(name, (rows, columns), "float32")
        graph(
            "linear",
            dict(r=rows, k=depth, o=columns, tiled_f32=True),
            x.view((rows * depth,)),
            weight(tensor_name),
            out.view((rows * columns,)),
        )
        return out

    def activated(x, name, *, bias=None, gelu=False, role="scratch"):
        rows, width = x.shape
        out = graph.buffer(name, x.shape, "float32", role=role)
        p = dict(r=rows, c=width)
        if bias is None:
            graph("gelu", p, x, out)
        else:
            graph("gelu_bias" if gelu else "column_bias", p, x, weight(bias), out)
        return out

    def norm(x, prefix, epsilon):
        rows, width = x.shape
        out = graph.buffer(prefix + ".normal", x.shape, "float32")
        graph(
            "affine_layer_norm",
            dict(r=rows, c=width, eps=epsilon),
            x,
            weight(prefix + ".weight"),
            weight(prefix + ".bias"),
            out,
        )
        return out

    def add(x, y, name, *, scale=None):
        rows, width = x.shape
        out = graph.buffer(name, x.shape, "float32")
        parameters = dict(r=rows, c=width)
        if scale is None:
            flat = (rows * width,)
            graph("add", parameters, x.view(flat), y.view(flat), out.view(flat))
        else:
            graph("scaled_residual", parameters, x, y, weight(scale), out)
        return out

    def convnext(x, prefix):
        rows, width = x.shape
        dw = prefix + ".dw.weight"
        taps = table[dw].shape[-1]
        filtered = graph.buffer(prefix + ".filtered", x.shape, "float32")
        graph(
            "causal_depthwise",
            dict(r=rows, c=width, taps=taps),
            x,
            weight(dw).view((width, taps)),
            weight(prefix + ".dw.bias"),
            filtered,
        )
        h = norm(filtered, prefix + ".norm", 1e-6)
        h = activated(
            dense(h, prefix + ".pw1"),
            prefix + ".gelu",
            bias=prefix + ".pw1.bias",
            gelu=True,
        )
        h = activated(
            dense(h, prefix + ".pw2"), prefix + ".biased", bias=prefix + ".pw2.bias"
        )
        return add(x, h, prefix + ".residual")

    def attention(x, prefix, window, controls):
        rows, width = x.shape
        heads, d = width // HEAD_DIM, HEAD_DIM
        positions, active, rope = controls
        normal = norm(x, prefix + ".norm1", 1e-5)
        qkv = dense(normal, prefix + ".qkv")
        q = graph.buffer(prefix + ".q", (rows * width,), "float32")
        k = graph.buffer(prefix + ".k", (rows * width,), "float32")
        v = graph.buffer(prefix + ".v", (rows, width), "float32")
        p = dict(r=rows, h=heads, kh=heads, d=d)
        graph("qkv_split", p, qkv.view((rows * 3 * width,)), q, k, v)
        qr, kr = (
            graph.buffer(prefix + name, (rows, width), "float32")
            for name in (".qr", ".kr")
        )
        for source, out in ((q, qr), (k, kr)):
            graph(
                "head_rms_rotary",
                dict(p, capacity=rows, scale_heads=1, normalize=False),
                source.view((rows, width)),
                head_unit,
                rope,
                positions,
                out,
            )
        shape = (1, heads, rows, d)
        kc, vc = (
            graph.buffer(prefix + name, shape, "float16") for name in (".kc", ".vc")
        )
        params = dict(p, capacity=rows, chunk=rows, window=window, splits=1)
        graph("window_cache", params, kr, v, kc, vc, positions, active)
        parts = graph.buffer(prefix + ".parts", (rows * heads * (d + 2),), "float32")
        graph(
            "window_attention",
            dict(params, query_rows=4 if rows % 4 == 0 else 1, key_rows=32),
            qr,
            kc,
            vc,
            positions,
            active,
            parts,
        )
        flat = graph.buffer(prefix + ".attended", (rows * width,), "float32")
        graph("window_merge", params, parts, flat, active)
        projected = dense(flat.view((rows, width)), prefix + ".wo")
        h = add(x, projected, prefix + ".attn_residual", scale=prefix + ".attn_scale")
        z = norm(h, prefix + ".norm2", 1e-5)
        z = activated(dense(z, prefix + ".ffn1"), prefix + ".ffn_gelu", gelu=True)
        z = dense(z, prefix + ".ffn2")
        return add(h, z, prefix + ".ffn_residual", scale=prefix + ".ffn_scale")

    semantic = graph.buffer("semantic", (frames,), "int32", role="input")
    residual = graph.buffer(
        "residual_codes", (frames, config.codebooks - 1), "int32", role="input"
    )
    semantic_table = root + "sem.table"
    vocab, width = table[semantic_table].shape
    if vocab != config.codes:
        raise ValueError("ZONOS3 semantic codebook size differs from its audio head")
    embedded = graph.buffer("semantic_embedding", (frames * width,), "float32")
    graph(
        "embedding",
        dict(r=frames, c=width, v=vocab),
        semantic,
        weight(semantic_table),
        embedded,
    )
    x = activated(
        embedded.view((frames, width)), "semantic_input", bias=root + "sem.bias"
    )
    for block in range(4):
        x = convnext(x, root + f"sem.{block}")
    x = activated(
        dense(x, root + "sem.out"), "semantic_output", bias=root + "sem.out.bias"
    )
    latent = graph.buffer("latent", x.shape, "float32")
    graph(
        "codebook_residual",
        dict(r=frames, c=x.shape[1], groups=config.codebooks - 1, vocab=vocab),
        x,
        residual,
        weight(root + "res.table"),
        weight(root + "res.bias"),
        latent,
    )
    x = latent
    for stage, window in enumerate(STAGE_WINDOWS):
        prefix = root + f"stage.{stage}"
        x = dense(x, prefix + ".in")
        rows, width = x.shape
        controls = (
            graph.buffer(f"codec_positions.{rows}", (rows,), "int32", role="input"),
            graph.buffer(f"codec_active.{rows}", (rows,), "int32", role="input"),
            graph.buffer(
                f"codec_rope.{rows}", (rows, HEAD_DIM), "float32", role="input"
            ),
        )
        for block in range(2):
            x = attention(x, prefix + f".blk.{block}", window, controls)
        if stage + 1 < len(STAGE_WINDOWS):
            x = dense(x, prefix + ".out")
            shuffled = graph.buffer(
                prefix + ".upsampled", (2 * rows, width // 2), "float32"
            )
            graph("interleave_rows", dict(r=rows, c=width, factor=2), x, shuffled)
            x = shuffled
    for block in range(8):
        x = convnext(x, root + f"head.{block}")
    x = norm(x, root + "head.norm", 1e-6)
    activated(
        dense(x, root + "head.out"),
        "spectrum",
        bias=root + "head.out.bias",
        role="output",
    )
    return graph


class Zonos3Mini:
    """A single ZONOS3 speech backbone request over common Forward bundles."""

    def __init__(self, checkpoint, directory, device, *, phases):
        self.config = checkpoint.config
        self.device = device
        self._resources = ExitStack()
        self._forwards, self._states = {}, {}
        self.position = 0
        directory = Path(directory)
        definitions = {}
        shared = {}
        try:
            for rows, phase in phases.items():
                manifest = (
                    phase
                    if isinstance(phase, dict)
                    else read_forward(directory / phase)
                )
                if manifest["provider"] != device.info["provider"]:
                    raise ValueError("ZONOS3 phase targets a different backend")
                graph = Graph.from_metadata(manifest["forward"])
                if graph.buffers["embedding"].shape != (rows, self.config.width):
                    raise ValueError(
                        "ZONOS3 phase row count does not match its manifest"
                    )
                definitions[rows] = (manifest, graph)
                for name, spec in graph.buffers.items():
                    if spec.role in ("state", "weight") and spec.alias_of is None:
                        previous = shared.setdefault(name, spec)
                        if previous != spec:
                            raise ValueError(
                                f"incompatible ZONOS3 phase buffer: {name}"
                            )
            if 1 not in definitions:
                raise ValueError("ZONOS3 requires a one-row decode phase")
            self.prefill_rows = max(definitions)
            self.context = shared["rope"].shape[0]
            weights = checkpoint.upload(
                device,
                (name for name, spec in shared.items() if spec.role == "weight"),
                context=self.context,
            )
            for buffer in weights.values():
                self._resources.callback(buffer.release)
            for name, spec in shared.items():
                if spec.role == "state":
                    buffer = device.zeros(spec.shape, spec.dtype)
                    self._states[name] = buffer
                    self._resources.callback(buffer.release)
            for rows, (manifest, graph) in definitions.items():
                supplied = {
                    name: buffer
                    for name, buffer in (weights | self._states).items()
                    if name in graph.buffers
                }
                forward = Forward(
                    directory,
                    device,
                    supplied,
                    manifest=manifest,
                    graphs=True,
                    allocate_io=True,
                )
                self._forwards[rows] = self._resources.enter_context(forward)
        except BaseException:
            self.close()
            raise

    def reset(self):
        write_many(
            self.device,
            (
                (buffer, np.zeros(buffer.shape, str(buffer.dtype)))
                for buffer in self._states.values()
            ),
        )
        self.position = 0

    def advance(self, embeddings, *, read=True):
        """Run a compiled chunk and optionally read its F32 hidden states."""
        rows = np.asarray(embeddings, dtype="float32")
        if rows.ndim != 2 or rows.shape[1] != self.config.width or not len(rows):
            raise ValueError("invalid ZONOS3 embedding rows")
        if self.position + len(rows) > self.context:
            raise ValueError("ZONOS3 request exceeds its compiled context")
        count = len(rows)
        eligible = [capacity for capacity in self._forwards if capacity >= count]
        if not eligible:
            raise ValueError("ZONOS3 request chunk does not have a compiled phase")
        capacity = min(eligible)
        forward = self._forwards[capacity]
        positions = np.zeros(capacity, dtype="int32")
        positions[:count] = np.arange(
            self.position, self.position + count, dtype="int32"
        )
        active = np.zeros(capacity, dtype="int32")
        active[:count] = 1
        values = dict(
            embedding=np.pad(rows, ((0, capacity - count), (0, 0)))
            if count < capacity
            else rows,
            positions=positions,
            active=active,
            lengths=np.array([count], dtype="int32"),
            mapping=np.zeros(1, dtype="int32"),
        )
        write_many(
            self.device,
            ((forward.buffers[name], value) for name, value in values.items()),
        )
        forward.run()
        hidden = forward.buffers["hidden"].to_numpy()[:count] if read else None
        self.position += len(rows)
        return hidden

    def close(self):
        self._resources.close()
        self._forwards.clear()
        self._states.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class CodeFrames:
    frames: np.ndarray
    generated_rows: int
    terminated: bool


def feedback_ids(config, codes):
    values = np.asarray(codes)
    if (
        values.ndim < 1
        or values.shape[-1] != config.codebooks
        or values.dtype.kind not in "iu"
        or np.any(values < 0)
        or np.any(values >= config.code_vocab)
    ):
        raise ValueError("invalid ZONOS3 codebook IDs")
    offsets = np.arange(config.codebooks, dtype="int32") * config.code_vocab
    return (values.astype("int32") + offsets).reshape(-1)


def feedback_embedding(config, embedded):
    rows = np.asarray(embedded, dtype="float32").reshape(
        -1, config.codebooks, config.width
    )
    return rows.sum(axis=1, dtype="float32")


def codebook_logits(config, projection):
    logits = np.asarray(projection, dtype="float32").reshape(
        -1, config.codebooks, config.code_vocab
    )
    limit = config.logit_softcap
    return limit * np.tanh(logits / limit)


def silence_prefix(config):
    rows = np.full((SILENCE_FRAMES, config.codebooks), config.codes + 1, dtype="int32")
    for index, row in enumerate(rows):
        row[: index + 1] = SILENCE[: index + 1]
    return rows


def _sample(logits, history, temperature, rng, eoa):
    value = logits.copy()
    value[eoa + 1] = -np.inf
    if history:
        used = np.unique(history)
        used = used[used < eoa]
        penalty = 2.4
        value[used] = np.where(
            value[used] > 0, value[used] / penalty, value[used] * penalty
        )
    if temperature == 0:
        return int(np.argmax(value))
    value /= temperature
    top_k, top_p = 80, 0.69
    threshold = np.partition(value, -top_k)[-top_k]
    indices = np.flatnonzero(value >= threshold)
    selected = value[indices].astype("float64")
    weights = np.exp(selected - selected.max())
    order = np.argsort(-weights, kind="stable")
    sorted_weights = weights[order]
    before = np.cumsum(sorted_weights) - sorted_weights
    weights[order[before > top_p * weights.sum()]] = 0
    cdf = np.cumsum(weights)
    return int(
        indices[np.searchsorted(cdf, rng.random_sample() * cdf[-1], side="right")]
    )


def generate_codes(config, initial_logits, advance, *, seed, temperature, limit):
    """Sample delayed rows until EOA and reserve its trailing delay for deinterleaving."""
    if (
        not np.isfinite(temperature)
        or temperature < 0
        or type(limit) is not int
        or limit < 1
    ):
        raise ValueError("invalid ZONOS3 sampling settings")
    rng = np.random.RandomState(seed)
    rows = list(silence_prefix(config))
    logits = initial_logits
    eoa, pad = config.codes, config.codes + 1
    eos_frame, stop_row = None, None
    columns = np.arange(config.codebooks)
    for step in range(limit):
        logits = np.asarray(logits, dtype="float32").reshape(
            config.codebooks, config.code_vocab
        )
        if not np.isfinite(logits).all():
            raise ValueError("ZONOS3 produced non-finite audio logits")
        recent = rows[max(SILENCE_FRAMES, len(rows) - 96) :]
        sample = np.array(
            [
                _sample(
                    logits[cb],
                    [row[cb] for row in recent] if cb < 6 else (),
                    temperature,
                    rng,
                    eoa,
                )
                for cb in range(config.codebooks)
            ],
            dtype="int32",
        )
        index = len(rows)
        sample[index + 1 :] = pad
        rows.append(sample)
        ended = np.flatnonzero(sample == eoa)
        if eos_frame is None and len(ended):
            eos_frame = max(0, index - int(ended[-1]))
            stop_row = index + config.codebooks
        if index == stop_row or step + 1 == limit:
            break
        logits = advance(sample)
    ready = max(0, len(rows) - config.codebooks + 1)
    end = ready
    if eos_frame is not None:
        end = min(end, eos_frame)
    delayed = np.asarray(rows, dtype="int32")
    codes = delayed[np.arange(end)[:, None] + columns, columns]
    return CodeFrames(
        np.minimum(codes, config.codes - 1),
        len(rows) - SILENCE_FRAMES,
        eos_frame is not None and ready >= eos_frame,
    )


def waveform(checkpoint, spectrum):
    """Reconstruct mono 44.1 kHz PCM from the two ZAC inverse STFT branches."""
    sizes, hop = (736, 2208), 184
    data = np.asarray(spectrum, dtype="float32")
    if data.ndim != 2 or data.shape[1] != sum(n + 2 for n in sizes) or not len(data):
        raise ValueError("invalid ZONOS3 ZAC spectrum")
    mixer = checkpoint.projector.array("a.gen.wav.mixer.weight")
    resample = checkpoint.projector.array("a.gen.wav.resample.weight")
    length, offset = len(data) * hop, 0
    mixed = np.zeros(length, dtype="float32")
    for n, taps in zip(sizes, mixer, strict=True):
        bins = n // 2 + 1
        magnitude = np.exp(np.minimum(data[:, offset : offset + bins], np.log(100)))
        phase = data[:, offset + bins : offset + 2 * bins]
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
        frames = np.fft.irfft(magnitude * np.exp(1j * phase), n=n, axis=1)
        signal = np.zeros(length + n, dtype="float32")
        envelope = np.zeros_like(signal)
        for i, frame in enumerate(frames):
            region = slice(i * hop, i * hop + n)
            signal[region] += frame * window
            envelope[region] += window**2
        centered = slice(n // 2, n // 2 + length)
        branch = signal[centered] / np.maximum(envelope[centered], 1e-11)
        mixed += np.correlate(branch, taps, mode="same")
        offset += 2 * bins
    block = 736
    width = (resample.shape[1] - block) // 2
    padded = np.pad(mixed, (width, width + (-length % block)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, resample.shape[1])[
        ::block
    ]
    samples = (windows @ resample.T).reshape(-1)[: (length * 735 + 735) // block]
    samples = (
        samples[SILENCE_FRAMES * checkpoint.config.samples_per_frame :] * 0.70794578
    )
    knee = 10 ** (-1 / 20)
    limit = knee + (1 - knee) * np.tanh((np.abs(samples) - knee) / (1 - knee))
    return np.where(
        np.abs(samples) <= knee, samples, np.copysign(limit, samples)
    ).astype("float32")


@dataclass(frozen=True)
class Speech:
    samples: np.ndarray
    sample_rate: int
    frames: int
    terminated: bool


class Zonos3MiniSynthesis:
    """Declare the checkpoint, compiled phases, and runtime for ZONOS3-Mini."""

    checkpoint_type = Zonos3MiniCheckpoint
    compose = staticmethod(compose)
    forward_index = ("speech.json", "akas.speech-program.v1")

    @classmethod
    def forward_phases(
        cls,
        checkpoint,
        *,
        provider,
        context=1024,
        prefill_rows=64,
        codec_frames=(16, 32, 64, 128),
    ):
        if (
            type(prefill_rows) is not int
            or type(context) is not int
            or not 1 <= prefill_rows <= context
            or not codec_frames
            or any(type(n) is not int or n < 1 for n in codec_frames)
        ):
            raise ValueError("invalid ZONOS3 phase geometries")
        geometries = (
            ("codec", sorted(set(codec_frames))),
            ("backbone", sorted({1, prefill_rows})),
            ("audio_head", (1,)),
        )
        return dict(context=context), (
            (
                phase,
                rows,
                cls.compose(
                    checkpoint,
                    phase=phase,
                    rows=rows,
                    provider=provider,
                    context=context,
                ),
            )
            for phase, sizes in geometries
            for rows in sizes
        )

    def __init__(self, checkpoint, directory, device):
        self.checkpoint, self.config, self.device = (
            checkpoint,
            checkpoint.config,
            device,
        )
        self._resources = ExitStack()
        self._codecs = {}
        directory = Path(directory).resolve()
        filename, schema = self.forward_index
        phases = read_forward_set(
            directory, filename, schema=schema, provider=device.info["provider"]
        )
        head = phases["audio_head"][1]
        codecs = phases["codec"]
        try:
            self.backbone = self._resources.enter_context(
                Zonos3Mini(checkpoint, directory, device, phases=phases["backbone"])
            )
            self._feedback_table = checkpoint.projector.array(CODEBOOK_EMBEDDING)
            self._directory = directory
            weights = {
                spec["name"]
                for phase in (head, *codecs.values())
                for spec in phase["forward"]["buffers"]
                if spec["role"] == "weight" and spec.get("alias_of") is None
            }
            self._weights = checkpoint.upload(device, sorted(weights))
            for buffer in self._weights.values():
                self._resources.callback(buffer.release)
            self._head = self._open(head)
            for count, phase in codecs.items():
                self._codecs[count] = self._open(phase)
        except BaseException:
            self.close()
            raise

    def _open(self, manifest):
        graph = Graph.from_metadata(manifest["forward"])
        supplied = {
            name: value
            for name, value in self._weights.items()
            if name in graph.buffers
        }
        for name, spec in graph.buffers.items():
            if (
                spec.alias_of is not None
                or spec.role != "input"
                or not name.startswith("codec_")
            ):
                continue
            control, _, size = name.partition(".")
            if control in ("codec_active", "codec_unit"):
                value = np.ones(spec.shape, dtype=spec.dtype)
            elif control == "codec_positions":
                value = np.arange(int(size), dtype="int32")
            else:
                positions = np.arange(int(size))[:, None]
                angle = positions * 10000 ** (-np.arange(0, HEAD_DIM, 2) / HEAD_DIM)
                value = np.concatenate((np.cos(angle), np.sin(angle)), axis=1).astype(
                    "float32"
                )
            buffer = self.device.from_numpy(value)
            supplied[name] = buffer
            self._resources.callback(buffer.release)
        return self._resources.enter_context(
            Forward(
                self._directory,
                self.device,
                supplied,
                manifest=manifest,
                graphs=True,
                allocate_io=True,
            )
        )

    def _predict(self, hidden):
        write_many(self.device, ((self._head.buffers["hidden"], hidden.reshape(-1)),))
        self._head.run()
        return codebook_logits(
            self.config, self._head.buffers["projection"].to_numpy()
        )[0]

    def _embedding(self, codes):
        ids = feedback_ids(self.config, codes)
        return feedback_embedding(self.config, self._feedback_table[ids])

    def generate_codes(
        self,
        text,
        *,
        lang="en",
        voice=None,
        speaker_row=None,
        seed=17,
        temperature=0.75,
        limit=384,
    ):
        """Generate aligned ZAC frames, including the initial codec silence."""
        if speaker_row is not None:
            if voice is not None:
                raise ValueError(
                    "choose either a predefined voice or a projected speaker row"
                )
            speaker_row = np.asarray(speaker_row)
            if (
                speaker_row.shape != (1, self.config.width)
                or speaker_row.dtype != np.float32
                or not np.isfinite(speaker_row).all()
            ):
                raise ValueError("invalid ZONOS3 projected F32 speaker row")
        else:
            speaker_row = self.checkpoint.voice_rows(voice, lang=lang)
        self.backbone.reset()
        ids = self.checkpoint.prompt_ids(text)
        text_rows = [
            self.checkpoint.backbone.row(TOKEN_EMBEDDING, token) for token in ids
        ]
        prefix = np.concatenate(
            (
                speaker_row,
                np.stack(text_rows),
                self._embedding(silence_prefix(self.config)),
            )
        )
        if len(prefix) + limit - 1 > self.backbone.context:
            raise ValueError("ZONOS3 text and frame limit exceed the compiled context")
        size = self.backbone.prefill_rows
        hidden = None
        for start in range(0, len(prefix), size):
            end = start + size
            hidden = self.backbone.advance(prefix[start:end], read=end >= len(prefix))

        def next_logits(row):
            return self._predict(self.backbone.advance(self._embedding(row))[-1])

        return generate_codes(
            self.config,
            self._predict(hidden[-1]),
            next_logits,
            seed=seed,
            temperature=temperature,
            limit=limit,
        )

    def decode(self, frames):
        count = len(frames)
        if count <= SILENCE_FRAMES:
            raise ValueError("ZONOS3 generated no complete speech frames")
        candidates = [n for n in self._codecs if n >= count]
        if not candidates:
            raise ValueError("ZONOS3 audio exceeds the compiled codec frame capacities")
        capacity = min(candidates)
        forward = self._codecs[capacity]
        codes = np.pad(frames, ((0, capacity - count), (0, 0)))
        write_many(
            self.device,
            (
                (forward.buffers["semantic"], codes[:, 0]),
                (forward.buffers["residual_codes"], codes[:, 1:]),
            ),
        )
        forward.run()
        spectrum = forward.buffers["spectrum"].to_numpy()[
            : count * SPECTRUM_ROWS_PER_FRAME
        ]
        return waveform(self.checkpoint, spectrum)

    def synthesize(self, text, **options):
        frames = self.generate_codes(text, **options)
        samples = self.decode(frames.frames)
        return Speech(
            samples,
            self.config.sample_rate,
            len(frames.frames) - SILENCE_FRAMES,
            frames.terminated,
        )

    def close(self):
        self._resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
