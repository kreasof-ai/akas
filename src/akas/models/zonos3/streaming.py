"""Multi-slot BF16 ZONOS3-Mini prefix-streaming composition, runtime, and scheduler."""

from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import json
from math import isfinite, prod
from pathlib import Path
from threading import Condition, Event, Thread

import numpy as np

from akas.artifacts.read import artifact_path, read_forward
from akas.benchmarking.provenance import digest as checksum
from akas.artifacts.write import publish
from akas.execution.forward import Forward
from akas.execution.state import StatePool
from akas.graph import Graph
from akas.memory.transfer import write_many
from akas.models.zonos3.model import HEAD_DIM, STAGE_WINDOWS, compose_backbone
from akas.models.zonos3.config import SILENCE, SILENCE_FRAMES
from akas.models.zonos3.weights import Zonos3MiniCheckpoint

STAGE_CAPACITIES = (256, 512, 768, 768, 768)
SAMPLES_PER_FRAME = 2940
HEADROOM_GAIN = float(10.0 ** (-3.0 / 20.0))
LIMITER_KNEE = float(10.0 ** (-1.0 / 20.0))
STREAM_INDEX = ("streaming.json", "akas.zonos3-streaming.v1")


def compose_lm_step(
    checkpoint, *, slots, chunk, pool=256, context=512, provider="cuda"
):
    """Compose mapped on-device frame embedding, BF16 backbone + head, and sampling."""
    return compose_backbone(
        checkpoint,
        slots=slots,
        chunk=chunk,
        pool=pool,
        context=context,
        provider=provider,
        include_head=True,
        embed_frames=True,
        sample=True,
    )


def compose_codec_stream(checkpoint, *, slots, frames, pool=256, provider="cuda"):
    """Compose stateful multi-slot keep2_plain prefix codec + streaming GPU synthesis."""
    from akas.ops.linear.definition import floating_matmuls as _floating_matmuls

    if (
        provider not in ("cuda", "webgpu")
        or any(type(v) is not int or v < 1 for v in (slots, frames, pool))
        or slots > pool
        or frames > 16
    ):
        raise ValueError("invalid ZONOS3 streaming codec geometry")
    config = checkpoint.config
    g = Graph(contract="reference")
    table = checkpoint.projector.tensors
    root = "a.gen.wav."

    mapping = g.buffer("mapping", (slots,), "int32", role="input")
    f_lengths = g.buffer("f_lengths", (slots,), "int32", role="input")
    len_16 = g.buffer("len_16", (slots,), "int32", role="input")
    semantic = g.buffer("semantic", (slots * frames,), "int32", role="input")
    residual = g.buffer(
        "residual_codes", (slots * frames, config.codebooks - 1), "int32", role="input"
    )
    head_unit = g.buffer("codec_unit.weight", (HEAD_DIM,), "float32", role="weight")
    del mapping  # consumed automatically by compilation.state:map_requests

    def weight(name):
        if name not in g.buffers:
            if table[name].type != 0:
                raise ValueError(f"ZONOS3 codec weight must be F32: {name}")
            checkpoint.graph_weight(g, name, provider=provider)
        return g.buffer_at(name)

    codec_rope = weight("codec_rope.weight").view((8192, HEAD_DIM))

    def dense(x, name):
        tensor_name = name + ".weight"
        columns, depth = table[tensor_name].shape
        rows = x.shape[0]
        out = g.buffer(name, (rows, columns), "float32")
        params = dict(r=rows, k=depth, o=columns)
        if provider == "webgpu":
            params["tiled_f32"] = True
        g(
            "linear",
            params,
            x.view((rows * depth,)),
            weight(tensor_name),
            out.view((rows * columns,)),
        )
        return out

    def activated(x, name, *, bias=None, gelu=False, role="scratch"):
        rows, width = x.shape
        out = g.buffer(name, x.shape, "float32", role=role)
        p = dict(r=rows, c=width)
        if bias is None:
            g("gelu", p, x, out)
        else:
            g("gelu_bias" if gelu else "column_bias", p, x, weight(bias), out)
        return out

    def norm(x, prefix, epsilon):
        rows, width = x.shape
        out = g.buffer(prefix + ".normal", x.shape, "float32")
        g(
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
        out = g.buffer(name, x.shape, "float32")
        p = dict(r=rows, c=width)
        if scale is None:
            flat = (rows * width,)
            g("add", p, x.view(flat), y.view(flat), out.view(flat))
        else:
            g("scaled_residual", p, x, y, weight(scale), out)
        return out

    def convnext_state(x, prefix, *, chunk_rows, length_buf):
        rows, width = x.shape
        dw = prefix + ".dw.weight"
        taps = table[dw].shape[-1]
        st = g.buffer(
            prefix + ".state", (pool, taps - 1, width), "float32", role="state"
        )
        filtered = g.buffer(prefix + ".filtered", x.shape, "float32")
        g(
            "causal_depthwise_state",
            dict(slots=slots, chunk=chunk_rows, c=width, taps=taps),
            x,
            weight(dw).view((width, taps)),
            weight(prefix + ".dw.bias"),
            st,
            length_buf,
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

    def attention(x, prefix, *, chunk_s, window, cap, pos_s, act_s):
        rows, width = x.shape
        heads, d = width // HEAD_DIM, HEAD_DIM
        normal = norm(x, prefix + ".norm1", 1e-5)
        qkv = dense(normal, prefix + ".qkv")
        q = g.buffer(prefix + ".q", (rows * width,), "float32")
        k = g.buffer(prefix + ".k", (rows * width,), "float32")
        v = g.buffer(prefix + ".v", (rows, width), "float32")
        p = dict(r=rows, h=heads, kh=heads, d=d)
        g("qkv_split", p, qkv.view((rows * 3 * width,)), q, k, v)
        qr, kr = (
            g.buffer(prefix + tag, (rows, width), "float32") for tag in (".qr", ".kr")
        )
        for src, out in ((q, qr), (k, kr)):
            g(
                "head_rms_rotary",
                dict(p, capacity=8192, scale_heads=1, normalize=False),
                src.view((rows, width)),
                head_unit,
                codec_rope,
                pos_s,
                out,
            )
        shape = (pool, heads, cap, d)
        kc, vc = (
            g.buffer(prefix + tag, shape, "float16", role="state")
            for tag in (".kc", ".vc")
        )
        ap = dict(p, capacity=cap, chunk=chunk_s, window=window, splits=1)
        g("window_cache", ap, kr, v, kc, vc, pos_s, act_s)
        parts = g.buffer(prefix + ".parts", (rows * heads * (d + 2),), "float32")
        q_rows = 16 if provider == "cuda" else (4 if chunk_s % 4 == 0 else 1)
        k_rows = 64 if provider == "cuda" else 32
        g(
            "window_attention",
            dict(ap, query_rows=q_rows, key_rows=k_rows),
            qr,
            kc,
            vc,
            pos_s,
            act_s,
            parts,
        )
        flat = g.buffer(prefix + ".attended", (rows * width,), "float32")
        g("window_merge", ap, parts, flat, act_s)
        projected = dense(flat.view((rows, width)), prefix + ".wo")
        h = add(x, projected, prefix + ".attn_residual", scale=prefix + ".attn_scale")
        z = norm(h, prefix + ".norm2", 1e-5)
        z = activated(dense(z, prefix + ".ffn1"), prefix + ".ffn_gelu", gelu=True)
        z = dense(z, prefix + ".ffn2")
        return add(h, z, prefix + ".ffn_residual", scale=prefix + ".ffn_scale")

    sem_rows = slots * frames
    sem_table = root + "sem.table"
    vocab, sem_width = table[sem_table].shape
    embedded = g.buffer("semantic_embedding", (sem_rows * sem_width,), "float32")
    g(
        "embedding",
        dict(r=sem_rows, c=sem_width, v=vocab),
        semantic,
        weight(sem_table),
        embedded,
    )
    x = activated(
        embedded.view((sem_rows, sem_width)), "semantic_input", bias=root + "sem.bias"
    )
    for block in range(4):
        x = convnext_state(
            x, root + f"sem.{block}", chunk_rows=frames, length_buf=f_lengths
        )
    x = activated(
        dense(x, root + "sem.out"), "semantic_output", bias=root + "sem.out.bias"
    )
    latent = g.buffer("latent", x.shape, "float32")
    g(
        "codebook_residual_masked",
        dict(r=sem_rows, c=x.shape[1], groups=config.codebooks - 1, vocab=vocab),
        x,
        residual,
        weight(root + "res.table"),
        weight(root + "res.bias"),
        latent,
    )
    x = latent
    for stage, (window, cap) in enumerate(
        zip(STAGE_WINDOWS, STAGE_CAPACITIES, strict=True)
    ):
        prefix = root + f"stage.{stage}"
        x = dense(x, prefix + ".in")
        rows_s, width = x.shape
        chunk_s = frames * (1 << stage)
        pos_s = g.buffer(f"pos_{stage}", (rows_s,), "int32", role="input")
        act_s = g.buffer(f"act_{stage}", (rows_s,), "int32", role="input")
        for blk in range(2):
            x = attention(
                x,
                prefix + f".blk.{blk}",
                chunk_s=chunk_s,
                window=window,
                cap=cap,
                pos_s=pos_s,
                act_s=act_s,
            )
        if stage + 1 < len(STAGE_WINDOWS):
            x = dense(x, prefix + ".out")
            shuffled = g.buffer(
                prefix + ".upsampled", (2 * rows_s, width // 2), "float32"
            )
            g("interleave_rows", dict(r=rows_s, c=width, factor=2), x, shuffled)
            x = shuffled

    chunk_16 = frames * 16
    rows_16 = slots * chunk_16
    for block in range(8):
        x = convnext_state(
            x, root + f"head.{block}", chunk_rows=chunk_16, length_buf=len_16
        )
    x = norm(x, root + "head.norm", 1e-6)
    spectrum = activated(
        dense(x, root + "head.out"),
        "spectrum",
        bias=root + "head.out.bias",
        role="output",
    )

    spec0 = g.buffer("spec0", (rows_16, 738), "float32")
    spec1 = g.buffer("spec1", (rows_16, 2210), "float32")
    g("istft_polar", dict(r=rows_16), spectrum, spec0, spec1)
    fr0 = dense(spec0, root + "fourier0")
    fr1 = dense(spec1, root + "fourier1")

    ola_state = g.buffer("istft_ola_state", (pool, 3564), "float32", role="state")
    span = chunk_16 * 184
    cat0 = g.buffer("istft_cat0", (slots, 862 + span + 552), "float32")
    cat1 = g.buffer("istft_cat1", (slots, 126 + span + 2024), "float32")
    g(
        "istft_ola",
        dict(slots=slots, frames=frames),
        fr0,
        fr1,
        ola_state,
        f_lengths,
        cat0,
        cat1,
    )
    mid = g.buffer("istft_mid", (slots, span), "float32")
    g(
        "istft_mix_commit",
        dict(slots=slots, frames=frames),
        cat0,
        cat1,
        weight(root + "mixer.weight").view((2, 127)),
        ola_state,
        f_lengths,
        mid,
    )
    rs_state = g.buffer("istft_rs_state", (pool, 322), "float32", role="state")
    pcm = g.buffer("pcm", (slots, frames * SAMPLES_PER_FRAME), "float32", role="output")
    g(
        "sinc_resample_limit",
        dict(slots=slots, frames=frames, gain=HEADROOM_GAIN, knee=LIMITER_KNEE),
        mid,
        weight(root + "resample.weight").view((735, 770)),
        rs_state,
        f_lengths,
        pcm,
    )
    g(
        "sinc_resample_commit",
        dict(slots=slots, frames=frames),
        mid,
        rs_state,
        f_lengths,
    )

    if provider == "cuda":
        _floating_matmuls(g, minimum_rows=9)
    return g


def compose_state_reset(state_specs, *, slots, pool):
    """Compose a batched clear_rows reset phase for recurrent/conv/synthesis states."""
    g = Graph(contract="state", phase="reset")
    mapping = g.buffer("mapping", (slots,), "int32", role="input")
    for name in sorted(state_specs):
        spec = state_specs[name]
        if (
            name.endswith((".kc", ".vc"))
            or name == "sample_history"
            or spec.shape[0] != pool
        ):
            continue
        st = g.buffer(name, spec.shape, spec.dtype, role="state")
        width = prod(spec.shape[1:])
        rows = g.view(name + ".rows", st, (pool, width))
        g(
            "clear_rows",
            dict(rows=slots, pool=pool, width=width, dtype=spec.dtype),
            mapping,
            rows,
        )
    return g


def compile_streaming_bundle(
    checkpoint,
    out_dir,
    *,
    device,
    provider="cuda",
    pool=256,
    context=512,
    prefill_slots=(1, 4, 16),
    prefill_chunk=64,
    decode_slots=(1, 4, 16, 64, 256),
    codec_slots=(1, 4, 16),
    codec_frames=(1, 4, 15),
    reset_slots=16,
    search_settings=None,
):
    """Compile and op-qualify all multi-slot ZONOS3 streaming phases with resume."""
    from akas.compilation.compile import build, plan
    from akas.compilation.tuning import SearchSettings

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = search_settings or SearchSettings(budget=1, samples=1, repeats=2)

    state_specs = {}
    records = {"prefill": {}, "decode": {}, "codec": {}}

    def register_states(graph):
        for name, spec in graph.buffers.items():
            if spec.role == "state" and spec.alias_of is None:
                prev = state_specs.setdefault(name, spec)
                if (prev.shape, prev.dtype) != (spec.shape, spec.dtype):
                    raise ValueError(f"incompatible state buffer across phases: {name}")

    def build_phase(graph, sub, *, search=True):
        target_json = out / sub / "forward.json"
        if target_json.is_file():
            return
        print(f"Building phase {sub}...", flush=True)
        bundle = plan(graph, filename=f"{sub}/forward.json", provider=provider)
        bundle.source_directory = f"{sub}/sources"
        bundle.artifact_directory = f"{sub}/kernels"
        build(bundle, out, search=search, search_settings=settings, device=device)

    for s in sorted(set(prefill_slots)):
        g = compose_lm_step(
            checkpoint,
            slots=s,
            chunk=prefill_chunk,
            pool=pool,
            context=context,
            provider=provider,
        )
        register_states(g)
        sub = f"lm-pre-s{s}-c{prefill_chunk}"
        build_phase(g, sub)
        records["prefill"][str(s)] = {
            "slots": s,
            "chunk": prefill_chunk,
            "directory": sub,
            "sha256": checksum(out / sub / "forward.json"),
        }

    for s in sorted(set(decode_slots)):
        g = compose_lm_step(
            checkpoint, slots=s, chunk=1, pool=pool, context=context, provider=provider
        )
        register_states(g)
        sub = f"lm-dec-s{s}"
        build_phase(g, sub)
        records["decode"][str(s)] = {
            "slots": s,
            "chunk": 1,
            "directory": sub,
            "sha256": checksum(out / sub / "forward.json"),
        }

    for f in sorted(set(codec_frames)):
        by_slot = {}
        for s in sorted(set(codec_slots)):
            g = compose_codec_stream(
                checkpoint, slots=s, frames=f, pool=pool, provider=provider
            )
            register_states(g)
            sub = f"codec-s{s}-f{f}"
            build_phase(g, sub)
            by_slot[str(s)] = {
                "slots": s,
                "frames": f,
                "directory": sub,
                "sha256": checksum(out / sub / "forward.json"),
            }
        records["codec"][str(f)] = by_slot

    reset_slots = max(int(reset_slots), max(prefill_slots))
    g_reset = compose_state_reset(state_specs, slots=reset_slots, pool=pool)
    build_phase(g_reset, "reset", search=False)
    records["reset"] = {
        "slots": reset_slots,
        "directory": "reset",
        "sha256": checksum(out / "reset" / "forward.json"),
    }

    filename, schema = STREAM_INDEX
    manifest = {
        "schema": schema,
        "provider": provider,
        "pool": pool,
        "context": context,
        "prefill_chunk": prefill_chunk,
        "phases": records,
    }
    publish(out / filename, manifest)
    return out / filename


def _envelope_tail_ratios():
    """Precompute exact Hann^2 tail-envelope ratios for EOS flush."""
    ratios = []
    for n_fft, steady, needed in ((736, 1.5, 368), (2208, 4.5, 1104)):
        win2 = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)) ** 2
        m = n_fft // 184
        # Tail overlap at positions [L, L + needed): sum of remaining overlapping Hann^2 windows
        blocks = win2.reshape(m, 184)
        env = np.concatenate([blocks[k:].sum(axis=0) for k in range(1, m)], axis=0)[
            :needed
        ]
        ratios.append((steady / env).astype("float32"))
    return tuple(ratios)


_FLUSH_RATIOS = _envelope_tail_ratios()


def _soft_limit_pcm(raw):
    scaled = raw.astype("float32", copy=False) * np.float32(HEADROOM_GAIN)
    mag = np.abs(scaled)
    over = mag > np.float32(LIMITER_KNEE)
    if not over.any():
        return scaled
    out = scaled.copy()
    w = np.float32(1.0 - LIMITER_KNEE)
    out[over] = np.sign(out[over]) * (
        np.float32(LIMITER_KNEE)
        + w * np.tanh((mag[over] - np.float32(LIMITER_KNEE)) / w)
    )
    return out


def flush_codec_tail(ola_row, rs_row, mixer, resample_weight):
    """Exact 1,470-sample EOS flush matching IstftHeadStream.flush + ResamplerStream.flush."""
    r0, r1 = _FLUSH_RATIOS
    ola0 = ola_row[:552]
    ola1 = ola_row[552:2576]
    mix0 = ola_row[2576:3438]
    mix1 = ola_row[3438:3564]
    zeros_h = np.zeros(63, dtype="float32")
    m0 = np.concatenate((mix0, ola0[:368] * r0, zeros_h))
    m1 = np.concatenate((mix1, ola1[:1104] * r1, zeros_h))
    w0 = np.lib.stride_tricks.sliding_window_view(m0, 127)
    w1 = np.lib.stride_tricks.sliding_window_view(m1, 127)
    mid_flush = (w0 @ mixer[0] + w1 @ mixer[1]).astype("float32")
    cat_r = np.concatenate((rs_row, mid_flush, np.zeros(17, dtype="float32")))
    rw = np.lib.stride_tricks.sliding_window_view(cat_r, 770)[::736][:2]
    tail = (rw @ resample_weight.T).reshape(-1)
    return _soft_limit_pcm(tail)


def undelay_codes(
    delayed_rows, start, end, *, codebooks=10, pad_id=4097, mask_pad=False
):
    """Extract undelayed [end - start, codebooks] int32 codec frames."""
    count = end - start
    total = len(delayed_rows)
    out = np.full((count, codebooks), pad_id, dtype="int32")
    arr = np.asarray(delayed_rows, dtype="int32")
    for cb in range(codebooks):
        avail = max(0, min(count, total - (start + cb)))
        if avail > 0:
            out[:avail, cb] = arr[start + cb : start + cb + avail, cb]
    out[:, 0] = np.clip(out[:, 0], 0, pad_id - 2)
    if mask_pad:
        rem = out[:, 1:]
        rem[rem >= pad_id - 1] = -1
    else:
        out[:, 1:] = np.clip(out[:, 1:], 0, pad_id - 2)
    return out


@dataclass
class _StreamRequest:
    text: str
    seed: int
    temperature: float
    max_tokens: int
    stream_decode: str
    stream_codebooks: int
    first_only: bool
    on_chunk: object
    on_done: object
    on_error: object
    cancelled: Event = field(default_factory=Event)
    prompt: np.ndarray | None = None
    slot: int = -1
    rng: object = None
    lm_pos: int = 0
    step: int = 0
    remain: int = 0
    delayed_rows: list = field(default_factory=list)
    last_frame: np.ndarray | None = None
    eos_frame: int | None = None
    countdown: int = -1
    finished: bool = False
    emitted: int = SILENCE_FRAMES
    produced: int = 7350
    skip: int = SILENCE_FRAMES * SAMPLES_PER_FRAME - 7350

    def take_samples(self, raw):
        arr = np.asarray(raw, dtype="float32")
        self.produced += len(arr)
        drop = min(len(arr), self.skip)
        self.skip -= drop
        return arr[drop:]

    def complete_frames(self, codebooks):
        complete = len(self.delayed_rows) - (codebooks - 1)
        if self.eos_frame is not None:
            complete = min(complete, self.eos_frame)
        return max(SILENCE_FRAMES, complete)

    def ready_frames(self, codebooks, pad_id):
        complete = self.complete_frames(codebooks)
        if self.stream_codebooks >= codebooks or (
            self.first_only and self.emitted > SILENCE_FRAMES
        ):
            return complete
        ready = len(self.delayed_rows) - (self.stream_codebooks - 1)
        if self.eos_frame is not None:
            ready = min(ready, self.eos_frame)
        while (
            ready > max(complete, self.emitted)
            and int(self.delayed_rows[ready - 1][0]) >= pad_id - 1
        ):
            ready -= 1
        return max(self.emitted, ready)


class Zonos3StreamingRuntime:
    """Own GPU weights, 256-slot state pool, and CUDA-graphed LM + codec phases."""

    def __init__(self, checkpoint, bundle_dir, device, *, max_requests=None):
        self.checkpoint = checkpoint
        self.config = checkpoint.config
        self.device = device
        self.directory = Path(bundle_dir).resolve()
        self._resources = ExitStack()

        filename, schema = STREAM_INDEX
        manifest = json.loads((self.directory / filename).read_text())
        if (
            manifest.get("schema") != schema
            or manifest.get("provider") != device.info["provider"]
        ):
            raise ValueError("incompatible ZONOS3 streaming bundle")
        self.pool_capacity = int(manifest["pool"])
        self.max_requests = (
            self.pool_capacity
            if max_requests is None
            else min(int(max_requests), self.pool_capacity)
        )
        self.context = int(manifest["context"])
        self.prefill_chunk = int(manifest["prefill_chunk"])
        self.pool = StatePool(self.pool_capacity)

        self._mixer = checkpoint.projector.array("a.gen.wav.mixer.weight")
        self._resample = checkpoint.projector.array("a.gen.wav.resample.weight")
        self._silence_rows = [
            np.array(
                [
                    *(SILENCE[: r + 1]),
                    *([checkpoint.audio_pad_id] * (self.config.codebooks - r - 1)),
                ],
                dtype="int32",
            )
            for r in range(SILENCE_FRAMES)
        ]

        self._prefill = {}
        self._decode = {}
        self._codec = {}
        self._states = {}
        try:

            def read_phase(record):
                path = artifact_path(
                    self.directory,
                    {
                        "path": f"{record['directory']}/forward.json",
                        "sha256": record["sha256"],
                    },
                )
                phase = read_forward(path)
                if phase["provider"] != manifest["provider"]:
                    raise ValueError("ZONOS3 phase provider differs from its index")
                return phase

            all_manifests = []
            for rec in manifest["phases"]["prefill"].values():
                m = read_phase(rec)
                all_manifests.append(
                    ("prefill", rec["slots"], None, rec["directory"], m)
                )
            for rec in manifest["phases"]["decode"].values():
                m = read_phase(rec)
                all_manifests.append(
                    ("decode", rec["slots"], None, rec["directory"], m)
                )
            for f_str, by_slot in manifest["phases"]["codec"].items():
                for rec in by_slot.values():
                    m = read_phase(rec)
                    all_manifests.append(
                        ("codec", rec["slots"], int(f_str), rec["directory"], m)
                    )
            reset_rec = manifest["phases"]["reset"]
            reset_m = read_phase(reset_rec)

            weight_names = set()
            state_specs = {}
            for _, _, _, _, m in (
                *all_manifests,
                ("reset", reset_rec["slots"], None, reset_rec["directory"], reset_m),
            ):
                g = Graph.from_metadata(m["forward"])
                for name, spec in g.buffers.items():
                    if spec.alias_of is not None:
                        continue
                    if spec.role == "weight":
                        weight_names.add(name)
                    elif spec.role == "state":
                        state_specs.setdefault(name, spec)

            self._weights = checkpoint.upload(
                device, sorted(weight_names), context=self.context
            )
            for buf in self._weights.values():
                self._resources.callback(buf.release)
            for name, spec in state_specs.items():
                buf = device.zeros(spec.shape, spec.dtype)
                self._states[name] = buf
                self._resources.callback(buf.release)

            supplied_all = self._weights | self._states

            def open_phase(sub, m):
                g = Graph.from_metadata(m["forward"])
                sup = {k: v for k, v in supplied_all.items() if k in g.buffers}
                sample_rec = next(iter(m.get("kernels", {}).values()), {})
                descriptor = sample_rec.get("logical", sample_rec)
                artifact = descriptor.get("path", descriptor.get("artifact", ""))
                base_dir = (
                    self.directory
                    if artifact.startswith(f"{sub}/")
                    else self.directory / sub
                )
                fwd = Forward(
                    base_dir,
                    device,
                    sup,
                    manifest=m,
                    graphs=True,
                    allocate_io=True,
                )
                return self._resources.enter_context(fwd)

            self._reset_slots = int(reset_rec["slots"])
            self._reset = open_phase(reset_rec["directory"], reset_m)
            for kind, slots, frames, sub, m in all_manifests:
                fwd = open_phase(sub, m)
                if kind == "prefill":
                    self._prefill[slots] = fwd
                elif kind == "decode":
                    self._decode[slots] = fwd
                else:
                    self._codec.setdefault(frames, {})[slots] = fwd

            self._prefill_capacities = tuple(sorted(self._prefill))
            self._decode_capacities = tuple(sorted(self._decode))
            self._codec_frame_caps = tuple(sorted(self._codec))
            self._codec_slot_caps = {
                f: tuple(sorted(by_s)) for f, by_s in self._codec.items()
            }
        except BaseException:
            self.close()
            raise

    def close(self):
        self._resources.close()
        self._prefill.clear()
        self._decode.clear()
        self._codec.clear()
        self._states.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def reset_slots(self, slots):
        for i in range(0, len(slots), self._reset_slots):
            chunk = slots[i : i + self._reset_slots]
            mapping = np.full(self._reset_slots, -1, dtype="int32")
            mapping[: len(chunk)] = chunk
            write_many(self.device, ((self._reset.buffers["mapping"], mapping),))
            self._reset.run()

    def run_codec_batch(self, items):
        """Run one batched prefix-codec call on [(slot, start_pos, codes_2d), ...]."""
        if not items:
            return []
        max_f = max(len(codes) for _, _, codes in items)
        f_cap = next(f for f in self._codec_frame_caps if f >= max_f)
        s_caps = self._codec_slot_caps[f_cap]
        max_s = s_caps[-1]
        outputs = []
        for begin in range(0, len(items), max_s):
            sub = items[begin : begin + max_s]
            s_cap = next(s for s in s_caps if s >= len(sub))
            fwd = self._codec[f_cap][s_cap]

            mapping = np.full(s_cap, -1, dtype="int32")
            f_lengths = np.zeros(s_cap, dtype="int32")
            semantic = np.zeros((s_cap, f_cap), dtype="int32")
            residual = np.full(
                (s_cap, f_cap, self.config.codebooks - 1), -1, dtype="int32"
            )
            stage_pos = [
                np.zeros((s_cap, f_cap * (1 << st)), dtype="int32") for st in range(5)
            ]
            stage_act = [
                np.zeros((s_cap, f_cap * (1 << st)), dtype="int32") for st in range(5)
            ]

            for b, (slot, pos0, codes) in enumerate(sub):
                n = len(codes)
                mapping[b] = slot
                f_lengths[b] = n
                semantic[b, :n] = codes[:, 0]
                residual[b, :n] = codes[:, 1:]
                for st in range(5):
                    cpf = 1 << st
                    n_st = n * cpf
                    stage_pos[st][b, :n_st] = np.arange(
                        pos0 * cpf, (pos0 + n) * cpf, dtype="int32"
                    )
                    stage_act[st][b, :n_st] = 1

            transfers = [
                (fwd.buffers["mapping"], mapping),
                (fwd.buffers["f_lengths"], f_lengths),
                (fwd.buffers["len_16"], f_lengths * 16),
                (fwd.buffers["semantic"], semantic.reshape(-1)),
                (
                    fwd.buffers["residual_codes"],
                    residual.reshape(s_cap * f_cap, self.config.codebooks - 1),
                ),
            ]
            for st in range(5):
                transfers.append((fwd.buffers[f"pos_{st}"], stage_pos[st].reshape(-1)))
                transfers.append((fwd.buffers[f"act_{st}"], stage_act[st].reshape(-1)))
            write_many(self.device, transfers)
            fwd.run()
            pcm = fwd.buffers["pcm"].to_numpy()
            for b, (_, _, codes) in enumerate(sub):
                outputs.append(pcm[b, : len(codes) * SAMPLES_PER_FRAME].copy())
        return outputs

    def prime_slots(self, slots):
        """Reset and prime 3 silence frames across newly admitted slots."""
        if not slots:
            return
        self.reset_slots(slots)
        sil = np.repeat(
            np.asarray(SILENCE, dtype="int32")[None, :], SILENCE_FRAMES, axis=0
        )
        self.run_codec_batch([(slot, 0, sil) for slot in slots])

    def prepare_prompt(self, text, max_tokens):
        """Validate one request before it can affect a GPU batch."""
        encoded, _ = self.checkpoint.prompt_encoded_frames(text)
        if (
            len(encoded) > self.prefill_chunk
            or len(encoded) + max_tokens > self.context
        ):
            raise ValueError("ZONOS3 prompt exceeds compiled prefill/context")
        return encoded

    def prefill_jobs(self, jobs):
        """Run batched LM prefill + initial sampling across newly admitted jobs."""
        max_s = self._prefill_capacities[-1]
        chunk = self.prefill_chunk
        c = self.config
        pad_frame = np.full(
            c.codebooks + 1, self.checkpoint.audio_pad_id, dtype="int32"
        )
        pad_frame[0] = -1
        pad_frame[c.codebooks] = c.vocab - 1

        for begin in range(0, len(jobs), max_s):
            sub = jobs[begin : begin + max_s]
            s_cap = next(s for s in self._prefill_capacities if s >= len(sub))
            fwd = self._prefill[s_cap]

            frames = np.repeat(pad_frame[None, :], s_cap * chunk, axis=0).reshape(
                s_cap, chunk, c.codebooks + 1
            )
            positions = np.zeros((s_cap, chunk), dtype="int32")
            active = np.zeros((s_cap, chunk), dtype="int32")
            lengths = np.zeros(s_cap, dtype="int32")
            mapping = np.full(s_cap, -1, dtype="int32")
            steps = np.full(s_cap, -1, dtype="int32")
            temps = np.zeros(s_cap, dtype="float32")
            uniforms = np.zeros((s_cap, c.codebooks), dtype="float32")

            for b, job in enumerate(sub):
                encoded = job.prompt
                n = len(encoded)
                frames[b, :n] = encoded
                positions[b, :n] = np.arange(n, dtype="int32")
                active[b, :n] = 1
                lengths[b] = n
                mapping[b] = job.slot
                steps[b] = 0
                temps[b] = job.temperature
                uniforms[b] = job.rng.random(c.codebooks, dtype="float32")
                job.lm_pos = n

            write_many(
                self.device,
                (
                    (
                        fwd.buffers["frames"],
                        frames.reshape(s_cap * chunk, c.codebooks + 1),
                    ),
                    (fwd.buffers["positions"], positions.reshape(-1)),
                    (fwd.buffers["active"], active.reshape(-1)),
                    (fwd.buffers["lengths"], lengths),
                    (fwd.buffers["mapping"], mapping),
                    (fwd.buffers["sample_steps"], steps),
                    (fwd.buffers["temperatures"], temps),
                    (fwd.buffers["uniforms"], uniforms),
                ),
            )
            fwd.run()
            sampled = fwd.buffers["sampled_frames"].to_numpy()
            for b, job in enumerate(sub):
                self._record_sampled_row(job, sampled[b])

    def decode_jobs(self, jobs):
        """Run one batched autoregressive LM step + sampling across active jobs."""
        if not jobs:
            return
        max_s = self._decode_capacities[-1]
        c = self.config
        pad_frame = np.full(
            c.codebooks + 1, self.checkpoint.audio_pad_id, dtype="int32"
        )
        pad_frame[0] = -1
        pad_frame[c.codebooks] = c.vocab - 1

        for begin in range(0, len(jobs), max_s):
            sub = jobs[begin : begin + max_s]
            s_cap = next(s for s in self._decode_capacities if s >= len(sub))
            fwd = self._decode[s_cap]

            frames = np.repeat(pad_frame[None, :], s_cap, axis=0)
            positions = np.zeros(s_cap, dtype="int32")
            active = np.zeros(s_cap, dtype="int32")
            lengths = np.zeros(s_cap, dtype="int32")
            mapping = np.full(s_cap, -1, dtype="int32")
            steps = np.full(s_cap, -1, dtype="int32")
            temps = np.zeros(s_cap, dtype="float32")
            uniforms = np.zeros((s_cap, c.codebooks), dtype="float32")

            for b, job in enumerate(sub):
                frames[b] = job.last_frame
                positions[b] = job.lm_pos
                active[b] = 1
                lengths[b] = 1
                mapping[b] = job.slot
                steps[b] = job.step
                temps[b] = job.temperature
                uniforms[b] = job.rng.random(c.codebooks, dtype="float32")
                job.lm_pos += 1

            write_many(
                self.device,
                (
                    (fwd.buffers["frames"], frames),
                    (fwd.buffers["positions"], positions),
                    (fwd.buffers["active"], active),
                    (fwd.buffers["lengths"], lengths),
                    (fwd.buffers["mapping"], mapping),
                    (fwd.buffers["sample_steps"], steps),
                    (fwd.buffers["temperatures"], temps),
                    (fwd.buffers["uniforms"], uniforms),
                ),
            )
            fwd.run()
            sampled = fwd.buffers["sampled_frames"].to_numpy()
            for b, job in enumerate(sub):
                self._record_sampled_row(job, sampled[b])

    def _record_sampled_row(self, job, row):
        c = self.config
        eoa = c.codes
        job.last_frame = row.copy()
        job.delayed_rows.append(row[: c.codebooks].copy())
        job.step += 1
        job.remain -= 1
        finished = job.remain <= 0 or job.lm_pos >= self.context
        if job.eos_frame is None:
            cols = [i for i in range(c.codebooks) if int(row[i]) == eoa]
            if cols:
                row_idx = len(job.delayed_rows) - 1
                job.eos_frame = max(SILENCE_FRAMES, row_idx - max(cols))
                job.countdown = c.codebooks + 1
        if job.countdown > 0:
            job.countdown -= 1
            if job.countdown == 0:
                finished = True
        job.finished = finished

    def flush_jobs(self, jobs):
        """Flush final 1,470 samples for completed jobs."""
        if not jobs:
            return []
        ola_all = self._states["istft_ola_state"].to_numpy()
        rs_all = self._states["istft_rs_state"].to_numpy()
        tails = []
        for job in jobs:
            raw = flush_codec_tail(
                ola_all[job.slot], rs_all[job.slot], self._mixer, self._resample
            )
            target = max(job.complete_frames(self.config.codebooks), job.emitted)
            cap = max(0, target * SAMPLES_PER_FRAME - job.produced)
            tails.append(job.take_samples(raw[:cap]))
        return tails


class Zonos3StreamingEngine:
    """Flat-admission concurrent streaming scheduler with safe slot lifecycle."""

    def __init__(self, runtime):
        self.runtime = runtime
        self._cond = Condition()
        self._waiting = deque()
        self._active = []
        self._stopped = False

    def submit(
        self,
        text,
        *,
        seed=1000,
        temperature=0.75,
        max_tokens=384,
        stream_decode="prefix",
        stream_codebooks=1,
        first_only=True,
        on_chunk,
        on_done,
        on_error,
    ):
        seed, temperature = int(seed), float(temperature)
        max_tokens, stream_codebooks = int(max_tokens), int(stream_codebooks)
        if seed < 0 or not isfinite(temperature) or temperature < 0:
            raise ValueError("seed and temperature must be finite and non-negative")
        if not 1 <= max_tokens <= self.runtime.context:
            raise ValueError(
                "max_tokens exceeds the compiled context or is not positive"
            )
        if not 1 <= stream_codebooks <= self.runtime.config.codebooks:
            raise ValueError("stream_codebooks is out of range")
        if stream_decode not in ("prefix", "whole"):
            raise ValueError("unsupported stream_decode")
        req = _StreamRequest(
            text=text,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            stream_decode=stream_decode,
            stream_codebooks=stream_codebooks,
            first_only=bool(first_only),
            on_chunk=on_chunk,
            on_done=on_done,
            on_error=on_error,
        )
        with self._cond:
            if self._stopped:
                raise RuntimeError("streaming engine stopped")
            self._waiting.append(req)
            self._cond.notify()
        return req

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    def _free_slot(self, job):
        if job.slot >= 0:
            self.runtime.pool.free(job.slot, release=lambda _: None)
            job.slot = -1

    def run_forever(self):
        rt = self.runtime
        c = rt.config
        pad_id = rt.checkpoint.audio_pad_id
        max_chunk = 15

        while True:
            with self._cond:
                while not self._stopped and not self._waiting and not self._active:
                    self._cond.wait()
                if self._stopped:
                    remaining = [*self._active, *self._waiting]
                    self._active.clear()
                    self._waiting.clear()
                    break

                admitted = []
                while (
                    self._waiting
                    and len(self._active) + len(admitted) < rt.max_requests
                ):
                    free_slots = np.flatnonzero(~rt.pool.occupied)
                    if not len(free_slots):
                        break
                    req = self._waiting.popleft()
                    if req.cancelled.is_set():
                        continue
                    slot = int(free_slots[0])
                    rt.pool.reserve(slot, reset=lambda _: None)
                    req.slot = slot
                    req.rng = np.random.default_rng(req.seed)
                    req.remain = req.max_tokens
                    req.delayed_rows = [row.copy() for row in rt._silence_rows]
                    admitted.append(req)

            if admitted:
                good = []
                for req in admitted:
                    if req.cancelled.is_set():
                        self._free_slot(req)
                        continue
                    try:
                        req.prompt = rt.prepare_prompt(req.text, req.max_tokens)
                    except ValueError as exc:
                        self._free_slot(req)
                        req.on_error(str(exc))
                    else:
                        good.append(req)
                if good:
                    try:
                        rt.prime_slots([job.slot for job in good])
                        rt.prefill_jobs(good)
                        self._active.extend(good)
                    except Exception as exc:
                        for job in good:
                            self._free_slot(job)
                            job.on_error(str(exc))

            # Drop cancelled requests cleanly and recycle their slots safely
            still_active = []
            for job in self._active:
                if job.cancelled.is_set():
                    self._free_slot(job)
                else:
                    still_active.append(job)
            self._active = still_active
            if not self._active:
                continue

            # Group ready codec chunks by frame bucket (1, 4, 15) for high GPU utilization
            buckets = {f: [] for f in rt._codec_frame_caps}
            for job in self._active:
                if job.stream_decode == "whole" and not job.finished:
                    continue
                if job.finished:
                    target = job.complete_frames(c.codebooks)
                    avail = target - job.emitted
                    if avail <= 0:
                        continue
                    n_frames = min(max_chunk, avail)
                    mask_pad = False
                else:
                    ready = job.ready_frames(c.codebooks, pad_id)
                    need = min(max_chunk, 1 + job.emitted - SILENCE_FRAMES)
                    if ready - job.emitted < need:
                        continue
                    n_frames = min(max_chunk, ready - job.emitted)
                    mask_pad = job.stream_codebooks < c.codebooks
                f_cap = next(f for f in rt._codec_frame_caps if f >= n_frames)
                codes = undelay_codes(
                    job.delayed_rows,
                    job.emitted,
                    job.emitted + n_frames,
                    codebooks=c.codebooks,
                    pad_id=pad_id,
                    mask_pad=mask_pad,
                )
                buckets[f_cap].append((job, job.emitted, codes))
                job.emitted += n_frames

            for f_cap in rt._codec_frame_caps:
                group = buckets[f_cap]
                if not group:
                    continue
                pcm_list = rt.run_codec_batch(
                    [(job.slot, pos0, codes) for job, pos0, codes in group]
                )
                for (job, _, _), raw_pcm in zip(group, pcm_list, strict=True):
                    samples = job.take_samples(raw_pcm)
                    if len(samples):
                        job.on_chunk(samples.tobytes())

            # Flush any finished jobs whose complete frames have all been decoded
            to_flush = [
                job
                for job in self._active
                if job.finished and job.emitted >= job.complete_frames(c.codebooks)
            ]
            if to_flush:
                tails = rt.flush_jobs(to_flush)
                flushed_ids = {id(job) for job in to_flush}
                for job, tail in zip(to_flush, tails, strict=True):
                    if len(tail):
                        job.on_chunk(tail.tobytes())
                    self._free_slot(job)
                    job.on_done()
                self._active = [
                    job for job in self._active if id(job) not in flushed_ids
                ]

            # Step all active decoding jobs in one batched LM decode launch
            decoding = [job for job in self._active if not job.finished]
            if decoding:
                rt.decode_jobs(decoding)

        for job in remaining:
            self._free_slot(job)
            if not job.cancelled.is_set():
                job.on_error("server stopped")


@dataclass(frozen=True)
class StreamingSpeechService:
    engine: Zonos3StreamingEngine
    metadata: dict


@contextmanager
def streaming_service(
    checkpoint_dir, codec_path, bundle_dir, *, device=0, max_requests=256
):
    """Open native BF16 ZONOS3-Mini + keep2_plain on CUDA for concurrent streaming."""
    import tensor

    loaded = Zonos3MiniCheckpoint(checkpoint_dir, codec_path)
    with (
        tensor.Device(device) as adapter,
        Zonos3StreamingRuntime(
            loaded, bundle_dir, adapter, max_requests=max_requests
        ) as runtime,
    ):
        engine = Zonos3StreamingEngine(runtime)
        worker = Thread(
            target=engine.run_forever, daemon=True, name="akas-zonos3-stream"
        )
        worker.start()
        try:
            yield StreamingSpeechService(
                engine=engine,
                metadata={
                    "engine": "tensor",
                    "provider": adapter.info["provider"],
                    "device": dict(adapter.info),
                    "program_sha256": checksum(Path(bundle_dir) / STREAM_INDEX[0]),
                    "sample_rate_hz": loaded.config.sample_rate,
                    "max_requests": runtime.max_requests,
                    "pool_capacity": runtime.pool_capacity,
                    "context": runtime.context,
                    "decoder": "keep2_plain",
                },
            )
        finally:
            engine.stop()
            worker.join()
