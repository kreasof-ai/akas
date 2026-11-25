"""Compile and run Qwen3 ASR using AKAS Forward bundles."""

from contextlib import ExitStack
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from akas.artifacts.read import read_forward_set
from akas.execution.forward import Forward
from akas.execution.plan import ExecutionPlan
from akas.memory.transfer import write_many
from akas.models.qwen3_asr import _composition
from akas.models.qwen3_asr.audio import (
    CONV_FRAMES,
    CONV_ROWS,
    HOP,
    SAMPLE_RATE,
    WINDOW_CHUNKS,
    log_mel,
    resample,
    wav_samples,
)
from akas.models.qwen3_asr.weights import (
    ALIGNED_SUFFIX,
    HALF_SUFFIX,
    TAP_SUFFIX,
    Qwen3ASRCheckpoint,
    TOKEN_EMBEDDING,
)
from akas.text.tokenizer import Tokenizer

_LANGUAGES = dict(
    af="Afrikaans",
    ar="Arabic",
    bn="Bengali",
    bs="Bosnian",
    ca="Catalan",
    ceb="Cebuano",
    cs="Czech",
    da="Danish",
    de="German",
    el="Greek",
    en="English",
    es="Spanish",
    eu="Basque",
    fa="Persian",
    fi="Finnish",
    fil="Filipino",
    fr="French",
    hi="Hindi",
    hr="Croatian",
    hu="Hungarian",
    id="Indonesian",
    is_="Icelandic",
    it="Italian",
    ja="Japanese",
    ko="Korean",
    lt="Lithuanian",
    lv="Latvian",
    mk="Macedonian",
    ms="Malay",
    mt="Maltese",
    nb="Norwegian",
    nl="Dutch",
    pl="Polish",
    pt="Portuguese",
    ro="Romanian",
    ru="Russian",
    sq="Albanian",
    sr="Serbian",
    sv="Swedish",
    th="Thai",
    tr="Turkish",
    uk="Ukrainian",
    vi="Vietnamese",
    yue="Cantonese",
    zh="Chinese",
)
_LANGUAGES["is"] = _LANGUAGES.pop("is_")
_PROMPT_START = (
    "<|im_start|>system\n<digits><nofiller><verbatim><|im_end|>\n"
    "<|im_start|>user\n<|audio_start|>"
)
_PROMPT_END = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n"


@dataclass(frozen=True)
class Transcript:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    terminated: bool
    timings: dict | None = None


def compose(
    checkpoint,
    *,
    phase,
    rows,
    provider="webgpu",
    context=512,
    integer="q8",
    signed_decode=True,
):
    if provider != "webgpu":
        raise ValueError("Qwen3 ASR requires WebGPU")
    if phase == "frontend" and rows == CONV_ROWS:
        return _composition.frontend(checkpoint)
    if phase == "encoder" and 4 <= rows <= CONV_ROWS * WINDOW_CHUNKS and rows % 4 == 0:
        return _composition.encoder(checkpoint, rows, integer=integer)
    if phase == "decoder" and 1 <= rows <= context <= checkpoint.config.max_context:
        splits = min(4, max(1, context // 128)) if rows >= 96 else 1
        return _composition.decoder(
            checkpoint, rows, context, integer=integer, attention_splits=splits
        )
    if (
        phase in ("decode", _composition.PARTITIONED_DECODE)
        and rows == 1
        and 1 <= context <= checkpoint.config.max_context
    ):
        splits = 1 if phase == "decode" else min(16, max(1, context // 32))
        return _composition.decoder(
            checkpoint,
            rows,
            context,
            decode=True,
            signed_decode=signed_decode,
            attention_splits=splits,
        )
    if phase == "head" and rows == 1:
        return _composition.head(checkpoint, signed_decode=signed_decode)
    if phase == "feedback" and rows == 1:
        return _composition.feedback()
    raise ValueError("invalid Qwen3 ASR forward geometry")


class Qwen3ASR:
    checkpoint_type = Qwen3ASRCheckpoint
    forward_index = ("speech.json", "akas.asr-program.v1")
    compose = staticmethod(compose)

    @classmethod
    def forward_phases(
        cls,
        checkpoint,
        *,
        provider,
        context=512,
        prefill_rows=96,
        prefill_buckets=(32, 64, 80),
        encoder_chunks=(2, 4, 5, 6, 8),
        integer="q8",
        signed_decode=True,
    ):
        if (
            type(context) is not int
            or type(prefill_rows) is not int
            or not 1 < prefill_rows <= context <= checkpoint.config.max_context
            or prefill_rows % 4
            or integer not in (False, True, "q8")
            or any(type(n) is not int or n < 4 or n % 4 for n in prefill_buckets)
            or any(
                type(n) is not int or not 1 <= n <= WINDOW_CHUNKS
                for n in encoder_chunks
            )
            or WINDOW_CHUNKS not in encoder_chunks
        ):
            raise ValueError("invalid Qwen3 ASR compiled context or prefill capacity")
        shapes = (
            ("frontend", CONV_ROWS),
            *(
                ("encoder", (n * CONV_ROWS + 3) // 4 * 4)
                for n in sorted(set(encoder_chunks))
            ),
            *(
                ("decoder", n)
                for n in sorted(
                    {prefill_rows, *(n for n in prefill_buckets if n < prefill_rows)}
                )
            ),
            ("decode", 1),
            *(
                [(_composition.PARTITIONED_DECODE, 1)]
                if context >= 2 * _composition.PARTITION_MIN_PREFIX
                else []
            ),
            ("head", 1),
            ("feedback", 1),
        )
        return dict(context=context), (
            (
                phase,
                rows,
                cls.compose(
                    checkpoint,
                    phase=phase,
                    rows=rows,
                    context=context,
                    provider=provider,
                    integer=integer,
                    signed_decode=signed_decode,
                ),
            )
            for phase, rows in shapes
        )

    def __init__(self, checkpoint, directory, device):
        self.checkpoint, self.config, self.device = (
            checkpoint,
            checkpoint.config,
            device,
        )
        self.tokenizer = Tokenizer.from_gguf(checkpoint.backbone.metadata)
        self.stop = (self.tokenizer.eos, self.tokenizer.ids["<|endoftext|>"])
        self._prefix = self._token_ids(_PROMPT_START)
        self._resources = ExitStack()
        filename, schema = self.forward_index
        phases = read_forward_set(
            directory, filename, schema=schema, provider=device.info["provider"]
        )
        self.prefill_rows = max(phases["decoder"])
        self.decoder_rows = sorted(phases["decoder"])
        self.encoder_rows = sorted(phases["encoder"])
        self.fused_decode = "decode" in phases
        definitions = {
            (name, rows): manifest
            for name, variants in phases.items()
            for rows, manifest in variants.items()
        }
        shared = {}
        for manifest in definitions.values():
            for spec in manifest["forward"]["buffers"]:
                if spec["role"] not in ("state", "weight") or spec.get("alias_of"):
                    continue
                if shared.setdefault(spec["name"], spec) != spec:
                    raise ValueError(
                        f"incompatible Qwen3 ASR shared buffer: {spec['name']}"
                    )
        self.context = shared["rope"]["shape"][0]
        try:
            supplied = checkpoint.upload(
                device,
                sorted(
                    name
                    for name in shared
                    if name in checkpoint.tensors
                    or name.endswith((HALF_SUFFIX, TAP_SUFFIX, ALIGNED_SUFFIX))
                ),
            )
            for buffer in supplied.values():
                self._resources.callback(buffer.release)
            constants = self._constants()
            for name, spec in shared.items():
                if name in supplied:
                    continue
                if spec["role"] == "state":
                    buffer = device.zeros(spec["shape"], spec["dtype"])
                else:
                    buffer = device.from_numpy(constants[name])
                supplied[name] = buffer
                self._resources.callback(buffer.release)
            self._forwards = {}
            feedback_manifest = definitions.pop(("feedback", 1), None)
            for key, manifest in definitions.items():
                names = {spec["name"] for spec in manifest["forward"]["buffers"]}
                forward = Forward(
                    directory,
                    device,
                    {name: supplied[name] for name in supplied.keys() & names},
                    manifest=manifest,
                    graphs=True,
                    allocate_io=True,
                )
                self._forwards[key] = self._resources.enter_context(forward)
            self._chunk_features = self._feature_view("frontend", CONV_ROWS)
            self._encoder_features = {
                rows: self._feature_view("encoder", rows) for rows in self.encoder_rows
            }
            self._feedback_plans = {}
            if feedback_manifest is not None:
                self._install_feedback("decode", directory, feedback_manifest, supplied)
                phase = _composition.PARTITIONED_DECODE
                if (phase, 1) in self._forwards:
                    self._install_feedback(
                        phase, directory, feedback_manifest, supplied
                    )
            prefix = self._embeddings(self._prefix)
            for start in range(0, len(prefix), self.prefill_rows):
                self._advance(prefix[start : start + self.prefill_rows], start)
        except BaseException:
            self.close()
            raise

    def _constants(self):
        dim = self.config.head_dim
        frequency = self.config.rope_theta ** (-np.arange(0, dim, 2) / dim)
        angle = np.arange(self.context)[:, None] * frequency
        return dict(
            rope=np.concatenate((np.cos(angle), np.sin(angle)), axis=1).astype(
                "float32"
            ),
            rope_origin=np.concatenate(
                (np.ones((1, dim // 2)), np.zeros((1, dim // 2))), axis=1
            ).astype("float32"),
            sinusoid=self.checkpoint.projector.array("a.position_embd.weight")[
                :CONV_ROWS
            ].copy(),
            head_active=np.ones(1, dtype="int32"),
            decode_stop_ids=np.array(self.stop, dtype="int32"),
        )

    def _install_feedback(self, phase, directory, manifest, supplied):
        decode = self._forwards[phase, 1]
        bindings = {
            name: decode.buffers[name] for name in _composition.DECODE_FEEDBACK_BUFFERS
        }
        for name in ("decode_journal", "decode_stop_ids"):
            bindings[name] = supplied[name]
        feedback = self._resources.enter_context(
            Forward(directory, self.device, bindings, manifest=manifest)
        )
        self._decode_journal = feedback.buffers["decode_journal"]
        calls = (*decode.plan, *feedback.plan)
        resources = (
            *decode.buffers.values(),
            *decode.kernels.values(),
            *feedback.buffers.values(),
            *feedback.kernels.values(),
        )
        capacity = self._decode_journal.shape[1] - 1
        plans = self._feedback_plans[phase] = {}
        for steps in range(1, capacity + 1):
            plan = ExecutionPlan(self.device, calls * steps)
            prepared = plan.capture(resources=resources)
            self._resources.callback(prepared.close)
            plans[steps] = prepared

    def _feature_view(self, phase, rows):
        features = self._forwards[phase, rows].buffers["features"]
        view = features.view((1, rows * self.config.audio_width))
        self._resources.callback(view.release)
        return view

    def _run(self, phase, rows, output, **inputs):
        forward = self._forwards[phase, rows]
        write_many(
            self.device,
            ((forward.buffers[name], value) for name, value in inputs.items()),
        )
        if output is None:
            forward.run()
            return
        return forward.graph.launch(readback=forward.buffers[output])

    def _encode(self, mel, mark, profile):
        frames = CONV_FRAMES * WINDOW_CHUNKS
        streams = []
        for i in range(0, len(mel), frames):
            window = mel[i : i + frames]
            count = (len(window) + CONV_FRAMES - 1) // CONV_FRAMES * CONV_ROWS
            capacity = next(rows for rows in self.encoder_rows if rows >= count)
            positions = np.arange(capacity, dtype="int32")
            features = self._encoder_features[capacity]
            start = perf_counter()
            for first in range(0, len(window), CONV_FRAMES):
                chunk = window[first : first + CONV_FRAMES]
                if len(chunk) < CONV_FRAMES:
                    chunk = np.pad(chunk, ((0, CONV_FRAMES - len(chunk)), (0, 0)))
                row = first // CONV_FRAMES * CONV_ROWS
                self._run("frontend", CONV_ROWS, None, mel=chunk[:, :, None])
                features.copy_columns_from(
                    self._chunk_features, row * self.config.audio_width
                )
            if profile:
                self.device.synchronize()
            mark("audio_frontend", start)
            # Match llama.cpp b12521's padded 100-frame convolution chunks.
            start = perf_counter()
            streams.append(
                self._run(
                    "encoder",
                    capacity,
                    "audio_embedding",
                    positions=positions,
                    limits=np.full(capacity, count - 1, dtype="int32"),
                    active=(positions < count).astype("int32"),
                )[:count]
            )
            mark("audio_transformer", start)
        return np.concatenate(streams)

    def _token_ids(self, text):
        return self.tokenizer.encode(text, add_bos=False, parse_special=True)

    def _embeddings(self, ids):
        return np.stack(
            [self.checkpoint.backbone.row(TOKEN_EMBEDDING, token) for token in ids]
        )

    def _advance(self, embeddings, start, *, read_hidden=True):
        count = len(embeddings)
        rows = next(rows for rows in self.decoder_rows if rows >= count)
        pos = np.arange(rows, dtype="int32")
        active = pos < count
        hidden = self._run(
            "decoder",
            rows,
            "hidden" if read_hidden else None,
            embedding=np.pad(embeddings, ((0, rows - count), (0, 0))),
            positions=np.where(active, start + pos, 0),
            active=active.astype("int32"),
        )
        if read_hidden:
            return hidden[count - 1 : count]

    def _decode_blocks(self, first, position, limit, mark):
        if first in self.stop:
            return []
        output = [first]
        if limit == 1:
            return output
        start = perf_counter()
        partitioned = _composition.PARTITIONED_DECODE
        phase = (
            partitioned
            if position >= _composition.PARTITION_MIN_PREFIX
            and partitioned in self._feedback_plans
            else "decode"
        )
        buffers = self._forwards[phase, 1].buffers
        write_many(
            self.device,
            (
                (buffers["previous_token"], np.array([first], dtype="int32")),
                (buffers["positions"], np.array([position], dtype="int32")),
                (buffers["active"], np.ones(1, dtype="int32")),
            ),
        )
        journal = self._decode_journal
        empty = np.zeros(journal.shape, dtype="int32")
        plans = self._feedback_plans[phase]
        capacity = max(plans)
        while len(output) < limit:
            steps = min(capacity, limit - len(output))
            write_many(self.device, ((journal, empty),))
            record = plans[steps].launch(readback=journal)[0]
            count = int(record[0])
            output.extend(record[1 : count + 1].tolist())
            if count < steps:
                break
        mark("decode", start)
        return output

    def _decode_tokens(self, token, prompt_tokens, limit, mark):
        if self._feedback_plans:
            return self._decode_blocks(token, prompt_tokens, limit, mark)
        output = []
        for index in range(limit):
            if token in self.stop:
                break
            output.append(token)
            if index + 1 < limit:
                start = perf_counter()
                position = prompt_tokens + index
                if self.fused_decode:
                    token = int(
                        self._run(
                            "decode",
                            1,
                            "token",
                            previous_token=np.array([token], dtype="int32"),
                            positions=np.array([position], dtype="int32"),
                            active=np.ones(1, dtype="int32"),
                        )[0]
                    )
                    mark("decode", start)
                else:
                    hidden = self._advance(self._embeddings((token,)), position)
                    mark("decode", start)
                    start = perf_counter()
                    token = int(self._run("head", 1, "token", hidden=hidden)[0])
                    mark("head", start)
        return output

    def transcribe(self, wav, *, language, limit=128, profile=False):
        if language not in self.config.languages or language not in _LANGUAGES:
            raise ValueError("unsupported Qwen3 ASR language")
        if type(limit) is not int or not 1 <= limit <= self.context:
            raise ValueError("invalid Qwen3 ASR output token limit")
        timings = {} if profile else None

        def mark(stage, start):
            if timings is not None:
                timings[stage] = perf_counter() - start + timings.get(stage, 0)

        start = perf_counter()
        samples, rate = wav_samples(wav)
        suffix = self._token_ids(
            f"{_PROMPT_END}language {_LANGUAGES[language]}<asr_text>"
        )
        resampled_length = (len(samples) * SAMPLE_RATE + rate - 1) // rate
        n_frames = max(resampled_length, 201) // HOP + 1
        audio_tokens = (n_frames + CONV_FRAMES - 1) // CONV_FRAMES * CONV_ROWS
        if len(self._prefix) + audio_tokens + len(suffix) + limit > self.context:
            raise ValueError("audio and requested output exceed the compiled context")
        features = log_mel(resample(samples, rate), self.config.mels)
        mark("audio_features", start)
        audio = self._encode(features, mark, profile)
        prefill_start = perf_counter()
        prompt = np.concatenate((audio, self._embeddings(suffix)))
        prefix_tokens = len(self._prefix)
        prompt_tokens = prefix_tokens + len(prompt)
        for start in range(0, len(prompt), self.prefill_rows):
            hidden = self._advance(
                prompt[start : start + self.prefill_rows],
                prefix_tokens + start,
                read_hidden=start + self.prefill_rows >= len(prompt),
            )
        mark("prefill", prefill_start)
        start = perf_counter()
        token = int(self._run("head", 1, "token", hidden=hidden)[0])
        mark("head", start)
        output = self._decode_tokens(token, prompt_tokens, limit, mark)
        text = self.tokenizer.decode(output, skip_special=True).strip()
        return Transcript(
            text, tuple(output), prompt_tokens, len(output) < limit, timings
        )

    def runtime_metadata(self):
        """Describe the audio input and the capacities of the loaded program."""
        return dict(
            sample_rate_hz=SAMPLE_RATE,
            languages=[lang for lang in self.config.languages if lang in _LANGUAGES],
            context=self.context,
            prefill_rows=self.prefill_rows,
            prefill_buckets=list(self.decoder_rows),
            encoder_buckets=list(self.encoder_rows),
            cached_prefix_tokens=len(self._prefix),
            decode_block_steps=max(self._feedback_plans.get("decode", {}), default=1),
        )

    def close(self):
        self._resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
