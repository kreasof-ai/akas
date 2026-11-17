"""Read the packed ZONOS3-Mini backbone and F32 projector."""

from collections import Counter
from contextlib import ExitStack
from functools import cached_property
from math import prod
from types import SimpleNamespace

from akas.checkpoints.gguf import GGUF
from akas.models.zonos3.config import Zonos3MiniConfig


CODEBOOK_HEAD = "a.gen.code.head.weight"
CODEBOOK_EMBEDDING = "a.gen.code.out_embd.weight"
TOKEN_EMBEDDING = "token_embd.weight"


def _backbone_tensors(c):
    """Expected logical shapes and GGML types for the supported checkpoint."""
    width = c.width
    shapes = {
        TOKEN_EMBEDDING: ((c.vocab, width), 8),
        "output_norm.weight": ((width,), 0),
        "output_res_score.weight": ((width,), 0),
    }
    for index, recurrent in enumerate(c.recurrent):
        prefix = f"blk.{index}."
        layer = {
            name + ".weight": ((width,), 0)
            for name in ("attn_norm", "attn_res_score", "ffn_norm", "ffn_res_score")
        }
        if recurrent:
            delta = c.delta_width
            layer.update(
                {
                    "attn_qkv.weight": ((3 * delta, width), 8),
                    "attn_gate.weight": ((delta, width), 8),
                    "ssm_alpha.weight": ((c.delta_heads, width), 8),
                    "ssm_beta.weight": ((c.delta_heads, width), 8),
                    "ssm_conv1d.weight": ((3 * delta, c.conv_width), 0),
                    "ssm_dt.bias": ((c.delta_heads,), 0),
                    "ssm_a": ((c.delta_heads,), 0),
                    "ssm_norm.weight": ((c.delta_dim,), 0),
                    "ssm_out.weight": ((width, delta), 8),
                }
            )
        else:
            query, key = c.heads * c.head_dim, c.kv_heads * c.head_dim
            layer.update(
                {
                    "attn_q.weight": ((query, width), 8),
                    "attn_k.weight": ((key, width), 8),
                    "attn_v.weight": ((key, width), 8),
                    "attn_q_norm.weight": ((c.heads, c.head_dim), 0),
                    "attn_gate.weight": ((c.heads, width), 8),
                    "attn_output.weight": ((width, query), 8),
                }
            )
        if index < c.dense_layers:
            layer.update(
                {
                    "ffn_gate.weight": ((c.dense_width, width), 8),
                    "ffn_up.weight": ((c.dense_width, width), 8),
                    "ffn_down.weight": ((width, c.dense_width), 8),
                }
            )
        else:
            expert_in = (c.experts, c.expert_width, c.latent_width)
            layer.update(
                {
                    "ffn_gate_inp.weight": ((c.experts, width), 0),
                    "exp_probs_b.bias": ((c.experts,), 0),
                    "ffn_latent_down.weight": ((c.latent_width, width), 8),
                    "ffn_latent_up.weight": ((width, c.latent_width), 8),
                    "ffn_gate_exps.weight": (expert_in, 8),
                    "ffn_up_exps.weight": (expert_in, 8),
                    "ffn_down_exps.weight": (
                        (c.experts, c.latent_width, c.expert_width),
                        8,
                    ),
                }
            )
        shapes.update((prefix + name, spec) for name, spec in layer.items())
    return shapes


def _validate(gguf, expected):
    for name, (shape, encoding) in expected.items():
        info = gguf.tensors.get(name)
        if info is None or (info.shape, info.type) != (shape, encoding):
            raise ValueError(f"incompatible ZONOS3 tensor: {name}")


class Zonos3MiniCheckpoint:
    """The matched GGUF pair for the ZONOS3-Mini Q8_0 speech model."""

    def __init__(self, backbone, projector):
        self.backbone = GGUF(backbone)
        self.projector = GGUF(projector)
        self.config = c = Zonos3MiniConfig.from_gguf(self.backbone.metadata)
        if self.backbone.metadata.get("general.file_type") != 7:
            raise ValueError("requires the ZONOS3-Mini Q8_0 backbone")
        _validate(self.backbone, _backbone_tensors(c))
        metadata = self.projector.metadata
        if (
            metadata.get("general.architecture") != "clip"
            or metadata.get("clip.gen.audio.projector_type") != "zonos3_gen"
            or metadata.get("clip.gen.audio.model_variant") != "keep2_plain"
        ):
            raise ValueError("requires the ZONOS3-Mini keep2_plain projector")
        shape = (c.codebooks * c.code_vocab, c.width)
        _validate(
            self.projector, {CODEBOOK_HEAD: (shape, 0), CODEBOOK_EMBEDDING: (shape, 0)}
        )

    def source(self, name):
        return self.backbone if name in self.backbone.tensors else self.projector

    @cached_property
    def tokenizer(self):
        from akas.text.tokenizer import Tokenizer

        return Tokenizer.from_gguf(self.backbone.metadata)

    def prompt_ids(self, text):
        """Encode normalized text with the default conditioning and boundaries."""
        if not text.strip():
            raise ValueError("ZONOS3 speech input is empty")
        tokenizer = self.tokenizer
        conditions = ("loudness_unk", "fmax_unk", "speaking_rate_unk")
        return [
            *(tokenizer.ids[f"<|{name}|>"] for name in conditions),
            *tokenizer.encode(text, add_bos=True),
            tokenizer.eos,
        ]

    @cached_property
    def default_voices(self):
        """Select the first shipped preset for each language, preferring the GGUF default."""
        from akas.models.zonos3.config import LANGUAGES

        default = self.projector.metadata.get("clip.gen.audio.default_voice")
        voices = {}
        for name in self.projector.tensors:
            voice = name.removeprefix("voice.")
            if voice == name:
                continue
            tag = voice.split("_", 2)
            if len(tag) < 3:
                continue
            language = tag[1].partition("-")[0]
            if language in LANGUAGES and (voice == default or language not in voices):
                voices[language] = voice
        return voices

    def voice_rows(self, voice=None, *, lang="en"):
        if voice is None:
            voice = self.default_voices.get(
                lang, self.projector.metadata.get("clip.gen.audio.default_voice")
            )
        name = f"voice.{voice}"
        info = self.projector.tensors.get(name)
        if (
            info is None
            or info.type != 0
            or len(info.shape) != 2
            or info.shape[1] != self.config.width
        ):
            raise ValueError(f"invalid ZONOS3 predefined voice: {voice}")
        return self.projector.array(name)

    def graph_weight(self, graph, name, *, provider):
        from akas.quantization.weights.ggml.definition import weight_storage

        info = self.source(name).tensors[name]
        count, dtype = weight_storage(
            info.type, prod(info.shape), packed_words=provider == "webgpu"
        )
        return graph.buffer(name, (count,), dtype, role="weight")

    def upload(self, device, names, *, context=None):
        import numpy as np

        from akas.checkpoints.upload import upload_packed

        names = tuple(names)
        regular = [
            name
            for name in names
            if name not in ("rope", "unit.weight", "head_unit.weight")
            and not name.endswith(".ssm_a_log")
        ]
        weights = {}
        with ExitStack() as cleanup:
            for source in (self.backbone, self.projector):
                tensors = {
                    name: source.tensors[name]
                    for name in regular
                    if self.source(name) is source
                }
                if tensors:
                    loaded = upload_packed(
                        device,
                        SimpleNamespace(
                            tensors=tensors,
                            packed=source.packed,
                            discard=source.discard,
                        ),
                    )
                    for value in loaded.values():
                        cleanup.callback(value.release)
                    weights.update(loaded)
            for name in names:
                if name in weights:
                    continue
                if name == "unit.weight":
                    value = np.ones(self.config.width, dtype="float32")
                elif name == "head_unit.weight":
                    value = np.ones(self.config.head_dim, dtype="float32")
                elif name == "rope":
                    c = self.config
                    if type(context) is not int or not 1 <= context <= c.max_context:
                        raise ValueError(
                            "ZONOS3 rotary upload requires a valid context"
                        )
                    frequency = c.rope_theta ** (
                        -np.arange(0, c.head_dim, 2) / c.head_dim
                    )
                    angle = np.arange(context)[:, None] * frequency
                    value = np.concatenate(
                        (np.cos(angle), np.sin(angle)), axis=1
                    ).astype("float32")
                else:
                    decay = self.backbone.array(name.removesuffix("_log"))
                    if not np.all(np.isfinite(decay) & (decay < 0)):
                        raise ValueError(f"invalid ZONOS3 GDN decay: {name}")
                    value = np.log(-decay)
                weights[name] = device.from_numpy(value)
                cleanup.callback(weights[name].release)
            cleanup.pop_all()
        return weights

    def inventory(self):
        return {
            label: dict(Counter(info.encoding for info in source.tensors.values()))
            for label, source in (
                ("backbone", self.backbone),
                ("projector", self.projector),
            )
        }
