"""Model geometry for the Qwen3 ASR decoder and audio encoder in GGUF."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Qwen3ASRConfig:
    width: int
    vocab: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ff: int
    max_context: int
    epsilon: float
    rope_theta: float
    audio_width: int
    audio_layers: int
    audio_heads: int
    audio_ff: int
    audio_epsilon: float
    mels: int
    languages: tuple[str, ...]

    @classmethod
    def from_gguf(cls, text, audio):
        if (
            text.get("general.architecture") != "qwen3vl"
            or audio.get("general.architecture") != "clip"
            or not audio.get("clip.has_audio_encoder")
            or audio.get("clip.audio.projector_type") != "qwen3a"
            or text.get("qwen3vl.n_deepstack_layers", 0) != 0
        ):
            raise ValueError(
                "requires a Qwen3 ASR qwen3vl decoder and qwen3a projector"
            )

        def positive(table, key, kind=int):
            value = table.get(key)
            if type(value) is not kind or not isfinite(value) or value <= 0:
                raise ValueError(f"invalid Qwen3 ASR metadata: {key}")
            return value

        def decoder(key, kind=int):
            return positive(text, "qwen3vl." + key, kind)

        def encoder(key, kind=int):
            return positive(audio, "clip.audio." + key, kind)

        settings = cls(
            width=decoder("embedding_length"),
            vocab=len(text["tokenizer.ggml.tokens"]),
            layers=decoder("block_count"),
            heads=decoder("attention.head_count"),
            kv_heads=decoder("attention.head_count_kv"),
            head_dim=decoder("attention.key_length"),
            ff=decoder("feed_forward_length"),
            max_context=decoder("context_length"),
            epsilon=decoder("attention.layer_norm_rms_epsilon", float),
            rope_theta=decoder("rope.freq_base", float),
            audio_width=encoder("embedding_length"),
            audio_layers=encoder("block_count"),
            audio_heads=encoder("attention.head_count"),
            audio_ff=encoder("feed_forward_length"),
            audio_epsilon=encoder("attention.layer_norm_epsilon", float),
            mels=encoder("num_mel_bins"),
            languages=tuple(text.get("general.languages", ())),
        )
        if (
            settings.width != encoder("projection_dim")
            or settings.head_dim != decoder("attention.value_length")
            or settings.head_dim % 2
            or settings.heads % settings.kv_heads
            or settings.audio_width % settings.audio_heads
            or settings.mels != 128
            or not settings.vocab
        ):
            raise ValueError("unsupported Qwen3 ASR encoder or decoder geometry")
        return settings
