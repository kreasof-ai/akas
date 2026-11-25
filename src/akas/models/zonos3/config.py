"""Geometry of the ZONOS3-Mini backbone and its audio codebooks."""

from dataclasses import dataclass
import math


SILENCE = (1918, 746, 3544, 234, 1267, 2914, 3944, 2849, 326, 1337)
SILENCE_FRAMES = 3


# Model card language coverage for already-normalized input.
LANGUAGES = (
    "en",
    "fr",
    "es",
    "de",
    "it",
    "pt",
    "nl",
    "el",
    "sv",
    "da",
    "nb",
    "fi",
    "pl",
    "cs",
    "sk",
    "hu",
    "ro",
    "bg",
    "ru",
    "uk",
    "tr",
    "ar",
    "he",
    "ka",
    "zh",
    "ja",
    "ko",
    "vi",
    "id",
    "ms",
    "th",
    "tl",
    "hi",
    "bn",
    "ta",
    "te",
    "mr",
    "gu",
    "kn",
    "ml",
    "pa",
    "or",
    "ur",
)


@dataclass(frozen=True)
class Zonos3MiniConfig:
    width: int
    vocab: int
    max_context: int
    recurrent: tuple[bool, ...]
    epsilon: float
    heads: int
    kv_heads: int
    head_dim: int
    rope_theta: float
    conv_width: int
    delta_heads: int
    delta_dim: int
    dense_layers: int
    dense_width: int
    experts: int
    experts_per_row: int
    expert_width: int
    latent_width: int
    condition_buckets: tuple[int, ...] = ()
    router_norm: bool = False

    codebooks = 10
    codes = 4096
    sample_rate = 44100
    samples_per_frame = 2940
    logit_softcap = 15.0

    @property
    def code_vocab(self):
        return self.codes + 2

    @property
    def delta_width(self):
        return self.delta_heads * self.delta_dim

    @classmethod
    def from_gguf(cls, metadata):
        if metadata.get("general.architecture") != "zonos3":
            raise ValueError("requires a zonos3 backbone GGUF")

        def positive(key):
            value = metadata.get("zonos3." + key)
            if type(value) is not int or value < 1:
                raise ValueError(f"invalid ZONOS3 metadata: {key}")
            return value

        layers = positive("block_count")
        interval = positive("full_attention_interval")
        recurrent = metadata.get("zonos3.attention.recurrent_layers")
        if recurrent is None:
            recurrent = [(i + 1) % interval != 0 for i in range(layers)]
        if (
            not isinstance(recurrent, list)
            or len(recurrent) != layers
            or any(type(flag) is not bool for flag in recurrent)
        ):
            raise ValueError("invalid ZONOS3 recurrent layer map")
        vocabulary = metadata.get("tokenizer.ggml.tokens")
        if not isinstance(vocabulary, list) or not vocabulary:
            raise ValueError("ZONOS3 backbone is missing its text vocabulary")
        epsilon = metadata.get("zonos3.attention.layer_norm_rms_epsilon")
        theta = metadata.get("zonos3.rope.freq_base")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (epsilon, theta)
        ):
            raise ValueError("invalid ZONOS3 normalization or rotary constants")
        config = cls(
            width=positive("embedding_length"),
            vocab=len(vocabulary),
            max_context=positive("context_length"),
            recurrent=tuple(recurrent),
            epsilon=epsilon,
            heads=positive("attention.head_count"),
            kv_heads=positive("attention.head_count_kv"),
            head_dim=positive("attention.key_length"),
            rope_theta=theta,
            conv_width=positive("ssm.conv_kernel"),
            delta_heads=positive("ssm.time_step_rank"),
            delta_dim=positive("ssm.state_size"),
            dense_layers=positive("leading_dense_block_count"),
            dense_width=positive("feed_forward_length"),
            experts=positive("expert_count"),
            experts_per_row=positive("expert_used_count"),
            expert_width=positive("expert_feed_forward_length"),
            latent_width=positive("moe_latent_size"),
        )
        if (
            config.heads % config.kv_heads
            or config.head_dim != positive("attention.value_length")
            or config.head_dim != positive("rope.dimension_count")
            or config.head_dim % 2
            or config.delta_heads != positive("ssm.group_count")
            or config.delta_width != positive("ssm.inner_size")
            or config.dense_layers > layers
            or config.experts_per_row > config.experts
        ):
            raise ValueError("unsupported ZONOS3 attention or expert geometry")
        return config

    @classmethod
    def from_zonos3(cls, metadata):
        """Parse native ZONOS3-Mini params.json model geometry."""
        model = metadata.get("model", metadata) if isinstance(metadata, dict) else None
        if not isinstance(model, dict) or model.get("model_type") != "zonos2":
            raise ValueError("requires native ZONOS3 model parameters")
        layers = model.get("n_layers")
        pattern = model.get("attention_pattern")
        if type(layers) is not int or layers < 1 or pattern != "gdn3_global1":
            raise ValueError("unsupported ZONOS3 layer schedule")
        recurrent = tuple((i + 1) % 4 != 0 for i in range(layers))
        width = int(model["dim"])
        head_dim = int(model["head_dim"])
        heads = int(model["n_heads"])
        kv_heads = int(model["n_kv_heads"])
        multiple = int(model.get("multiple_of", 64))
        moe_multiple = int(model.get("moe_multiple_of", 64))
        dense_raw = int(width * float(model["ffn_dim_multiplier"]))
        dense_width = multiple * ((dense_raw + multiple - 1) // multiple)
        expert_raw = int(width * float(model["moe_ffn_dim_multiplier"]))
        expert_width = moe_multiple * ((expert_raw + moe_multiple - 1) // moe_multiple)
        latent_width = int(round(width * float(model["moe_alpha"])))
        spec = model.get("text_tokenizer_spec") or {}
        text_vocab = int(model["text_vocab"])
        if spec.get("kind") != "superbpe" or text_vocab != int(
            spec.get("vocab_size", 0)
        ) + int(spec.get("offset", 0)):
            raise ValueError("unsupported ZONOS3 text tokenizer specification")
        buckets = tuple(int(n) for n in model.get("condition_bucket_counts", ()))
        if int(model.get("condition_count", 0)) != len(buckets):
            raise ValueError("inconsistent ZONOS3 condition bucket counts")
        if (
            int(model["n_codebooks"]) != cls.codebooks
            or int(model["codebook_size"]) != cls.codes
            or model.get("residual_type") != "full_attnres"
            or model.get("moe_router_type") != "linear"
            or bool(model.get("moe_router_use_eda", False))
        ):
            raise ValueError(
                "unsupported ZONOS3 codebook, residual or router configuration"
            )
        return cls(
            width=width,
            vocab=text_vocab + 1 + int(model.get("text_extra_tokens", 0)),
            max_context=int(model["max_seqlen"]),
            recurrent=recurrent,
            epsilon=float(model["norm_eps"]),
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=float(model["rope_theta"]),
            conv_width=4,
            delta_heads=4,
            delta_dim=128,
            dense_layers=int(model.get("moe_start_from_layer", 1)),
            dense_width=dense_width,
            experts=int(model["moe_n_experts"]),
            experts_per_row=int(model["moe_router_topk"]),
            expert_width=expert_width,
            latent_width=latent_width,
            condition_buckets=buckets,
            router_norm=True,
        )
