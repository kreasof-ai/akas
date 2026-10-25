"""Spark X2.5/X2.6 hybrid sliding/full attention architecture."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SparkConfig:
    width: int
    ff: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: tuple[str, ...]
    vocab: int
    epsilon: float
    max_context: int
    window: int
    theta: tuple[float, ...]
    rotary: tuple[int, ...]
    tied: bool = True

    def __post_init__(self):
        if (
            any(
                type(x) is not int or x <= 0
                for x in (
                    self.width,
                    self.ff,
                    self.heads,
                    self.kv_heads,
                    self.head_dim,
                    self.vocab,
                    self.max_context,
                    self.window,
                )
            )
            or self.heads % self.kv_heads
            or self.head_dim not in (64, 128, 256)
            or not self.layers
            or set(self.layers) - {"sliding_attention", "full_attention"}
            or len(self.theta) != len(self.layers)
            or len(self.rotary) != len(self.layers)
            or any(
                type(x) is not int or x <= 0 or x > self.head_dim or x % 2
                for x in self.rotary
            )
            or any(not math.isfinite(x) or x <= 0 for x in (*self.theta, self.epsilon))
        ):
            raise ValueError("unsupported Spark attention geometry")

    @classmethod
    def from_huggingface(cls, data):
        if data.get("model_type") != "spark2_5":
            raise ValueError("requires Spark2_5 configuration")
        if (
            data.get("hidden_act") != "gelu"
            or data.get("attention_bias", False)
            or data.get("mlp_bias", False)
            or not data.get("headwise_attn_output_gate", False)
            or data.get("gate_attn_act_mode") != "sigmoid"
            or data.get("rope_scaling")
        ):
            raise ValueError(
                "unsupported Spark activation, bias or rotary configuration"
            )
        layers = tuple(data["layer_types"])
        if len(layers) != data["num_hidden_layers"]:
            raise ValueError("Spark layer_types length differs from num_hidden_layers")
        rope = data["rope_parameters"]
        if any(p.get("rope_type", "default") != "default" for p in rope.values()):
            raise ValueError("Spark requires default RoPE")
        d = data["head_dim"]
        if any(
            d * rope[k]["partial_rotary_factor"]
            != int(d * rope[k]["partial_rotary_factor"])
            for k in layers
        ):
            raise ValueError("Spark rotary dimensions must be integral")
        return cls(
            data["hidden_size"],
            data["intermediate_size"],
            data["num_attention_heads"],
            data["num_key_value_heads"],
            d,
            layers,
            data["vocab_size"],
            data["rms_norm_eps"],
            data["max_position_embeddings"],
            data["sliding_window"],
            tuple(rope[k]["rope_theta"] for k in layers),
            tuple(int(d * rope[k]["partial_rotary_factor"]) for k in layers),
            data.get("tie_word_embeddings", False),
        )

    @classmethod
    def from_gguf(cls, gguf):
        m = gguf.metadata
        if m.get("general.architecture") != "spark2_5":
            raise ValueError("requires spark2_5 GGUF architecture")
        get = lambda key: m["spark2_5." + key]
        count = get("block_count")

        def per_layer(x):
            return tuple(x) if isinstance(x, list) else (x,) * count

        sliding = per_layer(get("attention.sliding_window_pattern"))
        if len(sliding) != count or any(type(x) is not bool for x in sliding):
            raise ValueError("invalid Spark sliding window pattern")
        base = get("rope.freq_base")
        swa_base = m.get("spark2_5.rope.freq_base_swa", base)
        rotary = per_layer(get("rope.dimension_count"))
        if len(rotary) != count or get("attention.value_length") != get(
            "attention.key_length"
        ):
            raise ValueError("invalid Spark rotary or value dimensions")
        swa_rotary = m.get("spark2_5.rope.dimension_count_swa", rotary[0])
        return cls(
            get("embedding_length"),
            get("feed_forward_length"),
            get("attention.head_count"),
            get("attention.head_count_kv"),
            get("attention.key_length"),
            tuple("sliding_attention" if x else "full_attention" for x in sliding),
            gguf.tensors["token_embd.weight"].shape[0],
            get("attention.layer_norm_rms_epsilon"),
            get("context_length"),
            get("attention.sliding_window"),
            tuple(swa_base if x else base for x in sliding),
            tuple(swa_rotary if x else rot for x, rot in zip(sliding, rotary)),
            "output.weight" not in gguf.tensors,
        )
