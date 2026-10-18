"""LFM2 architecture dimensions and supported GGUF geometry."""

import math
from dataclasses import dataclass

from akas.checkpoints.gguf import GGUFError


@dataclass(frozen=True)
class Config:
    width: int
    ff: int
    heads: int
    kv_heads: int
    layers: tuple[str, ...]
    vocab: int
    conv: int
    epsilon: float
    theta: float
    max_context: int

    @property
    def head_dim(self):
        return self.width // self.heads

    @classmethod
    def from_huggingface(cls, data):
        if data.get("model_type") != "lfm2":
            raise ValueError("requires LFM2 config.json")
        width = data["hidden_size"]
        ff = data.get("intermediate_size", data.get("block_ff_dim"))
        if data.get("block_auto_adjust_ff_dim", False):
            ff = int(2 * ff / 3)
            multiplier = data.get("block_ffn_dim_multiplier")
            if multiplier is not None:
                multiple = data.get("block_multiple_of", 256)
                ff = math.ceil(int(multiplier * ff) / multiple) * multiple
        count = data["num_hidden_layers"]
        indices = data.get("full_attn_idxs")
        layers = data.get("layer_types")
        if layers is None:
            if not isinstance(indices, list) or any(
                type(i) is not int or not 0 <= i < count for i in indices
            ):
                raise ValueError("invalid LFM2 attention layer indices")
            layers = ["attention" if i in indices else "conv" for i in range(count)]
        layers = tuple("attention" if x == "full_attention" else x for x in layers)
        rope = data.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default" or data.get("rope_scaling"):
            raise ValueError("reference path requires default RoPE")
        config = cls(
            width,
            ff,
            data["num_attention_heads"],
            data["num_key_value_heads"],
            layers,
            data["vocab_size"],
            data.get("conv_L_cache", 3),
            data.get("norm_eps", 1e-5),
            rope.get("rope_theta", data.get("rope_theta", 1000000.0)),
            data.get("max_position_embeddings", 128000),
        )
        if (
            len(layers) != count
            or not layers
            or set(layers) - {"attention", "conv"}
            or any(
                type(x) is not int or x <= 0
                for x in (
                    width,
                    ff,
                    config.heads,
                    config.kv_heads,
                    config.vocab,
                    config.max_context,
                )
            )
            or width % config.heads
            or config.heads % config.kv_heads
            or config.head_dim != 64
            or config.conv != 3
            or ff % 32
            or config.vocab % 4
            or not all(
                math.isfinite(x) and x > 0 for x in (config.epsilon, config.theta)
            )
            or data.get("conv_bias", False)
            or not data.get("block_use_swiglu", True)
            or not data.get("use_pos_enc", True)
        ):
            raise ValueError("unsupported LFM2 reference configuration")
        return config

    @classmethod
    def from_gguf(cls, gguf):
        return parse_config(gguf)


@dataclass(frozen=True)
class LFM2MoEConfig(Config):
    experts: int
    top_k: int
    expert_width: int
    dense_layers: int
    routing_scale: float = 1.0
    normalize_routes: bool = True

    @classmethod
    def from_gguf(cls, gguf):
        return parse_config(gguf, moe=True)


def parse_config(gguf, *, moe=False):
    m = gguf.metadata
    architecture = "lfm2moe" if moe else "lfm2"
    if m.get("general.architecture") != architecture:
        raise GGUFError(f"requires the {architecture} architecture")
    get = lambda name: m[architecture + "." + name]
    kv = get("attention.head_count_kv")
    if not isinstance(kv, list) or len(kv) != get("block_count"):
        raise GGUFError("requires per-layer attention metadata")
    unique = set(kv) - {0}
    if len(unique) != 1:
        raise GGUFError("requires uniform nonzero GQA head count")
    config = Config(
        get("embedding_length"),
        get("feed_forward_length"),
        get("attention.head_count"),
        unique.pop(),
        tuple("attention" if n else "conv" for n in kv),
        get("vocab_size"),
        get("shortconv.l_cache"),
        get("attention.layer_norm_rms_epsilon"),
        get("rope.freq_base"),
        get("context_length"),
    )
    if any(
        type(value) is not int or value <= 0
        for value in (
            config.width,
            config.ff,
            config.heads,
            config.kv_heads,
            config.vocab,
            config.max_context,
        )
    ):
        raise GGUFError("LFM2 dimensions must be positive integers")
    if not config.layers or any(type(n) is not int or n < 0 for n in kv):
        raise GGUFError("invalid LFM2 layer/head metadata")
    if not all(
        math.isfinite(value) and value > 0 for value in (config.epsilon, config.theta)
    ):
        raise GGUFError("LFM2 normalization/RoPE constants must be finite and positive")
    if (
        config.width % config.heads
        or config.heads % config.kv_heads
        or config.head_dim != 64
        or config.ff % 32
        or config.vocab % 4
        or (config.conv != 3)
    ):
        raise GGUFError("unsupported LFM2 head/convolution configuration")
    if moe:
        from dataclasses import asdict

        config = LFM2MoEConfig(
            **asdict(config),
            experts=get("expert_count"),
            top_k=get("expert_used_count"),
            expert_width=get("expert_feed_forward_length"),
            dense_layers=get("leading_dense_block_count"),
        )
        if (
            any(
                type(v) is not int or v < 1
                for v in (config.experts, config.top_k, config.expert_width)
            )
            or config.top_k > config.experts
            or config.expert_width % 32
            or type(config.dense_layers) is not int
            or not 0 <= config.dense_layers < len(config.layers)
            or get("expert_gating_func") != 2
        ):
            raise GGUFError("unsupported LFM2 sigmoid MoE routing configuration")
    from akas.models.lfm2.weights import validate_tensors

    validate_tensors(config, gguf)
    return config
