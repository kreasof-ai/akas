"""Architecture dimensions and checkpoint constraints for dense and MoE Qwen."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen35MoEConfig:
    layers: tuple[str, ...]
    context_limit: int
    epsilon: float
    rope_theta: float
    width: int = 2048
    vocab: int = 248320
    heads: int = 16
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    key_heads: int = 16
    value_heads: int = 32
    key_dim: int = 128
    value_dim: int = 128
    conv_width: int = 4
    experts: int = 256
    top_k: int = 8
    expert_width: int = 512

    @property
    def qkv_width(self):
        return 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim

    @property
    def value_width(self):
        return self.value_heads * self.value_dim

    @property
    def theta(self):
        return self.rope_theta

    @classmethod
    def from_gguf(cls, gguf):
        return parse_gguf(gguf, moe=True)

    @classmethod
    def from_json(cls, value):
        return parse_moe_config(value)

    def state_bytes(self, *, slots, context, block=128):
        from akas.memory.estimates import state_bytes

        return state_bytes(self, slots=slots, context=context, block=block)

    def expected_tensors(self):
        from akas.models.qwen35.weights import expected_tensors

        return expected_tensors(self)

    def expected_mtp_tensors(self):
        from akas.models.qwen35.weights import expected_mtp_tensors

        return expected_mtp_tensors(self)


@dataclass(frozen=True)
class Qwen35DenseConfig:
    width: int
    intermediate: int
    vocab: int
    layers: tuple[str, ...]
    heads: int
    kv_heads: int
    head_dim: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    epsilon: float
    theta: float
    rotary_dim: int
    context_limit: int

    @classmethod
    def from_json(cls, value):
        return parse_dense_config(value)

    @classmethod
    def from_gguf(cls, gguf):
        return parse_gguf(gguf, moe=False)

    @property
    def qkv_width(self):
        return 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim

    @property
    def value_width(self):
        return self.value_heads * self.value_dim

    def expected_tensors(self):
        from akas.models.qwen35.weights import expected_tensors

        return expected_tensors(self)


def parse_moe_config(value):
    text = value["text_config"]
    required = dict(
        model_type="qwen3_5_moe_text",
        hidden_size=2048,
        vocab_size=248320,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        hidden_act="silu",
        attn_output_gate=True,
        attention_bias=False,
    )
    for name, expected in required.items():
        if text.get(name) != expected:
            raise ValueError(f"unsupported Qwen35 configuration: {name}")
    layers = tuple(text["layer_types"])
    if layers != (("linear_attention",) * 3 + ("full_attention",)) * 10:
        raise ValueError("unsupported Qwen35 layer order")
    quant = value["quantization_config"]
    if (
        quant.get("quant_method") != "fp8"
        or quant.get("activation_scheme") != "dynamic"
        or quant.get("weight_block_size") != [128, 128]
    ):
        raise ValueError("Qwen35 requires dynamic block-128 native FP8 weights")
    rope = text["rope_parameters"]
    if (
        rope.get("rope_type") != "default"
        or rope.get("partial_rotary_factor") != 0.25
        or rope.get("mrope_section") != [11, 11, 10]
        or (not rope.get("mrope_interleaved"))
    ):
        raise ValueError("unsupported Qwen35 rotary contract")
    epsilon, theta = (float(text["rms_norm_eps"]), float(rope["rope_theta"]))
    limit = text["max_position_embeddings"]
    if (
        not math.isfinite(epsilon)
        or epsilon <= 0
        or (not math.isfinite(theta))
        or (theta <= 0)
        or (type(limit) is not int)
        or (limit < 48000)
    ):
        raise ValueError("invalid Qwen35 numeric configuration")
    if text.get("mlp_only_layers") or text.get("layer_scale", False):
        raise ValueError("unsupported Qwen35 decoder variant")
    return Qwen35MoEConfig(layers, limit, epsilon, theta)


def parse_dense_config(value):
    text = value["text_config"]
    rope = text["rope_parameters"]
    if (
        text["model_type"] != "qwen3_5_text"
        or text["hidden_act"] != "silu"
        or (not text["attn_output_gate"])
        or text["attention_bias"]
        or (text["linear_conv_kernel_dim"] != 4)
        or (not text["tie_word_embeddings"])
        or (rope["rope_type"] != "default")
        or (not rope["mrope_interleaved"])
    ):
        raise ValueError("unsupported dense Qwen variant")
    result = Qwen35DenseConfig(
        text["hidden_size"],
        text["intermediate_size"],
        text["vocab_size"],
        tuple(text["layer_types"]),
        text["num_attention_heads"],
        text["num_key_value_heads"],
        text["head_dim"],
        text["linear_num_key_heads"],
        text["linear_num_value_heads"],
        text["linear_key_head_dim"],
        text["linear_value_head_dim"],
        text["rms_norm_eps"],
        rope["rope_theta"],
        int(text["head_dim"] * rope["partial_rotary_factor"]),
        text["max_position_embeddings"],
    )
    if (
        len(result.layers) != text["num_hidden_layers"]
        or result.key_dim != 128
        or result.value_dim != 128
        or (result.head_dim != 256)
        or (result.rotary_dim != 64)
        or result.heads % result.kv_heads
        or result.value_heads % result.key_heads
    ):
        raise ValueError("unsupported dense Qwen geometry")
    return result


def parse_gguf(gguf, *, moe):
    """Read Qwen geometry from GGUF, independently of its weight encoding."""
    from akas.checkpoints.gguf import GGUFError

    metadata = gguf.metadata
    architecture = "qwen35moe" if moe else "qwen35"
    if metadata.get("general.architecture") != architecture:
        raise GGUFError(f"requires the {architecture} GGUF architecture")

    def integer(name, default=None):
        value = metadata.get(architecture + "." + name, default)
        if type(value) is not int or value <= 0:
            raise GGUFError(f"invalid {architecture}.{name}")
        return value

    count = integer("block_count")
    extra = metadata.get(architecture + ".nextn_predict_layers", 0)
    if type(extra) is not int or not 0 <= extra < count:
        raise GGUFError("invalid Qwen GGUF MTP layer count")
    count -= extra
    recurrent = metadata.get(architecture + ".attention.recurrent_layers")
    if recurrent is None:
        interval = integer("full_attention_interval", 4)
        recurrent = [(index + 1) % interval != 0 for index in range(count)] + [
            False
        ] * extra
    if (
        not isinstance(recurrent, (tuple, list))
        or len(recurrent) != count + extra
        or any(type(flag) is not bool for flag in recurrent)
    ):
        raise GGUFError("invalid Qwen GGUF recurrent layer array")
    layers = tuple(
        "linear_attention" if flag else "full_attention" for flag in recurrent[:count]
    )
    width = integer("embedding_length")
    vocab = len(metadata.get("tokenizer.ggml.tokens", ()))
    if not vocab:
        vocab = integer("vocab_size")
    epsilon = metadata.get(architecture + ".attention.layer_norm_rms_epsilon")
    theta = metadata.get(architecture + ".rope.freq_base")
    if any(
        type(value) not in (float, int) or not math.isfinite(value) or value <= 0
        for value in (epsilon, theta)
    ):
        raise GGUFError("invalid Qwen GGUF norm or rotary parameters")
    heads, kv_heads = (
        integer("attention.head_count"),
        integer("attention.head_count_kv"),
    )
    dim, value_dim = integer("attention.key_length"), integer("attention.value_length")
    key_heads, value_heads = integer("ssm.group_count"), integer("ssm.time_step_rank")
    state_dim = integer("ssm.state_size")
    rotary = integer("rope.dimension_count")
    if (
        dim != 256
        or value_dim != dim
        or rotary != 64
        or state_dim != 128
        or integer("ssm.conv_kernel") != 4
        or heads % kv_heads
        or value_heads % key_heads
        or integer("ssm.inner_size") != value_heads * state_dim
        or metadata.get(architecture + ".rope.dimension_sections") != [11, 11, 10, 0]
        or "full_attention" not in layers
    ):
        raise GGUFError("unsupported Qwen GGUF attention geometry")
    context = integer("context_length")
    if moe:
        experts, top = integer("expert_count"), integer("expert_used_count")
        expert_width = integer("expert_feed_forward_length")
        shared = integer("expert_shared_feed_forward_length", expert_width)
        if top > experts or shared != expert_width:
            raise GGUFError("unsupported Qwen GGUF expert geometry")
        return Qwen35MoEConfig(
            layers,
            context,
            epsilon,
            theta,
            width,
            vocab,
            heads,
            kv_heads,
            dim,
            rotary,
            key_heads,
            value_heads,
            state_dim,
            state_dim,
            4,
            experts,
            top,
            expert_width,
        )
    return Qwen35DenseConfig(
        width,
        integer("feed_forward_length"),
        vocab,
        layers,
        heads,
        kv_heads,
        dim,
        key_heads,
        value_heads,
        state_dim,
        state_dim,
        epsilon,
        theta,
        rotary,
        context,
    )


# Compatibility for existing imports and stored definition paths.
Qwen35Config = Qwen35MoEConfig
DenseConfig = Qwen35DenseConfig
