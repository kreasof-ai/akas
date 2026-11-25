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
        return expected_tensors(self)

    def expected_mtp_tensors(self):
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


def expected_moe_tensors(config):
    shapes = {
        "model.language_model.embed_tokens.weight": (config.vocab, config.width),
        "model.language_model.norm.weight": (config.width,),
        "lm_head.weight": (config.vocab, config.width),
    }

    def vector(name, shape):
        shapes[name] = shape

    def fp8(name, rows, columns):
        shapes[name + ".weight"] = (rows, columns)
        shapes[name + ".weight_scale_inv"] = (
            (rows + 127) // 128,
            (columns + 127) // 128,
        )

    for layer, kind in enumerate(config.layers):
        root = f"model.language_model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            vector(root + name + ".weight", (config.width,))
        if kind == "linear_attention":
            prefix = root + "linear_attn."
            vector(prefix + "A_log", (config.value_heads,))
            vector(prefix + "dt_bias", (config.value_heads,))
            vector(prefix + "conv1d.weight", (8192, 1, config.conv_width))
            vector(prefix + "norm.weight", (config.value_dim,))
            for name in ("a", "b"):
                vector(
                    prefix + "in_proj_" + name + ".weight",
                    (config.value_heads, config.width),
                )
            for name, rows, columns in (
                ("in_proj_qkv", 8192, config.width),
                ("in_proj_z", 4096, config.width),
                ("out_proj", config.width, 4096),
            ):
                fp8(prefix + name, rows, columns)
        else:
            prefix = root + "self_attn."
            for name, rows, columns in (
                ("q_proj", 8192, config.width),
                ("k_proj", 512, config.width),
                ("v_proj", 512, config.width),
                ("o_proj", config.width, 4096),
            ):
                fp8(prefix + name, rows, columns)
            for name in ("q_norm", "k_norm"):
                vector(prefix + name + ".weight", (config.head_dim,))
        prefix = root + "mlp."
        vector(prefix + "gate.weight", (config.experts, config.width))
        vector(prefix + "shared_expert_gate.weight", (1, config.width))
        for expert in (
            "shared_expert",
            *(f"experts.{i}" for i in range(config.experts)),
        ):
            for name, rows, columns in (
                ("gate_proj", config.expert_width, config.width),
                ("up_proj", config.expert_width, config.width),
                ("down_proj", config.width, config.expert_width),
            ):
                fp8(prefix + expert + "." + name, rows, columns)
    return shapes


def expected_dense_tensors(config):
    shapes = {
        "model.language_model.embed_tokens.weight": (config.vocab, config.width),
        "model.language_model.norm.weight": (config.width,),
    }
    for layer, kind in enumerate(config.layers):
        root = f"model.language_model.layers.{layer}."
        shapes.update(
            {
                root + n + ".weight": (config.width,)
                for n in ("input_layernorm", "post_attention_layernorm")
            }
        )
        for name, shape in (
            ("gate", (config.intermediate, config.width)),
            ("up", (config.intermediate, config.width)),
            ("down", (config.width, config.intermediate)),
        ):
            shapes[root + "mlp." + name + "_proj.weight"] = shape
        if kind == "linear_attention":
            prefix = root + "linear_attn."
            shapes.update(
                {
                    prefix + "A_log": (config.value_heads,),
                    prefix + "dt_bias": (config.value_heads,),
                    prefix + "conv1d.weight": (config.qkv_width, 1, 4),
                    prefix + "norm.weight": (config.value_dim,),
                }
            )
            for name, shape in (
                ("qkv", (config.qkv_width, config.width)),
                ("z", (config.value_width, config.width)),
                ("a", (config.value_heads, config.width)),
                ("b", (config.value_heads, config.width)),
            ):
                shapes[prefix + "in_proj_" + name + ".weight"] = shape
            shapes[prefix + "out_proj.weight"] = (config.width, config.value_width)
        elif kind == "full_attention":
            prefix = root + "self_attn."
            for name, shape in (
                ("q", (2 * config.heads * config.head_dim, config.width)),
                ("k", (config.kv_heads * config.head_dim, config.width)),
                ("v", (config.kv_heads * config.head_dim, config.width)),
                ("o", (config.width, config.heads * config.head_dim)),
            ):
                shapes[prefix + name + "_proj.weight"] = shape
            shapes.update(
                {prefix + n + "_norm.weight": (config.head_dim,) for n in ("q", "k")}
            )
        else:
            raise ValueError("unsupported dense layer type")
    full = next((i for i, kind in enumerate(config.layers) if kind == "full_attention"))
    prefix = f"model.language_model.layers.{full}."
    shapes.update(
        {
            "mtp.layers.0." + name[len(prefix) :]: shape
            for name, shape in tuple(shapes.items())
            if name.startswith(prefix)
        }
    )
    shapes.update(
        {
            "mtp." + n + ".weight": (config.width,)
            for n in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden")
        }
    )
    shapes["mtp.fc.weight"] = (config.width, 2 * config.width)
    return shapes


def expected_mtp_tensors(config):
    """The official one-layer draft branch; embedding/head are shared."""
    layer = next(i for i, kind in enumerate(config.layers) if kind == "full_attention")
    prefix = f"model.language_model.layers.{layer}."
    shapes = {
        "mtp.layers.0." + name[len(prefix) :]: shape
        for name, shape in expected_tensors(config).items()
        if name.startswith(prefix)
    }
    shapes.update(
        {
            "mtp." + name + ".weight": (config.width,)
            for name in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden")
        }
    )
    shapes["mtp.fc.weight"] = (config.width, 2 * config.width)
    return shapes


def expected_tensors(config):
    return (
        expected_moe_tensors(config)
        if isinstance(config, Qwen35MoEConfig)
        else expected_dense_tensors(config)
    )


# Compatibility for existing imports and stored definition paths.
Qwen35Config = Qwen35MoEConfig
DenseConfig = Qwen35DenseConfig
