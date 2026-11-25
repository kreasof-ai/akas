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
    def from_huggingface(cls, data):
        from dataclasses import asdict

        if data.get("model_type") != "lfm2_moe":
            raise ValueError("requires LFM2 MoE config.json")
        base = Config.from_huggingface({**data, "model_type": "lfm2"})
        result = cls(
            **asdict(base),
            experts=data["num_experts"],
            top_k=data["num_experts_per_tok"],
            expert_width=data["moe_intermediate_size"],
            dense_layers=data["num_dense_layers"],
            routing_scale=data.get("routed_scaling_factor", 1.0),
            normalize_routes=data.get("norm_topk_prob", True),
        )
        if (
            any(
                type(n) is not int or n < 1
                for n in (result.experts, result.top_k, result.expert_width)
            )
            or result.top_k > result.experts
            or result.expert_width % 128
            or type(result.dense_layers) is not int
            or not 0 <= result.dense_layers < len(result.layers)
            or type(result.normalize_routes) is not bool
            or not math.isfinite(result.routing_scale)
            or result.routing_scale <= 0
            or data.get("use_expert_bias") is not True
        ):
            raise ValueError("unsupported LFM2 MoE routing configuration")
        return result

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
    validate_tensors(config, gguf)
    return config


def tensor_shapes(config, *, output=False):
    expected = {
        "token_embd.weight": (config.vocab, config.width),
        "token_embd_norm.weight": (config.width,),
    }
    for layer, kind in enumerate(config.layers):
        p = f"blk.{layer}."
        c = config.width
        f = config.ff
        d = config.head_dim
        kh = config.kv_heads
        expected.update(
            {
                p + "attn_norm.weight": (c,),
                p + "ffn_norm.weight": (c,),
            }
        )
        if isinstance(config, LFM2MoEConfig) and layer >= config.dense_layers:
            e, f = config.experts, config.expert_width
            expected.update(
                {
                    p + "ffn_gate_inp.weight": (e, c),
                    p + "exp_probs_b.bias": (e,),
                    p + "ffn_gate_exps.weight": (e, f, c),
                    p + "ffn_up_exps.weight": (e, f, c),
                    p + "ffn_down_exps.weight": (e, c, f),
                }
            )
        else:
            expected.update(
                {
                    p + "ffn_gate.weight": (f, c),
                    p + "ffn_up.weight": (f, c),
                    p + "ffn_down.weight": (c, f),
                }
            )
        if kind == "conv":
            expected.update(
                {
                    p + "shortconv.in_proj.weight": (3 * c, c),
                    p + "shortconv.out_proj.weight": (c, c),
                    p + "shortconv.conv.weight": (c, 3),
                }
            )
        else:
            expected.update(
                {
                    p + "attn_q.weight": (c, c),
                    p + "attn_k.weight": (kh * d, c),
                    p + "attn_v.weight": (kh * d, c),
                    p + "attn_output.weight": (c, c),
                    p + "attn_q_norm.weight": (d,),
                    p + "attn_k_norm.weight": (d,),
                }
            )
    if output:
        expected["output.weight"] = (config.vocab, config.width)
    return expected


def validate_tensors(config, gguf):
    expected = tensor_shapes(config, output="output.weight" in gguf.tensors)
    if set(expected) != set(gguf.tensors):
        raise GGUFError("unsupported/missing LFM2 tensor inventory")
    for name, shape in expected.items():
        if gguf.tensors[name].shape != shape:
            raise GGUFError(f"{name}: incompatible LFM2 tensor shape")
        if len(shape) == 1 or name.endswith(("conv.weight", "ffn_gate_inp.weight")):
            if gguf.tensors[name].type != 0:
                raise GGUFError(f"{name}: requires float32 auxiliary weights")
