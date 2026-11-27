"""Text geometry for the released fixed-pass Inkling 2 Mini checkpoint."""

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class Inkling2MiniConfig:
    width: int
    layers: int
    vocab: int
    padded_vocab: int
    heads: int
    kv_heads: int
    head_dim: int
    relative_dim: int
    relative_extent: int
    local_layers: tuple[int, ...]
    window: int
    taps: int
    block_size: int
    experts: int
    expert_width: int
    top_k: int
    loop_start: int
    loop_end: int
    passes: int
    gate_rank: int
    max_context: int
    epsilon: float
    qk_clip: float
    relative_clip: float
    log_floor: int
    log_alpha: float
    route_scale: float
    gate_cap: float
    up_cap: float
    logit_cap: float

    @classmethod
    def from_huggingface(cls, data):
        if data.get("model_type") != "inkling2_loop":
            raise ValueError("requires an Inkling 2 loop checkpoint")
        text = data["text_config"]
        loop = text["loop_config"]
        local_layers = tuple(text["local_layer_ids"])
        if (
            text.get("model_type") != "inkling2_loop_text"
            or loop["is_varloop"]
            or text.get("loop_share_cache", False)
            or text["q_bias"]
            or text["o_bias"]
            or text["n_shared_experts"]
            or not text["use_gate_bias"]
            or not text["attn_res_normalize_values"]
            or not text["use_value_norm_no_gamma"]
            or any(type(layer) is not int for layer in local_layers)
            or len(set(local_layers)) != len(local_layers)
            or any(
                text[name] != text["swa_" + name]
                for name in ("head_dim", "num_attention_heads", "num_key_value_heads")
            )
        ):
            raise ValueError("unsupported Inkling 2 Mini text configuration")

        return cls(
            width=text["hidden_size"],
            layers=text["num_hidden_layers"],
            vocab=text["unpadded_vocab_size"],
            padded_vocab=text["vocab_size"],
            heads=text["num_attention_heads"],
            kv_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            relative_dim=text["d_rel"],
            relative_extent=text["rel_extent"],
            local_layers=local_layers,
            window=text["sliding_window_size"],
            taps=text["sconv_kernel_size"],
            block_size=text["n_layers_per_attn_res_block"],
            experts=text["n_routed_experts"],
            expert_width=text["intermediate_size"],
            top_k=text["num_experts_per_tok"],
            loop_start=loop["core_start"],
            loop_end=loop["core_end"],
            passes=loop["num_passes"],
            gate_rank=loop["gate_rank"],
            max_context=text["model_max_length"],
            epsilon=text["rms_norm_eps"],
            qk_clip=text["qk_l2_clip_threshold"],
            relative_clip=text["rel_l2_clip_threshold"],
            log_floor=text["log_scaling_n_floor"],
            log_alpha=text["log_scaling_alpha"],
            route_scale=text["route_scale"],
            gate_cap=text["mlp_gate_softcap"],
            up_cap=text["mlp_up_softcap"],
            logit_cap=text["final_logit_softcapping"],
        )

    def __post_init__(self):
        dimensions = (getattr(self, f.name) for f in fields(self) if f.type is int)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Inkling 2 Mini dimensions must be positive integers")
        scales = (getattr(self, f.name) for f in fields(self) if f.type is float)
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in scales
        ):
            raise ValueError("invalid Inkling 2 Mini scaling constants")
        if (
            not 0 < self.loop_start < self.loop_end < self.layers
            or self.loop.start % self.block_size
            or self.loop.stop % self.block_size
            or self.passes < 2
            or self.heads % self.kv_heads
            or self.top_k > self.experts
            or self.vocab > self.padded_vocab
            or self.window > self.max_context
            or self.taps < 2
            or not set(self.local_layers) <= set(range(self.layers))
        ):
            raise ValueError("invalid Inkling 2 Mini text geometry")

    @property
    def loop(self):
        return range(self.loop_start, self.loop_end)

    def visits(self):
        """Physical weight indices and independent cache pass indices."""
        yield from ((layer, 0) for layer in range(self.loop.start))
        for pass_index in range(self.passes):
            yield from ((layer, pass_index) for layer in self.loop)
        yield from ((layer, 0) for layer in range(self.loop.stop, self.layers))

    def kv_bytes_per_slot(self, context):
        """Payload for the two 16-bit KV caches, excluding convolution state."""
        if type(context) is not int or not 1 <= context <= self.max_context:
            raise ValueError("KV context lies outside the model geometry")
        tokens = sum(
            min(context, self.window) if layer in self.local_layers else context
            for layer, _ in self.visits()
        )
        return 4 * tokens * self.kv_heads * self.head_dim
