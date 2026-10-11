"""Qwen3.5 architecture; select MoE/FP8 or dense/BF16 execution explicitly."""

from akas.execution.bindings.block_fp8.resources import BlockFP8Binding
from akas.execution.bindings.bf16.resources import BF16Binding
from akas.models.qwen35.weights import Qwen35Checkpoint
from akas.models.qwen35.weights import DenseCheckpoint
from akas.models.qwen35.config import Qwen35Config
from akas.models.blocks import DecoderArchitecture
from akas.models.blocks import DecoderBlock


def architecture(config, checkpoint):
    feed_forward = "moe" if isinstance(config, Qwen35Config) else "dense"
    blocks = []
    for index, kind in enumerate(config.layers):
        prefix = f"model.language_model.layers.{index}."
        blocks.append(
            DecoderBlock(
                index,
                prefix,
                kind,
                prefix
                + ("linear_attn." if kind == "linear_attention" else "self_attn."),
                feed_forward,
                prefix + "mlp.",
                prefix + "input_layernorm",
                prefix + "post_attention_layernorm",
            )
        )
    embedding = "model.language_model.embed_tokens.weight"
    return DecoderArchitecture(
        tuple(blocks),
        embedding,
        "model.language_model.norm.weight",
        "lm_head.weight" if feed_forward == "moe" else embedding,
    )


class Qwen35Batch(BlockFP8Binding):
    checkpoint_type = Qwen35Checkpoint
    definition = staticmethod(architecture)


class DenseEngine(BF16Binding):
    checkpoint_type = DenseCheckpoint
    definition = staticmethod(architecture)
