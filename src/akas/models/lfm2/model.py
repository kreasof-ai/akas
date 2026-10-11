"""LFM2 architecture composed over packed-weight execution."""

from akas.execution.independent.resident import PackedModel
from akas.models.lfm2.weights import LFM2Checkpoint
from akas.models.blocks import DecoderArchitecture
from akas.models.blocks import DecoderBlock


def architecture(config, checkpoint):
    blocks = []
    for index, kind in enumerate(config.layers):
        prefix = f"blk.{index}."
        blocks.append(
            DecoderBlock(
                index,
                prefix,
                kind,
                prefix + ("shortconv." if kind == "conv" else "attn_"),
                "packed",
                prefix + "ffn_",
                prefix + "attn_norm.weight",
                prefix + "ffn_norm.weight",
            )
        )
    head = (
        "output.weight"
        if "output.weight" in checkpoint.gguf.tensors
        else "token_embd.weight"
    )
    return DecoderArchitecture(
        tuple(blocks), "token_embd.weight", "token_embd_norm.weight", head
    )


class LFM2(PackedModel):
    checkpoint_type = LFM2Checkpoint
    definition = staticmethod(architecture)
