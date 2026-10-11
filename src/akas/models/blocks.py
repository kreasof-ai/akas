"""Declarative decoder composition; execution and kernels are supplied separately."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderBlock:
    index: int
    prefix: str
    mixer: str
    mixer_prefix: str
    feed_forward: str
    feed_forward_prefix: str
    input_norm: str
    post_norm: str

    def compose(self, context):
        context.input_norm(context, self)
        context.mixers[self.mixer](context, self)
        context.post_norm(context, self)
        return context.feed_forwards[self.feed_forward](context, self)


@dataclass(frozen=True)
class DecoderArchitecture:
    blocks: tuple[DecoderBlock, ...]
    embedding: str
    final_norm: str
    head: str
