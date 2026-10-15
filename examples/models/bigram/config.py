"""Geometry for a tiny token-to-next-token score table."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vocab: int = 32
    context: int = 128

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in (self.vocab, self.context)):
            raise ValueError("vocabulary and context must be positive integers")
