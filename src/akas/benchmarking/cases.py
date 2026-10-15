"""Reproducible workload matrices, independent of model architecture/providers."""

from dataclasses import dataclass, asdict
from itertools import product
from akas.benchmarking.workload import STANDARD


@dataclass(frozen=True)
class Case:
    batch: int
    prompt: int
    generated: int = STANDARD["output_tokens"]
    depth: int = 0
    mixed: bool = False

    def __post_init__(self):
        if (
            any(
                type(x) is not int or x < 1
                for x in (self.batch, self.prompt, self.generated)
            )
            or type(self.depth) is not int
            or self.depth < 0
            or type(self.mixed) is not bool
        ):
            raise ValueError(
                "positive batch/prompt/generation and nonnegative depth required"
            )

    @property
    def lengths(self):
        return [
            max(1, self.prompt - i % min(self.prompt, 7)) if self.mixed else self.prompt
            for i in range(self.batch)
        ]

    @property
    def capacity(self):
        return self.depth + self.prompt + self.generated

    def metadata(self):
        return asdict(self)


def matrix(
    *,
    batches=tuple(STANDARD["batches"]),
    prompts=tuple(STANDARD["prefix_tokens"]),
    depths=(0,),
    generated=STANDARD["output_tokens"],
    mixed=False,
):
    return [
        Case(b, p, generated, d, mixed) for b, p, d in product(batches, prompts, depths)
    ]
