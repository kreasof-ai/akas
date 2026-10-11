"""Reproducible workload matrices, independent of model architecture/providers."""

from dataclasses import dataclass, asdict
from itertools import product


@dataclass(frozen=True)
class Case:
    batch: int
    prompt: int
    generated: int = 32
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
    batches=(1, 4, 16, 64, 256),
    prompts=(32, 2048),
    depths=(0,),
    generated=32,
    mixed=False,
):
    return [
        Case(b, p, generated, d, mixed) for b, p, d in product(batches, prompts, depths)
    ]
