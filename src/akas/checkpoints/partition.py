"""Row-aligned floating weight partitions for portable storage binding limits."""

from dataclasses import dataclass
import math

DEFAULT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class Part:
    name: str
    start: int
    rows: int
    shape: tuple[int, ...]


def partitions(name, shape, *, max_bytes=DEFAULT_BYTES):
    shape = tuple(shape)
    if (
        type(max_bytes) is not int
        or max_bytes < 4
        or not shape
        or any(type(x) is not int or x < 1 for x in shape)
    ):
        raise ValueError("invalid floating weight partition contract")
    if math.prod(shape) * 4 <= max_bytes:
        return (Part(name, 0, shape[0], shape),)
    if len(shape) != 2 or shape[1] * 4 > max_bytes:
        raise ValueError("weight cannot be partitioned into complete matrix rows")
    rows = max_bytes // (shape[1] * 4)
    return tuple(
        Part(
            f"{name}.part{i}",
            start,
            min(rows, shape[0] - start),
            (min(rows, shape[0] - start), shape[1]),
        )
        for i, start in enumerate(range(0, shape[0], rows))
    )


class Weights:
    """Declare the same physical partitions that checkpoint upload will allocate."""

    def __init__(self, plan, shapes, *, max_bytes=DEFAULT_BYTES):
        self.plan = plan
        self.parts = {
            name: partitions(name, shape, max_bytes=max_bytes)
            for name, shape in shapes.items()
        }
        self.buffers = {
            part.name: plan.buffer(
                part.name, (math.prod(part.shape),), "float32", role="weight"
            )
            for parts in self.parts.values()
            for part in parts
        }

    def __getitem__(self, name):
        parts = self.parts[name]
        if len(parts) != 1:
            raise ValueError("operation needs a partition-aware matrix binding")
        return self.buffers[parts[0].name]

    def matrix(self, name):
        return [(self.buffers[p.name], p.start, p.rows) for p in self.parts[name]]
