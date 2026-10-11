"""Request lifecycle data shared by scheduling and serving."""

from dataclasses import dataclass, field
from threading import Event


@dataclass
class Request:
    prompt: list[int]
    limit: int
    channel: object = None
    cancelled: Event = field(default_factory=Event)
    slot: int | None = None
    offset: int = 0
    output: list[int] = field(default_factory=list)
    published: int = 0
