"""Minimal scheduling interface; model adapters declare their execution limits."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Capabilities:
    native_batch: bool
    speculative: bool = False


class ScheduledModel(Protocol):
    capacity: int
    context: int

    def admit(self) -> int: ...
    def release(self, slot: int) -> None: ...
    def decode(self, slots: list[int], tokens: list[int]) -> list[int]: ...
    def close(self) -> None: ...


class IndependentRequest(Protocol):
    """Sequence state and greedy execution; providers choose their readback path."""

    def reset(self) -> None: ...
    def next_token(self, tokens: list[int]) -> int: ...
    def close(self) -> None: ...


class IndependentModel(IndependentRequest, Protocol):
    """Shared resources plus a default request; close also closes child handles."""

    context: int
    config: object

    def new_request(self) -> IndependentRequest: ...
