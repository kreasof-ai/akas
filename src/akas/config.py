"""Engine scheduling budgets; model and device geometry stay with their owners."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    prefill_chunk: int = 32
    prefill_requests: int = 32

    def __post_init__(self):
        for value in (self.prefill_chunk, self.prefill_requests):
            if type(value) is not int or value < 1:
                raise ValueError("prefill budgets must be positive integers")
