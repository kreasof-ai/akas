"""Engine scheduling budgets; model and device geometry stay with their owners."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    prefill_chunk: int = 32
    prefill_requests: int = 32
    prefill_token_budget: int | None = None
    decode_prefill_token_budget: int | None = None

    def __post_init__(self):
        for value in (self.prefill_chunk, self.prefill_requests):
            if type(value) is not int or value < 1:
                raise ValueError("prefill budgets must be positive integers")
        if self.prefill_token_budget is not None and (
            type(self.prefill_token_budget) is not int or self.prefill_token_budget < 1
        ):
            raise ValueError("prefill token budget must be a positive integer")
        if self.decode_prefill_token_budget is not None and (
            type(self.decode_prefill_token_budget) is not int or self.decode_prefill_token_budget < 1
        ):
            raise ValueError("decode prefill token budget must be a positive integer")
