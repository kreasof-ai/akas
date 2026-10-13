"""The proposal interface shared by speculative heads and history lookups."""

from typing import Protocol


class Proposer(Protocol):
    def admit(self, slot: int) -> None: ...
    def release(self, slot: int) -> None: ...
    def prefill(self, requests, sequences, predictions, *, target, chunk: int) -> None: ...
    def propose(self, requests, limits, *, target) -> list[list[int]]: ...
    def repair(self, requests, outputs, positions, *, target, chunk: int) -> None: ...
    def close(self) -> None: ...


class LookupProposer:
    """An ordinary proposer; the verifier decides whether history is correct."""

    def __init__(self, **options):
        self.options = options
        self.history = {}

    def admit(self, slot):
        from akas.generation.lookup import OutputLookup

        self.history[slot] = OutputLookup(**self.options)

    def release(self, slot):
        del self.history[slot]

    def prefill(self, requests, sequences, predictions, *, target, chunk):
        for request, sequence, prediction in zip(requests, sequences, predictions):
            history = self.history[request.slot]
            tokens = [*sequence]
            if request.offset + len(sequence) == len(request.prompt):
                tokens.append(prediction)
            for token in tokens:
                history.append(token)

    def propose(self, requests, limits, *, target):
        return [self.history[request.slot].propose(limit) or []
                for request, limit in zip(requests, limits)]

    def repair(self, requests, outputs, positions, *, target, chunk):
        for request, sequence in zip(requests, outputs):
            for token in sequence:
                self.history[request.slot].append(token)

    def close(self):
        self.history.clear()
