"""The shared scheduler preserves exact independent and speculative stream counts."""

from types import SimpleNamespace
import numpy as np
import pytest
from akas import Engine
from akas import EngineConfig
from akas.generation.speculative.decoding import SpeculativeDecoder


class Target:
    def __init__(self, capacity=8):
        self.capacity, self.context = capacity, 100
        self.config = SimpleNamespace(vocab=1000)
        self.positions = np.zeros(capacity, "int32")
        self.live, self.releases, self.closes = set(), [], 0
        self.pending = {}

    def admit(self):
        slot = next(index for index in range(self.capacity) if index not in self.live)
        self.live.add(slot)
        self.positions[slot] = 0
        return slot

    def release(self, slot):
        self.live.remove(slot)
        self.releases.append(slot)

    def batch(self, count, chunk):
        return SimpleNamespace(run=lambda slots, sequences: self.prefill(slots, sequences))

    def prefill(self, slots, sequences):
        for slot, sequence in zip(slots, sequences):
            self.positions[slot] += len(sequence)
        return [sequence[-1] + 1 for sequence in sequences]

    def prefill_batch(self, requests, sequences, chunk):
        return self.prefill([request.slot for request in requests], sequences)

    def decode(self, slots, tokens):
        return self.prefill(slots, [[token] for token in tokens])

    def verify(self, slots, sequences, chunk):
        self.pending = dict(zip(slots, sequences))
        return [[token + 1 for token in sequence] for sequence in sequences]

    def commit(self, slots, counts):
        self.positions[slots] += counts
        self.pending.clear()

    def discard(self, slots):
        self.pending.clear()

    def close(self):
        assert not self.live
        self.closes += 1


class Proposer:
    def __init__(self):
        self.closes = 0

    def admit(self, slot): pass
    def release(self, slot): pass
    def prefill(self, *args, **kwargs): pass
    def repair(self, *args, **kwargs): pass
    def propose(self, requests, limits, **options):
        return [[request.output[-1] + step + 1 for step in range(limit)]
                for request, limit in zip(requests, limits)]
    def close(self): self.closes += 1


def drain(engine):
    while engine.step():
        pass


def test_native_scheduler_preserves_eight_independent_streams():
    model = Target()
    with Engine(model, config=EngineConfig(prefill_chunk=4, prefill_requests=8)) as engine:
        requests = [engine.submit([10 + slot] * count, max_tokens=slot + 1)
                    for slot, count in enumerate((11, 7, 2, 8, 1, 4, 13, 6))]
        drain(engine)
        for slot, request in enumerate(requests):
            assert request.output == list(range(11 + slot, 11 + slot + request.limit))
            assert model.positions[slot] == len(request.prompt) + request.limit - 1
        assert engine.scheduler.stats["tokens"] == sum(request.limit for request in requests)
        assert set(model.releases) == set(range(8))
        assert all(request.slot is None for request in requests)
        for request, event in engine.events:
            assert event["meta_info"]["prompt_tokens"] == len(request.prompt)
            assert event["meta_info"]["completion_tokens"] == len(event["output_ids"])




def test_verified_chunks_preserve_exact_stream_counts_and_completion():
    target, proposer = Target(1), Proposer()
    with Engine(SpeculativeDecoder(target, proposer, window=4)) as engine:
        request = engine.submit([12, 24], max_tokens=7)
        drain(engine)
        assert request.output == list(range(25, 32))
        events = [event for owner, event in engine.events if owner is request]
        assert events[-1]["output_ids"] == request.output
        assert events[-1]["meta_info"]["completion_tokens"] == 7
        assert events[-1]["meta_info"]["finish_reason"]["type"] == "length"
        assert engine.scheduler.stats["tokens"] == 7


def test_resident_resources_are_released_after_prefix_initialization_failure():
    target, proposer = Target(1), Proposer()
    def fail(*args, **kwargs):
        raise RuntimeError("injected prefix initialization failure")
    target.prefill_batch = fail
    with pytest.raises(RuntimeError, match="injected prefix"):
        with Engine(SpeculativeDecoder(target, proposer)) as engine:
            engine.submit([12], max_tokens=2)
            engine.step()
    assert target.closes == proposer.closes == 1
    assert not target.live


def test_resident_speculative_resources_reuse_owners_across_fresh_cohorts():
    target, proposer = Target(1), Proposer()
    backend = SpeculativeDecoder(target, proposer, window=4)
    with Engine(backend) as engine:
        for prompt in (10, 30):
            request = engine.submit([prompt, prompt], max_tokens=5)
            drain(engine)
            assert request.output == list(range(prompt + 1, prompt + 6))
            assert target.positions[0] == 6
            assert target.closes == proposer.closes == 0
        assert engine.scheduler.stats["tokens"] == 10
    assert target.closes == proposer.closes == 1
