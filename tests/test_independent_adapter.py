"""The common executor isolates unrelated recurrence definitions and lifetimes."""

from types import SimpleNamespace

import numpy as np
import pytest

from akas import Engine, EngineConfig
from akas.execution.executor import Executor


class Counter(Executor):
    multiplier = 1
    config = SimpleNamespace(vocab=100)

    def __init__(self, capacity=2):
        super().__init__(capacity=capacity, context=64, reset=self.reset_slot)
        self.values = np.zeros(capacity, dtype="int32")
        self.fail_reset = False
        self.resources_closed = False

    def reset_slot(self, slot):
        if self.fail_reset:
            raise RuntimeError("injected reset failure")
        self.values[slot] = 0

    def _execute_batch(self, slots, sequences, chunk):
        for slot, sequence in zip(slots, sequences):
            for token in sequence:
                self.values[slot] = (self.multiplier * self.values[slot] + token) % 100
            self.positions[slot] += len(sequence)
        return self.values[slots].astype(int).tolist()

    def _close_resources(self):
        self.resources_closed = True


class Recurrent(Counter):
    multiplier = 3


@pytest.mark.parametrize("kind,expected", [(Counter, [6, 12, 24, 48]), (Recurrent, [18, 72, 88, 52])])
def test_sequences_complete_cancel_and_reuse_without_provider_fields(kind, expected):
    model = kind()
    with Engine(model, config=EngineConfig(prefill_chunk=2)) as engine:
        first = engine.submit([1, 2, 3], max_tokens=4)
        second = engine.submit([7], max_tokens=2)
        while engine.step():
            pass
        assert first.output == expected
        assert second.output == ([7, 14] if kind is Counter else [7, 28])
        assert not model.occupied.any() and not model.closed
        replacement = engine.submit([1, 2, 3], max_tokens=4)
        while engine.step():
            pass
        assert replacement.output == first.output
        cancelled = engine.submit([2], max_tokens=20)
        assert engine.step()
        engine.cancel(cancelled)
        while engine.step():
            pass
        assert cancelled.slot is None and not model.occupied.any()
    assert model.closed and model.resources_closed
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.admit()


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_capacity_validation_before_resources(capacity):
    with pytest.raises(ValueError, match="capacity"):
        Counter(capacity)


def test_failed_reset_does_not_claim_or_mutate_another_slot():
    model = Counter()
    assert model.admit() == 0
    model.values[0] = 73
    model.fail_reset = True
    with pytest.raises(RuntimeError, match="reset failure"):
        model.admit()
    assert model.occupied.tolist() == [True, False] and model.values[0] == 73
    model.fail_reset = False
    assert model.admit() == 1
    with pytest.raises(ValueError, match="exhausted"):
        model.admit()
    model.release(1)
    model.release(0)
    assert not model.closed and not model.occupied.any()
    model.close()


def test_engine_close_releases_both_active_sequences():
    model = Recurrent()
    with Engine(model, config=EngineConfig(prefill_chunk=1)) as engine:
        first = engine.submit([1, 2, 3], max_tokens=4)
        second = engine.submit([7, 8, 9], max_tokens=2)
        engine.step()
        assert model.occupied.tolist() == [True, True]
    assert first.slot is None and second.slot is None
    assert model.closed and not model.occupied.any()
