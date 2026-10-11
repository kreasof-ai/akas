"""Independent scheduling against unrelated request implementations."""

from threading import get_ident
from types import SimpleNamespace

import pytest

from akas import Engine, EngineConfig
from akas.execution.handles import HandleExecutor


class Handle:
    def __init__(self, multiplier=1, *, fail_reset=False):
        self.multiplier = multiplier
        self.fail_reset = fail_reset
        self.state = 0
        self.closed = False
        self.resets = 0
        self.thread = get_ident()

    def reset(self):
        assert not self.closed and get_ident() == self.thread
        if self.fail_reset:
            raise RuntimeError("injected reset failure")
        self.state = 0
        self.resets += 1

    def next_token(self, tokens):
        assert not self.closed and get_ident() == self.thread
        for token in tokens:
            self.state = (self.multiplier * self.state + token) % 100
        return self.state

    def close(self):
        assert get_ident() == self.thread
        self.closed = True


class Counter(Handle):
    config = SimpleNamespace(vocab=100)
    context = 64

    def __init__(self):
        super().__init__()
        self.children = []
        self.fail_child = False

    def new_request(self):
        child = Handle(self.multiplier, fail_reset=self.fail_child)
        self.children.append(child)
        return child

    def close(self):
        for child in self.children:
            child.close()
        super().close()


class Recurrent(Counter):
    def __init__(self):
        super().__init__()
        self.multiplier = 3


@pytest.mark.parametrize(
    "kind,expected", [(Counter, [6, 12, 24, 48]), (Recurrent, [18, 72, 88, 52])]
)
def test_independent_handles_schedule_complete_cancel_and_reuse_without_provider_fields(
    kind, expected
):
    model = kind()
    with Engine.from_independent(
        model, capacity=2, config=EngineConfig(prefill_chunk=2)
    ) as engine:
        first = engine.submit([1, 2, 3], max_tokens=4)
        second = engine.submit([7], max_tokens=2)
        while engine.step():
            pass
        assert first.output == expected
        assert second.output == ([7, 14] if kind is Counter else [7, 28])
        assert not engine.model.handles
        assert not model.closed and model.children[0].closed
        replacement = engine.submit([1, 2, 3], max_tokens=4)
        while engine.step():
            pass
        assert replacement.output == first.output
        cancelled = engine.submit([2], max_tokens=20)
        assert engine.step()
        engine.cancel(cancelled)
        while engine.step():
            pass
        assert cancelled.slot is None and not engine.model.handles
    assert model.closed
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.model.admit()


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_capacity_validation_does_not_take_model_ownership(capacity):
    model = Counter()
    with pytest.raises(ValueError, match="capacity"):
        HandleExecutor(model, capacity=capacity)
    assert not model.closed and not model.children


def test_capacity_and_failed_child_reset_leave_no_orphan_handle():
    model = Counter()
    adapter = HandleExecutor(model, capacity=2)
    assert adapter.admit() == 0
    model.fail_child = True
    with pytest.raises(RuntimeError, match="reset failure"):
        adapter.admit()
    assert model.children[0].closed and set(adapter.handles) == {0}
    model.fail_child = False
    assert adapter.admit() == 1
    with pytest.raises(ValueError, match="exhausted"):
        adapter.admit()
    adapter.release(1)
    assert model.children[1].closed
    adapter.release(0)
    assert not model.closed
    adapter.close()
    assert model.closed and not adapter.handles


def test_engine_close_releases_both_active_independent_sequences():
    model = Recurrent()
    with Engine.from_independent(
        model, capacity=2, config=EngineConfig(prefill_chunk=1)
    ) as engine:
        first = engine.submit([1, 2, 3], max_tokens=4)
        second = engine.submit([7, 8, 9], max_tokens=2)
        engine.step()
        assert set(engine.model.handles) == {0, 1}
    assert first.slot is None and second.slot is None
    assert model.closed and model.children[0].closed
