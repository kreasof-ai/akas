"""All forward bindings preserve one request lifecycle and row mapping contract."""

from types import SimpleNamespace
from threading import Thread

import numpy as np
import pytest

from akas.execution.executor import Executor


@pytest.fixture
def executor(monkeypatch):
    model = Executor(capacity=8, context=64)
    model.config = SimpleNamespace(vocab=100)
    model.values = np.zeros(8, dtype="int32")
    monkeypatch.setattr(model, "_close_resources", lambda: None)
    def select(count, chunk, **options):
        def run(slots, sequences, positions):
            result = []
            for slot, sequence in zip(slots, sequences):
                for token in sequence:
                    model.values[slot] = (3 * model.values[slot] + token) % 100
                result.append(int(model.values[slot]))
            return result
        return SimpleNamespace(run=run)
    model.batches = SimpleNamespace(select=select, owner=model)
    model._reset_state = lambda slot: model.values.__setitem__(slot, 0)
    yield model
    model.close()


def test_sparse_reordered_requests_ragged_prefill_and_reuse(executor):
    first, middle, last = [executor.admit() for _ in range(3)]
    executor.release(middle)
    assert executor.prefill([last, first], [[7], [1, 2, 3]], chunk=2) == [7, 18]
    assert executor.positions[[first, middle, last]].tolist() == [3, 0, 1]
    assert executor.decode([last, first], [7, 18]) == [28, 72]
    assert executor.admit() == middle
    assert executor.decode([middle], [4]) == [4]
    executor.release(first)
    assert executor.admit() == first
    assert executor.decode([first], [1]) == [1]


@pytest.mark.parametrize(
    "slots,sequences",
    [
        ([0, 0], [[1], [2]]),
        ([0, 7], [[1], [2]]),
        ([0], [[100]]),
        ([0], [[]]),
        ([0], [[1], [2]]),
    ],
)
def test_invalid_batch_does_not_advance_any_request(executor, slots, sequences):
    executor.admit()
    before = executor.positions.copy()
    with pytest.raises(ValueError):
        executor.run_batch(slots, sequences)
    np.testing.assert_array_equal(executor.positions, before)
    assert executor.decode([0], [1]) == [1]


def test_pending_verification_protects_release_and_forward(executor):
    slot = executor.admit()
    executor.pending[slot] = object()
    with pytest.raises(ValueError, match="verification"):
        executor.release(slot)
    with pytest.raises(ValueError, match="verification"):
        executor.decode([slot], [1])
    assert executor.occupied[slot] and executor.positions[slot] == 0
    executor.pending.clear()
    executor.release(slot)
    with pytest.raises(ValueError, match="not owned"):
        executor.release(slot)


def test_context_overflow_is_rejected_before_any_chunk(executor):
    slot = executor.admit()
    executor.positions[slot] = executor.context - 1
    with pytest.raises(ValueError):
        executor.prefill([slot], [[1, 2]], chunk=1)
    assert executor.positions[slot] == executor.context - 1


def test_close_and_mutation_remain_on_owner_thread(executor):
    failures = []

    def mutate():
        for operation in (executor.admit, executor.close):
            try:
                operation()
            except RuntimeError as error:
                failures.append(str(error))

    thread = Thread(target=mutate)
    thread.start()
    thread.join()
    assert len(failures) == 2 and all("owner thread" in e for e in failures)
    assert not executor.closed and not executor.occupied.any()
    executor.close()
    executor.close()
    with pytest.raises(RuntimeError, match="closed"):
        executor.admit()


def test_scheduler_uses_same_executor_for_all_bindings(executor):
    from akas import Engine
    from akas import EngineConfig

    with Engine(executor, config=EngineConfig(prefill_chunk=2)) as engine:
        requests = [engine.submit([1, 2, 3], max_tokens=4) for _ in range(12)]
        while engine.step():
            pass
        assert [r.output for r in requests] == [[18, 72, 88, 52]] * 12
        assert not executor.occupied.any() and not executor.pending
        assert engine.scheduler.stats["completed"] == 12


def test_scheduled_text_generation_releases_only_its_own_slot(executor):
    tokenizer = SimpleNamespace(
        encode=lambda _: [1, 2], eos=None, decode=lambda ids: str(ids)
    )
    peer = executor.admit()
    executor.decode([peer], [7])
    before = executor.positions[peer]
    result = executor.generate("text", max_tokens=2, chat=False, tokenizer=tokenizer)
    assert result["generated_tokens"] == [5, 20]
    assert executor.positions[peer] == before
    assert executor.occupied.sum() == 1


def test_generation_error_reclaims_an_admitted_slot(executor, monkeypatch):
    tokenizer = SimpleNamespace(encode=lambda _: [1], eos=None)

    def fail(*args, **kwargs):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(executor, "prefill", fail)
    with pytest.raises(RuntimeError, match="forward failed"):
        executor.generate("text", max_tokens=2, chat=False, tokenizer=tokenizer)
    assert not executor.occupied.any()
