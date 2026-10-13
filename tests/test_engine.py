"""Engine lifecycle consumes the same scheduler contract across model adapters."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from akas import Engine, EngineConfig
from test_dense_scheduler import FakeEngine


class Adapter(FakeEngine):
    config = SimpleNamespace(vocab=100)
    context = 128

    def close(self):
        assert not self.state
        self.closed = True


def test_engine_completion_cancellation_and_close_release_request_state():
    model = Adapter(1)
    with Engine(model, config=EngineConfig(prefill_chunk=2)) as engine:
        first = engine.submit([1, 2, 3], max_tokens=20)
        second = engine.submit([8], max_tokens=2)
        assert engine.step()
        engine.cancel(first)
        while engine.step():
            pass
        assert first.slot is None and second.output == [8, 16]
        assert not model.state
        assert (
            sum(
                event["meta_info"]["finish_reason"] is not None
                for _, event in engine.events
            )
            == 1
        )
        pending = engine.submit([2, 3, 4], max_tokens=20)
        engine.step()
        assert pending.slot is not None
    assert pending.slot is None and model.closed
    engine.close()
    with pytest.raises(RuntimeError, match="stopped"):
        engine.submit([1], max_tokens=1)


def test_engine_rejects_invalid_requests_and_wrong_thread_before_admission():
    model = Adapter(1)
    with Engine(model) as engine:
        for prompt, limit in [([], 1), ([True], 1), ([100], 1), ([1], 0), ([1], 128)]:
            with pytest.raises(ValueError):
                engine.submit(prompt, max_tokens=limit)
        engine.submit([1], max_tokens=2)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="owner thread"):
                pool.submit(engine.step).result()
        assert not model.state


def test_checkpoint_factory_and_adapter_are_supplied_by_the_caller():
    opened = []

    def load(checkpoint, bundle, device, *, capacity):
        opened.append((checkpoint, bundle, device))
        return Adapter(capacity)

    with Engine.from_checkpoint(
        load, "new-model", "compiled", None, capacity=1
    ) as engine:
        request = engine.submit([3], max_tokens=2)
        while engine.step():
            pass
        assert request.output == [3, 6]
    assert opened == [("new-model", "compiled", None)]


def test_failed_adapter_construction_releases_the_loaded_model():
    model = Adapter(1)

    def fail_adapter(value):
        raise ValueError("adapter failed")

    with pytest.raises(ValueError, match="adapter failed"):
        Engine.from_checkpoint(lambda *args: model, "", "", None, adapter=fail_adapter)
    assert model.closed
