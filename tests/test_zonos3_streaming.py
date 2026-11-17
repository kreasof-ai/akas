"""Exercise request admission and slot ownership without model weights."""

from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest

from akas.execution.state import StatePool
from akas.models.zonos3.streaming import Zonos3StreamingEngine


class Runtime:
    def __init__(self):
        self.pool = StatePool(2)
        self.max_requests = 2
        self.context = 512
        self.config = SimpleNamespace(codebooks=10)
        self.checkpoint = SimpleNamespace(audio_pad_id=4097)
        self._codec_frame_caps = (1, 4, 15)
        self._silence_rows = [np.zeros(10, dtype="int32")] * 3
        self.scheduled = Event()
        self.primed = []

    def prepare_prompt(self, text, max_tokens):
        if text == "too long":
            raise ValueError("ZONOS3 prompt exceeds compiled prefill/context")
        return np.zeros((50, 11), dtype="int32")

    def prime_slots(self, slots):
        self.primed.extend(slots)

    def prefill_jobs(self, jobs):
        for job in jobs:
            assert job.prompt.shape == (50, 11)
            job.finished = True

    def flush_jobs(self, jobs):
        return [np.empty(0, dtype="float32") for _ in jobs]


def test_invalid_prompt_does_not_poison_other_requests_in_admission_batch():
    rt = Runtime()
    engine = Zonos3StreamingEngine(rt)
    rejected = []
    callbacks = dict(
        on_chunk=lambda _: None, on_done=rt.scheduled.set, on_error=rejected.append
    )
    bad = engine.submit("too long", **callbacks)
    good = engine.submit("delivery", **callbacks)
    worker = Thread(target=engine.run_forever)
    worker.start()
    try:
        assert rt.scheduled.wait(2)
    finally:
        engine.stop()
        worker.join(2)
    assert not worker.is_alive()
    assert rejected == ["ZONOS3 prompt exceeds compiled prefill/context"]
    assert rt.primed == [1]
    assert bad.slot == good.slot == -1
    assert not rt.pool.occupied.any()


@pytest.mark.parametrize(
    "options",
    [
        {"seed": -1},
        {"temperature": float("nan")},
        {"max_tokens": 0},
        {"stream_codebooks": 11},
    ],
)
def test_invalid_controls_do_not_enter_admission_queue(options):
    engine = Zonos3StreamingEngine(Runtime())
    with pytest.raises(ValueError):
        engine.submit("delivery", on_chunk=None, on_done=None, on_error=None, **options)
    assert not engine._waiting


def test_stop_releases_active_slots_without_admitting_queued_requests():
    rt = Runtime()
    engine = Zonos3StreamingEngine(rt)
    errors = []
    callbacks = dict(on_chunk=None, on_done=None, on_error=errors.append)
    active = engine.submit("active", **callbacks)
    waiting = engine.submit("waiting", **callbacks)
    engine._waiting.popleft()
    active.slot = rt.pool.reserve(0, reset=lambda _: None)
    engine._active.append(active)
    engine.stop()
    engine.run_forever()
    assert errors == ["server stopped", "server stopped"]
    assert not rt.pool.occupied.any()
    assert not rt.primed
    assert waiting.slot == active.slot == -1
