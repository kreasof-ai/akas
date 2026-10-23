"""Native block measurements count actual emitted tokens and release all slots."""

from types import SimpleNamespace as NS
import numpy as np
import pytest

from akas.benchmarking.cases import Case
from akas.benchmarking.runner import run
from akas.benchmarking.speculative import SpeculativeSession


class Decoder:
    def __init__(self):
        base = NS(nbytes=64)
        alias = NS(_base=base)
        self.capacity, self.context = 4, 256
        self.target = NS(capacity=4, context=256, config=NS(vocab=100),
                         positions=np.zeros(4, dtype="int32"), device=NS(synchronize=lambda: None),
                         program=dict(model_options=dict(kv_dtype="float16")),
                         weights=dict(head=base), batches=NS(states={}, frames={}))
        draft = NS(weights=dict(head=alias), batches=NS(states={}, frames={}))
        self.proposer = NS(draft=draft)
        self.stats = dict(rounds=0, emitted=0)
        self.slots, self.chunks, self.decoded = set(), [], []

    def admit(self):
        slot = next(i for i in range(4) if i not in self.slots)
        self.slots.add(slot)
        self.target.positions[slot] = 0
        return slot

    def release(self, slot):
        self.slots.remove(slot)

    def prefill_geometry(self, chunk):
        return chunk

    def prefill_batch(self, requests, sequences, chunk):
        self.chunks.append([len(sequence) for sequence in sequences])
        for request, sequence in zip(requests, sequences):
            assert request.prompt[request.offset:request.offset + len(sequence)] == sequence
            self.target.positions[request.slot] += len(sequence)
        return [(sequence[-1] + 1) % 100 for sequence in sequences]

    def decode_many(self, requests):
        self.stats["rounds"] += 1
        result = []
        for request in requests:
            count = min(3, request.limit - len(request.output))
            result.append([(request.output[-1] + step) % 100 for step in range(1, count + 1)])
            self.target.positions[request.slot] += count
            self.stats["emitted"] += count
        self.decoded.append([len(row) for row in result])
        return result


def test_native_speculation_measures_real_blocks_and_clips_final_round():
    decoder = Decoder()
    with SpeculativeSession(decoder, batch=2, chunk=3) as session:
        report = run(session, Case(2, 8, generated=8, depth=3, mixed=True), repeats=1, warmup=0)
        assert decoder.chunks == [[3, 3], [3, 3], [3, 3], [2, 1]]
        assert decoder.decoded == [[3, 3], [3, 3], [1, 1]]
        assert [len(request.output) for request in session.requests] == [8, 8]
        assert session.stats == dict(rounds=3, emitted=14)
        assert report["decode_tokens"] == 14 and report["generated_tokens"] == 16
        assert report["allocated_bytes"] == 64  # Target and draft borrow one head allocation.
    assert not decoder.slots


def test_incremental_decode_preserves_tokens_and_request_state():
    decoder = Decoder()
    with SpeculativeSession(decoder, batch=1, chunk=4) as session:
        session.reset()
        assert session.prefill([[1, 2, 3]]) == [4]
        assert session.decode(4) == [8]
        assert session.decode(2) == [10]
        assert session.requests[0].output == list(range(4, 11))
        assert decoder.target.positions[0] == 9
        session.reset()
        assert decoder.target.positions[0] == 0
    assert not decoder.slots


def test_timeout_is_checked_between_prefill_chunks_and_reclaims_requests(monkeypatch):
    decoder = Decoder()
    clocks = iter([0, 0, 2])
    monkeypatch.setattr("akas.benchmarking.speculative.time.perf_counter", lambda: next(clocks))
    with pytest.raises(TimeoutError):
        with SpeculativeSession(decoder, batch=1, chunk=3, timeout=1) as session:
            session.reset()
            session.prefill([[1] * 7])
    assert decoder.chunks == [[3]] and not decoder.slots


def test_decode_failure_releases_both_models_requests(monkeypatch):
    decoder = Decoder()

    def fail(requests):
        raise RuntimeError("verification failed")

    monkeypatch.setattr(decoder, "decode_many", fail)
    with pytest.raises(RuntimeError, match="verification"):
        with SpeculativeSession(decoder, batch=2, chunk=3) as session:
            session.reset()
            session.prefill([[1], [2]])
            session.decode(7)
    assert not decoder.slots


def test_evaluator_measures_speculative_backend_and_separates_native_statistics(monkeypatch):
    from akas.optimization import evaluation
    from akas.generation.speculative import decoding

    decoder = Decoder()
    calls = []

    def coordinator(target, proposer, **options):
        assert target is decoder.target and proposer == "paired head"
        assert options["capture"] is False
        return decoder

    def head(target, **options):
        calls.append(options)
        return "paired head"

    def measured(engine, batch, prefix, output, **kwargs):
        decoder.stats["rounds"] += 3
        decoder.stats["emitted"] += output - 1
        anchor = int(evaluation.prompts(100, batch, prefix)[0][-1]) + 1
        return dict(seconds=1, output_tokens_per_second=output, ttft_ms=100,
                    tpot_ms=100, output_ids=[[(anchor + i) % 100 for i in range(output)]])

    monkeypatch.delenv("AKAS_SKIP_AUTOTUNE", raising=False)
    monkeypatch.setenv("AKAS_BENCHMARK_WARMUP", "0")
    monkeypatch.setenv("AKAS_BENCHMARK_REPEATS", "2")
    monkeypatch.setattr(decoding, "SpeculativeDecoder", coordinator)
    monkeypatch.setattr(evaluation, "symbol", lambda name: head)
    monkeypatch.setattr(evaluation, "diagnostic", lambda *args, **kw: np.ones((1,)))
    monkeypatch.setattr(evaluation, "Engine", lambda backend, **kw:
                        NS(model=backend, scheduler=NS(stats={}), close=lambda: None))
    monkeypatch.setattr(evaluation, "measured_requests", measured)
    rows, _, stats = evaluation.evaluate(decoder.target,
        dict(chunk=3, window=4, prefill_requests=1, graphs=True, deferred=False,
             round_capture=False, head="factory", head_options=dict(bundle="draft"), mode="dspark"),
        dict(batches=[1], prefix_tokens=[8], output_tokens=8))
    assert calls == [dict(bundle="draft")]
    assert rows[0]["mode"] == rows[0]["native_phase_mode"] == "dspark"
    assert rows[0]["native_output_ids"] == rows[0]["output_ids"]
    assert rows[0]["decode_tokens"] == 7
    assert rows[0]["native_speculation"] == dict(rounds=3, emitted=7)
    assert stats["speculation"] == dict(rounds=6, emitted=14)
    assert not decoder.slots
