"""Timing/accounting invariants, reproducible workloads and regression matching."""

from types import SimpleNamespace
import json
import numpy as np
import pytest
from akas.benchmarking import Case
from akas.benchmarking import matrix
from akas.benchmarking import run
from akas.benchmarking.report import compare
from akas.benchmarking.report import write_report
from akas.benchmarking.report import table


class Session:
    batch, capacity, allocated_bytes = 4, 1000, 1024
    config = SimpleNamespace(vocab=31)

    def __init__(self):
        self.calls = []
        self.device = SimpleNamespace(synchronize=lambda: self.calls.append("sync"))

    def reset(self):
        self.calls.append("reset")

    def prefill(self, prompts, *, read):
        assert not read
        self.calls.append([p.tolist() for p in prompts])

    def decode(self, n, *, read):
        assert not read
        self.calls.append(n)


def test_program_session_continues_device_feedback_and_releases_owned_requests():
    from akas.benchmarking.session import Session as ProgramSession

    calls = []
    available = iter([3, 1])
    base = SimpleNamespace(nbytes=64)
    view = SimpleNamespace(_base=base, nbytes=64)
    executor = SimpleNamespace(
        capacity=4, context=32, device=object(), config=SimpleNamespace(vocab=73),
        program=dict(model_options=dict(kv_dtype="fp8")),
        weights={"weight": base},
        batches=SimpleNamespace(states={}, frames={"decode": SimpleNamespace(buffers={"view": view})}),
        admit=lambda: next(available), release=lambda slot: calls.append(("release", slot)),
        prefill=lambda slots, prompts, **options: [11, 12],
        decode_steps=lambda slots, tokens, steps, **options: calls.append(("decode", tuple(slots), tokens, steps, options)),
    )
    with ProgramSession(executor, batch=2, chunk=4) as session:
        session.reset()
        assert session.allocated_bytes == 64
        session.prefill([np.array([1, 2]), np.array([3])], read=False)
        session.decode(3, read=False)
        session.decode(2, read=False)
    assert calls == [("decode", (3, 1), [11, 12], 3, dict(read=False)),
                     ("decode", (3, 1), None, 2, dict(read=False)),
                     ("release", 1), ("release", 3)]


def test_program_session_reclaims_admission_when_batch_setup_fails():
    from akas.benchmarking.session import Session as ProgramSession

    released = []
    def admit():
        if not hasattr(admit, "called"):
            admit.called = True
            return 2
        raise RuntimeError("capacity exhausted")
    executor = SimpleNamespace(capacity=4, context=32, device=object(), config=object(),
                               program={}, admit=admit, release=released.append)
    session = ProgramSession(executor, batch=2)
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        session.reset()
    assert released == [2] and session.slots == []


def test_synchronized_measurements_count_only_executed_tokens():
    session = Session()
    # Each trial measures context 2s, prompt 4s, decode 8s.
    times = iter([0, 2, 2, 6, 6, 14] * 3)
    result = run(
        session,
        Case(4, 5, generated=2, depth=10, mixed=True),
        repeats=2,
        warmup=1,
        clock=lambda: next(times),
    )
    assert result["prompt_tokens"] == 14
    assert result["generated_tokens"] == 8
    assert result["summary"]["prefill_tokens_per_second"]["mean"] == 3.5
    assert result["decode_tokens"] == 4
    assert result["summary"]["decode_tokens_per_second"]["mean"] == 0.5
    assert result["summary"]["decode_tokens_per_second_per_request"]["mean"] == 0.125
    assert session.calls.count(1) == 3
    assert result["summary"]["ttft_seconds"]["mean"] == 4
    assert len(result["samples"]) == 2
    assert session.calls.count("sync") == 18
    prompts = [call for call in session.calls if isinstance(call, list)]
    assert prompts[0] == prompts[2] == prompts[4]
    assert prompts[1] == prompts[3] == prompts[5]


@pytest.mark.parametrize("case", [Case(1, 2), Case(4, 995, generated=10)])
def test_mismatched_session_is_rejected(case):
    with pytest.raises(ValueError, match="batch/capacity"):
        run(Session(), case)


def test_invalid_workload_settings_fail_before_measurement():
    for args in ((True, 2), (0, 2), (1, -1)):
        with pytest.raises(ValueError):
            Case(*args)
    with pytest.raises(ValueError):
        run(Session(), Case(4, 2), repeats=0)
    session = Session()
    with pytest.raises(ValueError, match="two total output"):
        run(session, Case(4, 2, generated=1))
    assert session.calls == []
    assert len(matrix()) == 10
    assert {c.batch for c in matrix()} == {1, 4, 16, 64, 256}
    assert {c.prompt for c in matrix()} == {32, 2048}


def test_report_roundtrip_and_matched_regression(tmp_path):
    session = Session()
    times = iter([0, 2, 2, 6])
    row = run(session, Case(4, 5), repeats=1, warmup=0, clock=lambda: next(times))
    metadata = dict(
        checkpoint_sha256="same",
        definition="test",
        device="gpu",
        kv_dtype="fp8",
        provider="cuda",
        protocol="native-token-step-v1",
        graphs=True,
    )
    report = write_report(tmp_path / "report.json", metadata, [row])
    restored = json.loads((tmp_path / "report.json").read_text())
    assert restored == report
    assert compare(report, restored)[0]["ratios"] == dict(
        prefill_tokens_per_second=1, decode_tokens_per_second=1
    )
    restored["metadata"]["kv_dtype"] = "float16"
    with pytest.raises(ValueError, match="baseline differs"):
        compare(report, restored)
    assert "tg/request" in table([row])


def test_scaling_guard_catches_prefill_cliff_even_when_decode_improves():
    from akas.benchmarking.report import scaling_regressions

    def result(batch, pp, tg, prompt=2048):
        return dict(
            status="ok",
            case=Case(batch, prompt).metadata(),
            summary=dict(
                prefill_tokens_per_second=dict(mean=pp),
                decode_tokens_per_second=dict(mean=tg),
            ),
        )

    rows = [
        result(64, 112049, 10438),
        result(256, 44418, 17133),
        result(1, 9000, 761, prompt=32),
    ]
    losses = scaling_regressions(rows, 0.2)
    assert len(losses) == 1
    assert losses[0]["metric"] == "prefill_tokens_per_second"
    assert losses[0]["previous_batch"] == 64
    assert losses[0]["case"]["batch"] == 256
    rows[1] = result(256, 113209, 16978)
    assert scaling_regressions(rows, 0.2) == []


def test_program_session_forced_step_and_feature_read_use_the_last_phase():
    from akas.benchmarking.session import Session as ProgramSession
    calls = []
    values = np.arange(12, dtype="float32").reshape(4, 3)
    frame = SimpleNamespace(rows=4, features={"logits": "declared"},
                            feature=lambda name: SimpleNamespace(to_numpy=lambda: values))
    executor = SimpleNamespace(capacity=4, context=32, device=object(), config=object(), program={},
        admit=lambda: 0, release=lambda slot: None,
        decode_steps=lambda slots, tokens, steps, **options: calls.append((tokens, steps, options)) or [8],
        batches=SimpleNamespace(select=lambda rows, chunk: frame))
    session = ProgramSession(executor, batch=1)
    session.reset()
    assert session.step([7]) == [8]
    assert calls == [([7], 1, {"read": True})]
    actual = session.read_feature("logits")
    np.testing.assert_array_equal(actual, [[0, 1, 2]])
    actual[:] = 99
    assert values[0, 0] == 0
    session.close()
    with pytest.raises(ValueError, match="completed token phase"):
        session.read_feature("logits")
