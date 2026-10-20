"""Matrix fairness, budgets, precision requirements and retained baseline rows."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from akas.benchmarking.history import write
from akas.optimization.context import Context
from akas.optimization.planning import recipes
from tools.baselines import common
from tools import campaign


def test_required_mode_and_cache_are_context_not_schedule_options():
    context = Context(
        "weights",
        "model:Model",
        dict(batches=[1], prefix_tokens=[32], output_tokens=128),
        mode="mtp",
        kv_dtype="fp8",
    )
    assert context.metadata()["kv_dtype"] == "fp8"
    with pytest.raises(ValueError):
        replace(context, mode="draft")
    probe = dict(
        status="ok",
        provider="cuda",
        device=dict(arch="sm_89"),
        chunks=[32],
        policies=[{}],
        model_options={},
        option_variants=[],
        windows=[2, 4],
        deferred_windows=[],
        head="family:Head",
    )
    plans = recipes(context.metadata(), [probe])
    assert plans[0]["window"] is None  # Independent AR output reference.
    assert all(p["window"] for p in plans[1:])


def test_qwen_diagnostics_compare_last_active_tokens_across_chunk_sizes():
    from types import SimpleNamespace
    import numpy as np
    from akas.optimization.evaluation import diagnostic_values

    frame = SimpleNamespace(
        rows=2,
        chunk=4,
        lengths=np.array([4, 2]),
        feature=lambda _: SimpleNamespace(shape=(8, 3)),
    )
    prefill = np.arange(24).reshape(2, 12)
    observed = diagnostic_values(prefill, frame, "hidden")
    np.testing.assert_array_equal(observed, [[15, 16, 17]])
    frame.chunk = 1
    frame.lengths[:] = 1
    frame.feature = lambda _: SimpleNamespace(shape=(2, 3))
    decode = diagnostic_values(np.vstack([np.zeros((1, 3)), observed]), frame, "hidden")
    assert np.stack([observed, decode, decode]).shape == (3, 1, 3)


def test_partial_baseline_survives_later_oom_and_prompts_match_optimizer(
    tmp_path, monkeypatch
):
    from akas.optimization.evaluation import prompts

    assert common.synthetic_prompts(400, 4, 32) == prompts(400, 4, 32)
    monkeypatch.setattr(common, "_output", tmp_path / "result.json")
    config = dict(shapes=[[1, 32], [4, 32]], output_tokens=2, warmup=0, repeats=2)
    report = dict(cases=[])

    def invoke(batch, prefix):
        if batch == 4:
            raise MemoryError("GPU capacity")
        return [[1, 2]], {}

    with pytest.raises(MemoryError):
        common.measurements(config, invoke, report)
    stored = json.loads((tmp_path / "result.json").read_text())
    assert [r["status"] for r in stored["cases"]] == ["passed", "failed"]
    assert len(stored["cases"][0]["samples"]) == 2


def test_matrix_resumes_without_repeating_completed_entries(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = tmp_path / "spec.json"
    write(
        spec,
        dict(
            models=["lfm2.5-230m"],
            formats=["bf16"],
            backends=["cuda", "webgpu"],
            caches=["default"],
            workload="standard-v1",
        ),
    )
    calls = []

    def optimize(context, out, **kwargs):
        calls.append((context.backend, kwargs["budget"]))
        return dict(status="complete", deployment=str(out / "deployment.json"))

    monkeypatch.setattr(campaign, "optimize", optimize)
    state = campaign.run(
        spec,
        tmp_path / "out",
        budget=60,
        slice_seconds=10,
        history=tmp_path / "history",
    )
    assert state["status"] == "complete"
    assert calls == [("cuda", 10), ("webgpu", 10)]
    campaign.run(
        spec,
        tmp_path / "out",
        budget=60,
        slice_seconds=10,
        history=tmp_path / "history",
    )
    assert len(calls) == 2
    with pytest.raises(ValueError, match="budget changed"):
        campaign.run(
            spec,
            tmp_path / "out",
            budget=61,
            slice_seconds=10,
            history=tmp_path / "history",
        )

    monkeypatch.setenv("AKAS_SHAPE_TIMEOUT", "2")
    with pytest.raises(ValueError, match="measurement limits"):
        campaign.run(
            spec,
            tmp_path / "out",
            budget=60,
            slice_seconds=10,
            history=tmp_path / "history",
        )


def test_mtp_cannot_establish_its_own_ar_reference(tmp_path, monkeypatch):
    from akas.optimization import supervisor
    import akas.cli

    calls = []
    monkeypatch.setattr(akas.cli, "checkpoint_digest", lambda _: "weights")
    monkeypatch.setattr(supervisor, "software", lambda: {})
    probe = dict(
        status="ok",
        provider="cuda",
        device=dict(name="test", arch="sm_89"),
        chunks=[1],
        policies=[{}],
        model_options={},
        windows=[2],
        head="family:Head",
    )

    def execute(job, directory, **kwargs):
        if job["action"] == "probe":
            return probe
        calls.append(job["recipe"]["window"])
        return dict(status="failed", error="AR model reference failed")

    monkeypatch.setattr(supervisor, "execute", execute)
    context = Context(
        "weights",
        "family:Model",
        dict(batches=[1], prefix_tokens=[32], output_tokens=128),
        backend="cuda",
        mode="mtp",
    )
    result = supervisor.optimize(
        context, tmp_path / "out", budget=30, history=tmp_path / "history"
    )
    assert calls == [None]
    assert result["status"] == "failed" and not result["deployment"]


def test_large_workload_deadlines_use_observed_completed_checks():
    from akas.optimization.planning import worker_allowance

    jobs = [dict(status="passed", recipe=dict(provider="cuda"), elapsed_seconds=800)]
    assert worker_allowance(1800, 1800, jobs, reference={}, provider="cuda") == 1200
    assert worker_allowance(1800, 300, jobs, reference={}, provider="cuda") == 300
    assert worker_allowance(1800, 1800, jobs, reference=None, provider="webgpu") == 810


def test_nondeterministic_baseline_keeps_all_repeats_and_later_shapes(
    tmp_path, monkeypatch
):
    import sys

    calls = {}

    def invoke(batch, prefix):
        calls[batch] = calls.get(batch, 0) + 1
        token = calls[batch] % 2 if batch == 1 else 1
        return [[token, 2] for _ in range(batch)], {}

    config = dict(shapes=[[1, 32], [4, 32]], output_tokens=2, warmup=0, repeats=3)
    write(tmp_path / "config.json", config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "--config",
            str(tmp_path / "config.json"),
            "--out",
            str(tmp_path / "out.json"),
        ],
    )

    def run(config):
        report = dict(cases=[])
        common.measurements(config, invoke, report)
        return report

    with pytest.raises(SystemExit) as error:
        common.main(run)
    assert error.value.code == 1
    report = json.loads((tmp_path / "out.json").read_text())
    assert report["status"] == "failed"
    assert [r["status"] for r in report["cases"]] == ["failed", "passed"]
    assert all(len(r["samples"]) == 3 and r["output_count"] for r in report["cases"])
    assert calls == {1: 3, 4: 3}


def test_speculative_prefill_has_the_exact_teacher_geometry():
    from akas.optimization.planning import geometries

    data = dict(
        batches=[1, 4, 16, 64, 256], prefix_tokens=[32, 2048], output_tokens=128
    )
    recipe = dict(chunk=32, prefill_requests=32, window=2)
    shapes = geometries(data, recipe)
    assert (32, 32) in shapes and (32, 32, "proposal") in shapes
    assert (256, 32, "proposal") not in shapes
    assert (256, 2, "verify") in shapes and (256, 2, "proposal") in shapes
    assert (256, 1, "proposal") in shapes
    for b, t, *phase in shapes:
        if not phase and t > 1:
            assert (b, t, "proposal") in shapes
    assert all(len(s) == 2 for s in geometries(data, {**recipe, "window": None}))


def test_large_prefill_tiles_keep_a_short_prompt_geometry():
    from akas.optimization.planning import geometries

    data = dict(batches=[1], prefix_tokens=[32, 2048, 130944], output_tokens=128)
    recipe = dict(chunk=512, prefill_requests=1, window=None)
    assert geometries(data, recipe) == [(1, 1), (1, 32), (1, 512)]
    speculative = geometries(data, {**recipe, "window": 4})
    assert (1, 32, "proposal") in speculative
    assert (1, 512, "proposal") in speculative
    assert (1, 4, "verify") in speculative
