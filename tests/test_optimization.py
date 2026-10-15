"""Optimization intent, complete-workload selection, durable search and supervision."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from akas.benchmarking.history import digest, write
from akas.compilation.tuning import tune, Workload, Measurement
from akas.kernels.contracts import Implementation
from akas.optimization.context import Context, discover, duration, workload
from akas.optimization.planning import recipes, refine, score
from akas.optimization.supervisor import execute, lock, verify_result


def context():
    return dict(
        checkpoint="weights",
        definition="example:Model",
        workload=dict(batches=[1, 4], prefix_tokens=[32, 2048], output_tokens=128),
        backend="auto",
        device=0,
        memory_bytes=None,
        ttft_ms=None,
        tpot_ms=None,
    )


def report(factor=1):
    return dict(
        metadata=dict(run_id="baseline"),
        results=[
            dict(
                case=dict(batch=b, prompt=p, generated=128),
                status="ok",
                correctness=dict(correct=True),
                serving=dict(
                    seconds=(b + p) * factor, ttft_ms=4 * factor, tpot_ms=factor
                ),
            )
            for b in (1, 4)
            for p in (32, 2048)
        ],
    )


def test_intent_uses_workload_and_constraints_without_execution_hyperparameters(
    tmp_path,
):
    assert duration("30m") == 1800
    assert duration("4h") == 14400
    assert workload()["batches"] == [1, 4, 16, 64, 256]
    for invalid in ("nan", "-1", "0m", "1week"):
        with pytest.raises(ValueError):
            duration(invalid)
    with pytest.raises(ValueError):
        Context(**{**context(), "memory_bytes": -1})
    p = tmp_path / "workload.json"
    write(
        p, dict(batches=[1], prefix_tokens=[32], output_tokens=128, prefix_cache=True)
    )
    with pytest.raises(ValueError, match="prefix_cache"):
        workload(p)


def test_model_discovery_comes_from_family_declarations(tmp_path):
    for architecture, expected in (
        ("lfm2", "LFM2"),
        ("qwen3_5_text", "Qwen35Dense"),
        ("qwen3_5_moe_text", "Qwen35MoE"),
    ):
        write(tmp_path / "config.json", dict(text_config=dict(model_type=architecture)))
        assert discover(tmp_path).endswith(":" + expected)
    write(tmp_path / "config.json", dict(model_type="unknown"))
    with pytest.raises(ValueError, match="custom model"):
        discover(tmp_path)


def test_selection_requires_all_shapes_correctness_latency_and_no_large_regression():
    assert score(report(0.5), report(), context()) == pytest.approx(2)
    missing = report()
    missing["results"].pop()
    with pytest.raises(ValueError, match="omitted"):
        score(missing, report(), context())
    wrong = report()
    wrong["results"][0]["correctness"]["correct"] = False
    with pytest.raises(ValueError, match="unqualified"):
        score(wrong, report(), context())
    bad = report(0.5)
    bad["results"][0]["serving"]["seconds"] *= 4
    assert score(bad, report(), context()) is None
    assert score(report(), report(), {**context(), "ttft_ms": 3}) is None


def test_automatic_recipes_preserve_model_precision_and_use_operation_policies():
    p = dict(
        status="ok",
        provider="cuda",
        device=dict(arch="sm_89"),
        chunks=[32, 128, 1],
        policies=[{}, dict(linear=dict(partitions=2))],
        model_options=dict(kv_dtype="bfloat16"),
        windows=[2, 4],
        head="qwen:MTP",
    )
    choices = recipes(context(), [p, dict(status="unsupported", provider="webgpu")])
    assert not choices[0]["graphs"] and choices[0]["trials"] == 1
    assert len({c["id"] for c in choices}) == len(choices)
    assert any(c["policy"] for c in choices)
    assert {c["window"] for c in choices} == {None, 2, 4}
    assert all(c["model_options"] == p["model_options"] for c in choices)
    assert all(c["trials"] == 1 for c in choices)
    selected = next(c for c in choices if c["window"] == 4)
    deeper = refine(selected, 32)
    assert (
        deeper["window"] == 4
        and deeper["trials"] == 32
        and deeper["id"] != selected["id"]
    )
    assert refine(deeper, 8) is None


def test_partial_beam_progress_resumes_and_requalifies_winner(tmp_path):
    variant = Implementation(
        "add",
        "baseline",
        "packed",
        space=(("threads", (64, 128, 256, 512)),),
        factory="kernel",
    )
    options = dict(
        contract="packed",
        target="sm_89",
        catalog=[variant],
        checkpoint=tmp_path / "progress.json",
    )
    seen = []

    def measure(item, p, w):
        seen.append(p.get("threads"))
        return Measurement(
            {None: 10, 64: 8, 128: 6, 256: 4, 512: 2}[p.get("threads")], True
        )

    tune("add", dict(c=256), [Workload("shape")], measure, budget=2, **options)
    assert len(json.loads(options["checkpoint"].read_text())["events"]) == 2
    seen.clear()
    result = tune("add", dict(c=256), [Workload("shape")], measure, budget=4, **options)
    assert seen == [
        None,
        256,
        512,
        512,
    ]  # baseline, new trials, fresh winning qualification
    assert result.schedule == dict(threads=512) and result.score == 0.2
    data = json.loads(options["checkpoint"].read_text())
    data["events"][0]["times"] = [0.0001]
    write(options["checkpoint"], data)
    with pytest.raises(ValueError, match="checksum"):
        tune("add", dict(c=256), [Workload("shape")], measure, budget=4, **options)


def test_resumed_candidate_cannot_reuse_stale_fast_timing(tmp_path):
    v = Implementation(
        "add", "baseline", "packed", space=(("threads", (64, 128)),), factory="kernel"
    )
    options = dict(
        contract="packed",
        target="sm_89",
        catalog=[v],
        checkpoint=tmp_path / "progress.json",
        budget=2,
    )
    measure = lambda item, p, w: Measurement(10 if "threads" not in p else 1, True)
    tune("add", dict(c=256), [Workload("shape")], measure, **options)
    result = tune(
        "add",
        dict(c=256),
        [Workload("shape")],
        lambda item, p, w: Measurement(10, "threads" not in p),
        **options,
    )
    assert result.schedule == {} and result.score == 1


def test_timeout_terminates_worker_process_group_and_retains_log(tmp_path, monkeypatch):
    real = subprocess.Popen
    launched = []

    def popen(command, **options):
        process = real(
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('entered',flush=True); time.sleep(60)",
            ],
            **options,
        )
        launched.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    result = execute(
        dict(action="probe"), tmp_path, deadline=time.monotonic() + 2, seconds=0.2
    )
    assert result["status"] == "budget_exhausted"
    assert launched[0].poll() is not None
    assert "entered" in (tmp_path / "worker.log").read_text()
    assert (
        json.loads((tmp_path / "result.json").read_text())["status"]
        == "budget_exhausted"
    )


def test_optimizer_lock_and_evidence_integrity(tmp_path):
    with lock(tmp_path / "lock"):
        with pytest.raises(RuntimeError, match="another optimizer"):
            with lock(tmp_path / "lock"):
                pass
    result = {}
    for name in ("report", "evidence", "deployment"):
        path = tmp_path / name
        path.write_text(name)
        result[name] = str(path)
        result[name + "_sha256"] = digest(path)
    verify_result(result)
    Path(result["evidence"]).write_text("changed")
    with pytest.raises(ValueError, match="evidence"):
        verify_result(result)


def test_unqualified_deployment_is_not_servable(tmp_path):
    from akas.optimization.deployment import load

    path = tmp_path / "candidate.json"
    write(path, dict(schema="akas.deployment.v1", status="candidate"))
    with pytest.raises(ValueError, match="not passed"):
        load(path)


def test_historical_seeding_matches_hardware_weights_and_verified_qualification(
    tmp_path,
):
    from test_benchmark_history import program, report as history_report
    from akas.benchmarking.history import Recorder
    from akas.graph import Graph
    from akas.kernels.registry import select
    from akas.optimization.incumbents import seed

    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("x=1\n")
    directory = tmp_path / "program"
    profile = program(directory)
    parameters = profile["entries"][0]["parameters"]
    profile["entries"][0]["schedule"]["implementation"] = select(
        "linear",
        parameters,
        contract="bf16",
        phase="decode",
        provider="cuda",
        target="sm_89",
    ).identifier
    phase = json.loads((directory / "phase.json").read_text())
    phase["tuning"]["profile"] = profile
    write(directory / "phase.json", phase)
    evidence = history_report()
    evidence["metadata"]["checkpoint_sha256"] = "weights"
    run = Recorder(tmp_path / "history", evidence["metadata"], source_root=source)
    run.program(directory)
    run.update(evidence, final=True)
    graph = Graph(contract="bf16", phase="decode")
    x = graph.buffer("x", (4, 256), "bfloat16", role="input")
    w = graph.buffer("w", (256, 256), "bfloat16", role="weight")
    out = graph.buffer("out", (4, 256), "float32", role="output")
    graph("linear", parameters, x, w, out)
    options = dict(
        provider="cuda",
        target="sm_89",
        checkpoint_sha256="weights",
        hardware="NVIDIA L4",
    )
    selected, origins = seed(tmp_path / "history", [("decode", graph)], **options)
    assert selected.data == profile and origins[0]["run_id"] == run.id
    assert seed(
        tmp_path / "history", [("decode", graph)], **{**options, "hardware": "other"}
    ) == (None, [])
    path = run.directory / run.record["search"][0]["qualification"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        seed(tmp_path / "history", [("decode", graph)], **options)


def test_campaign_resumes_completed_jobs_and_preserves_deployment_on_changed_evidence(
    tmp_path, monkeypatch
):
    import akas.optimization.supervisor as supervisor
    import akas.optimization.deployment as deployment

    weights = tmp_path / "model.gguf"
    weights.write_bytes(b"checkpoint")
    intent = Context(**{**context(), "checkpoint": str(weights), "backend": "cuda"})
    calls = []

    def execute(job, directory, **options):
        if job["action"] == "probe":
            return dict(
                status="ok",
                provider="cuda",
                device=dict(name="test", arch="sm_89"),
                chunks=[1],
                policies=[{}],
                model_options={},
                windows=[],
                head=None,
            )
        calls.append(job["recipe"]["id"])
        directory.mkdir(parents=True)
        data = report(1)
        write(directory / "report.json", data)
        (directory / "diagnostics.npz").write_bytes(b"diagnostics")
        write(directory / "candidate.json", dict(program="program", artifacts=[]))
        result = dict(
            status="passed",
            run_id=str(len(calls)),
            report=str(directory / "report.json"),
            evidence=str(directory / "diagnostics.npz"),
            deployment=str(directory / "candidate.json"),
        )
        for name in ("report", "evidence", "deployment"):
            result[name + "_sha256"] = digest(result[name])
        return result

    monkeypatch.setattr(supervisor, "execute", execute)
    monkeypatch.setattr(
        deployment, "load", lambda path: json.loads(Path(path).read_text())
    )
    result = supervisor.optimize(
        intent, tmp_path / "out", budget=30, history=tmp_path / "history"
    )
    assert result["status"] == "complete" and result["deployment"]
    count = len(calls)
    before = Path(result["deployment"]).read_bytes()
    resumed = supervisor.optimize(
        intent, tmp_path / "out", budget=30, history=tmp_path / "history"
    )
    assert len(calls) == count and resumed["deployment"] == result["deployment"]
    (tmp_path / "out/jobs" / calls[0] / "attempt-0001/diagnostics.npz").write_bytes(
        b"changed"
    )
    with pytest.raises(ValueError, match="evidence"):
        supervisor.optimize(
            intent, tmp_path / "out", budget=30, history=tmp_path / "history"
        )
    assert Path(result["deployment"]).read_bytes() == before


def test_http_objective_measures_all_concurrent_shapes_and_checks_outputs():
    import asyncio
    from contextlib import contextmanager
    from test_serving import Independent
    from akas.engine import Engine
    from akas.serving.http import Service
    from akas.optimization.evaluation import http_benchmark, prompts

    @contextmanager
    def service():
        model = Independent(19)
        model.capacity = 4
        with Engine(model) as engine:
            yield Service(engine)

    rows = []
    for b in (1, 4):
        for p in (2, 8):
            ids = prompts(19, b, p)
            rows.append(
                dict(
                    case=dict(batch=b, prompt=p, generated=4),
                    output_ids=[[(s[-1] + n + 1) % 19 for n in range(4)] for s in ids],
                    serving=dict(seconds=1),
                )
            )
    result = asyncio.run(http_benchmark(service, rows, vocab=19))
    assert result["status"] == "passed" and result["shapes"] == 4
    assert all(
        len(r["serving"]["samples"]) == 3
        and r["serving"]["ttft_ms"] > 0
        and r["engine"]["seconds"] == 1
        for r in rows
    )


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_real_checkpoint_compiles_qualifies_and_serves_without_manual_settings(
    tmp_path, provider
):
    if os.environ.get("AKAS_OPTIMIZE_" + provider.upper()) != "1":
        pytest.skip("enable the complete optimization GPU integration check")
    from test_model_conventions import checkpoint
    from akas.optimization.supervisor import optimize
    from akas.optimization.deployment import load

    root = tmp_path / "checkpoint"
    root.mkdir()
    checkpoint(root)
    intent = Context(
        str(root),
        "examples.models.bigram.model:Bigram",
        dict(batches=[1, 4], prefix_tokens=[2, 8], output_tokens=4),
        backend=provider,
    )
    result = optimize(
        intent, tmp_path / "optimized", budget=90, history=tmp_path / "history"
    )
    assert result["deployment"]
    manifest = load(result["deployment"])
    assert manifest["status"] == "qualified"
    state = json.loads((tmp_path / "optimized/campaign.json").read_text())
    passed = [j for j in state["jobs"] if j["status"] == "passed"]
    assert len(passed) >= 2
    for job in passed:
        measured = json.loads(Path(job["report"]).read_text())
        assert measured["metadata"]["http"]["status"] == "passed"
        assert measured["metadata"]["http"]["shapes"] == 4
        assert len(measured["results"]) == 4
    resumed = optimize(
        intent, tmp_path / "optimized", budget=30, history=tmp_path / "history"
    )
    assert resumed["deployment"] == result["deployment"]
    assert len(
        json.loads((tmp_path / "optimized/campaign.json").read_text())["jobs"]
    ) == len(state["jobs"])
    data = json.loads((root / "config.json").read_text())
    data["context"] += 1
    write(root / "config.json", data)
    with pytest.raises(ValueError, match="checkpoint"):
        optimize(
            intent, tmp_path / "optimized", budget=30, history=tmp_path / "history"
        )
