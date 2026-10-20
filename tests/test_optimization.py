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
    assert {
        c["chunk"] for c in choices if c["policy"] == dict(linear=dict(partitions=2))
    } == {32, 128, 1}
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


def test_probe_discovers_policies_from_later_chunk_geometries(monkeypatch):
    from dataclasses import dataclass
    from types import SimpleNamespace
    import akas.optimization.planning as planning
    import akas.compilation.policy as policy
    from akas.graph import Graph

    @dataclass
    class Config:
        width: int = 64

    def compose(checkpoint, **options):
        graph = Graph(contract="reference")
        graph.chunk = options["chunk"]
        return graph

    model = SimpleNamespace(
        checkpoint_type=lambda path: SimpleNamespace(config=Config()),
        compose=compose,
        optimization_chunks=(32, 4096),
        optimization_space=(),
    )
    monkeypatch.setattr(planning, "symbol", lambda name: model)
    monkeypatch.setattr(planning, "hardware", lambda *args: dict(arch="sm_89"))
    monkeypatch.setattr(planning, "software", lambda: {})
    monkeypatch.setattr(
        policy,
        "candidates",
        lambda graph, **kwargs: (
            [{}, dict(attention=dict(decoded_kv=True))] if graph.chunk == 4096 else [{}]
        ),
    )
    settings = {
        **context(),
        "mode": "ar",
        "workload": {**context()["workload"], "prefix_tokens": [32, 8192]},
    }
    result = planning.probe(settings, "cuda")
    assert result["status"] == "ok"
    assert dict(attention=dict(decoded_kv=True)) in result["policies"]


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


@pytest.mark.parametrize("phase", [None, "validating-and-measuring", "http-validation"])
def test_timeout_terminates_worker_process_group_and_retains_log(
    tmp_path, monkeypatch, phase
):
    if phase is not None:
        write(tmp_path / "progress.json", dict(phase=phase))
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


@pytest.mark.parametrize("extra_family", [False, True])
def test_search_starts_from_saved_geometry(monkeypatch, extra_family):
    monkeypatch.delenv("AKAS_SKIP_AUTOTUNE", raising=False)
    choice = Implementation(
        "add",
        "linear.fake",
        "reference",
        factory="kernel",
        defaults=(("tile", 16),),
        space=(("tile", (16, 32, 64)),),
        variant="saved",
    )
    measured = []

    def measure(implementation, p, workload):
        tile = p.get("tile", 16)
        measured.append((implementation.identifier, tile))
        return Measurement({16: 4.0, 32: 2.0, 64: 1.0}[tile], True)

    result = tune(
        "add",
        dict(c=256),
        [Workload("shape")],
        measure,
        contract="reference",
        provider="cuda",
        target="sm_89",
        catalog=(replace(choice, variant="fallback"), choice)
        if extra_family
        else (choice,),
        baseline=dict(implementation=choice.identifier, tile=64),
        budget=1,
        width=1,
    )
    assert measured == [(choice.identifier, 64), (choice.identifier, 64)]
    assert result.schedule["tile"] == 64


def test_long_context_search_refines_before_costly_small_prefills():
    from akas.optimization.planning import search_order

    plans = [
        dict(id=str(chunk), provider="cuda", chunk=chunk)
        for chunk in (256, 512, 1024, 128, 32, 1)
    ]
    ordered = search_order(plans, dict(prefix_tokens=[32, 130944]))
    markers = [i for i, p in enumerate(ordered) if "refine" in p]
    assert [ordered[i]["refine"] for i in markers] == [8, 32, 128]
    assert max(i for i, p in enumerate(ordered) if p.get("chunk", 0) >= 256) < min(
        markers
    )
    assert max(markers) < min(
        i for i, p in enumerate(ordered) if p.get("chunk", 256) < 256
    )
    assert {p["id"] for p in ordered if "id" in p} == {p["id"] for p in plans}
    regular = search_order(plans, dict(prefix_tokens=[32, 2048]))
    assert regular[: len(plans)] == plans


def test_native_session_timeout_checks_prefill_chunks_and_releases_slot(monkeypatch):
    from types import SimpleNamespace
    from akas.benchmarking import session as module

    clock, released = [0.0], []
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])

    def prefill(slots, prompts, *, chunk, check):
        for _ in range(4):
            clock[0] += 0.6
            check()
        raise AssertionError("deadline did not stop prefill")

    executor = SimpleNamespace(
        capacity=1,
        context=16,
        config=SimpleNamespace(vocab=8),
        device=SimpleNamespace(synchronize=lambda: None),
        program={},
        admit=lambda: 0,
        release=released.append,
        prefill=prefill,
    )
    with pytest.raises(TimeoutError, match="native shape B1"):
        with module.Session(executor, batch=1, chunk=2, timeout=1) as session:
            session.reset()
            session.prefill([[1] * 8])
    assert released == [0] and clock[0] == pytest.approx(1.2)


def test_native_session_timeout_bounds_decode_without_host_token_readbacks(monkeypatch):
    from types import SimpleNamespace
    from akas.benchmarking import session as module

    clock, calls = [0.0], []
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])

    def decode(slots, tokens, steps, *, read):
        calls.append((tokens, steps, read))
        clock[0] += steps * 0.1

    executor = SimpleNamespace(
        capacity=1,
        context=128,
        config=SimpleNamespace(vocab=8),
        device=SimpleNamespace(synchronize=lambda: None),
        program={},
        admit=lambda: 0,
        release=lambda _: None,
        decode_steps=decode,
    )
    with module.Session(executor, batch=1, chunk=2, timeout=1) as session:
        session.reset()
        session.tokens = [3]
        with pytest.raises(TimeoutError):
            session.decode(127, read=False)
    assert calls == [([3], 8, False), (None, 8, False)]


def test_http_shape_timeout_rejects_slow_candidate(monkeypatch):
    import asyncio
    from contextlib import contextmanager
    from test_serving import Independent
    from akas.engine import Engine
    from akas.serving.http import Service
    from akas.optimization.evaluation import http_benchmark

    class Slow(Independent):
        def prefill_batch(self, requests, sequences, chunk):
            time.sleep(0.05)
            return super().prefill_batch(requests, sequences, chunk)

    @contextmanager
    def service():
        model = Slow(19)
        model.capacity = 1
        with Engine(model) as engine:
            yield Service(engine)

    monkeypatch.setenv("AKAS_SHAPE_TIMEOUT", "0.01")
    monkeypatch.setenv("AKAS_BENCHMARK_WARMUP", "0")
    rows = [
        dict(
            case=dict(batch=1, prompt=2, generated=4),
            output_ids=[[3, 4, 5, 6]],
            serving=dict(seconds=1),
        )
    ]
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(http_benchmark(service, rows, vocab=19))


def test_shape_timeout_keeps_independent_native_phase_measurements(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    import numpy as np
    from akas.optimization import evaluation

    settings, measured = [], []

    class Session:
        allocated_bytes = 1

        def __init__(self, executor, **kwargs):
            settings.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *error):
            pass

    def native(session, case, **kwargs):
        measured.append(case.prompt)
        return dict(
            case=asdict(case), status="ok", summary={"native": True}, samples=[]
        )

    monkeypatch.setenv("AKAS_SHAPE_TIMEOUT", "2")
    monkeypatch.setattr(evaluation, "Session", Session)
    monkeypatch.setattr(evaluation, "run", native)
    monkeypatch.setattr(evaluation, "diagnostic", lambda *a, **k: np.ones((3, 1, 4)))
    monkeypatch.setattr(
        evaluation,
        "Engine",
        lambda *a, **k: SimpleNamespace(
            scheduler=SimpleNamespace(stats={}), close=lambda: None
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "measured_requests",
        lambda *a, **k: dict(
            seconds=1,
            output_tokens_per_second=4,
            ttft_ms=100,
            tpot_ms=300,
            output_ids=[[1, 2, 3, 4]],
        ),
    )
    recipe = dict(chunk=32, window=None, prefill_requests=1)
    rows, _, _ = evaluation.evaluate(
        SimpleNamespace(),
        recipe,
        dict(batches=[1], prefix_tokens=[32, 64], output_tokens=4),
    )
    assert measured == [32, 64]
    assert all(row["summary"] == {"native": True} for row in rows)
    assert [s["timeout"] for s in settings if "timeout" in s] == [2, 2]


@pytest.mark.parametrize("failure", ["greedy", "determinism"])
def test_rejected_generation_retains_native_and_serving_evidence(monkeypatch, failure):
    from dataclasses import asdict
    from types import SimpleNamespace
    import numpy as np
    from akas.optimization import evaluation

    closed, calls = [], {}

    class Session:
        allocated_bytes = 1

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def measured(engine, batch, prefix, output, **kwargs):
        calls[prefix] = calls.get(prefix, 0) + 1
        token = 5 if prefix == 64 and (failure == "greedy" or calls[prefix] % 2) else 4
        return dict(
            seconds=1,
            output_tokens_per_second=4,
            ttft_ms=100,
            tpot_ms=300,
            output_ids=[[1, 2, 3, token]],
        )

    monkeypatch.delenv("AKAS_SKIP_AUTOTUNE", raising=False)
    monkeypatch.setenv("AKAS_BENCHMARK_WARMUP", "0")
    monkeypatch.setenv("AKAS_BENCHMARK_REPEATS", "2")
    monkeypatch.setattr(evaluation, "Session", Session)
    monkeypatch.setattr(
        evaluation,
        "run",
        lambda session, case, **kw: dict(
            case=asdict(case), status="ok", summary={"native": case.prompt}, samples=[]
        ),
    )
    monkeypatch.setattr(evaluation, "diagnostic", lambda *a, **k: np.ones((3, 1, 4)))
    monkeypatch.setattr(
        evaluation,
        "Engine",
        lambda *a, **k: SimpleNamespace(
            scheduler=SimpleNamespace(stats={}), close=lambda: closed.append(True)
        ),
    )
    monkeypatch.setattr(evaluation, "measured_requests", measured)
    reference = dict(
        metadata=dict(run_id="reference"),
        results=[
            dict(case=dict(batch=1, prompt=p), status="ok", output_ids=[[1, 2, 3, 4]])
            for p in (32, 64, 96)
        ],
    )
    with pytest.raises(ValueError) as caught:
        evaluation.evaluate(
            SimpleNamespace(),
            dict(chunk=32, window=None, prefill_requests=1),
            dict(batches=[1], prefix_tokens=[32, 64, 96], output_tokens=4),
            reference=reference,
        )
    rows = caught.value.results
    assert [r["summary"] for r in rows] == [{"native": p} for p in (32, 64, 96)]
    assert rows[0]["correctness"]["correct"]
    assert rows[1]["status"] == "error" and not rows[1]["correctness"]["correct"]
    assert rows[1]["correctness"]["expected_output_ids"] == [[1, 2, 3, 4]]
    outputs = [s["output_ids"] for s in rows[1]["serving"]["samples"]]
    assert len(outputs) == 2 and outputs[0] == [[1, 2, 3, 5]]
    if failure == "determinism":
        assert outputs[1] != outputs[0]
    assert rows[2]["error"] == "model validation incomplete"
    assert set(caught.value.diagnostics) == {"b1-p32", "b1-p64", "b1-p96"}
    assert closed == [True] and 96 not in calls
