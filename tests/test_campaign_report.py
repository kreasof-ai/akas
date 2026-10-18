"""Campaign tables preserve missing shapes, failed baselines and portable evidence."""

import csv
import json
from pathlib import Path
import re
import shutil

import pytest

from akas.benchmarking.history import digest, write
from tools.campaign_report import check, export, read, render, validate


@pytest.fixture
def campaign_evidence(tmp_path):
    history = tmp_path / "benchmarks"
    build = tmp_path / "build"
    model = dict(model="example/toy", modes=["ar"])
    shape = dict(batches=[1, 4], prefix_tokens=[32], output_tokens=128)
    source = dict(revision="abc123", tree_sha256="source", dirty=False)
    spec = dict(
        models=["toy"],
        formats=["bf16"],
        backends=["cuda", "webgpu"],
        caches=["default"],
        workload=shape,
    )
    sample = dict(
        prefill_tokens_per_second=100,
        decode_tokens_per_second=200,
        output_tokens_per_second=150,
        wall_seconds=1,
        phase_protocol="request-phase-v1",
        ttft_seconds=0.1,
        tpot_seconds=0.01,
    )

    def record(run_id, runtime, status, start, end, burst=300):
        rows = []
        for batch in shape["batches"]:
            rows.append(
                dict(
                    case=dict(batch=batch, prompt=32, generated=128),
                    status="ok",
                    samples=[
                        {**sample, "output_tokens_per_second": burst} for _ in range(3)
                    ],
                    engine=dict(samples=[dict(output_tokens_per_second=400)] * 3),
                    serving=dict(
                        samples=[
                            dict(
                                output_tokens_per_second=500, ttft_ms=v, tpot_ms=v / 10
                            )
                            for v in (1, 3, 2)
                        ]
                    ),
                )
            )
        data = dict(
            id=run_id,
            status=status,
            started_at=start,
            completed_at=end,
            source=source,
            metadata=dict(
                runtime=runtime,
                model=model["model"],
                protocol="baseline-burst-v3"
                if runtime != "akas"
                else "native-token-phase-v1",
                purpose="baseline" if runtime != "akas" else "optimization",
                provider="cuda",
                mode="ar",
                kv_dtype="float16",
                weight_format="bf16",
                cpu_peak_rss_bytes=256 * 1024**2,
                error="greedy mismatch" if status == "failed" else None,
            ),
            search=[
                dict(
                    phase="decode",
                    profile=dict(entries=[dict(schedule=dict(block_m=32))]),
                )
            ],
            artifacts=[],
            results=rows,
            dependencies=dict(numpy="test"),
            environment=dict(cpu_count=4),
        )
        path = history / "runs" / run_id / "run.json"
        write(
            path.parent / "report.json", dict(metadata=data["metadata"], results=rows)
        )
        data["artifacts"] = [
            dict(path="report.json", sha256=digest(path.parent / "report.json"))
        ]
        write(path, data)
        return data

    records = [
        record(
            "base-old", "vllm", "passed", "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"
        ),
        record(
            "base-failed",
            "vllm",
            "failed",
            "2026-01-01T00:02:00Z",
            "2026-01-01T00:03:00Z",
            123,
        ),
        record(
            "base-overlap",
            "vllm",
            "passed",
            "2026-01-01T00:04:00Z",
            "2026-01-01T00:07:00Z",
            999,
        ),
        record(
            "winner", "akas", "passed", "2026-01-01T00:06:00Z", "2026-01-01T00:08:00Z"
        ),
    ]
    write(
        history / "index.json",
        dict(
            runs=[
                dict(
                    id=r["id"],
                    started_at=r["started_at"],
                    purpose="baseline",
                    model=model["model"],
                    runtime="vllm",
                    provider="cuda",
                    weight_format="bf16",
                    mode="ar",
                    shapes=[
                        dict(batch=b, prefix_tokens=32, output_tokens=128)
                        for b in shape["batches"]
                    ],
                )
                for r in records[:-1]
            ]
        ),
    )
    recipe = dict(
        id="r1", trials=1, provider="cuda", chunk=32, prefill_requests=4, graphs=True
    )
    record("rejected", "akas", "failed", "2026-01-01T00:08:00Z", "2026-01-01T00:09:00Z")
    entry = dict(
        id="toy-bf16-cuda-default-ar",
        status="budget_exhausted",
        context=dict(workload=shape),
        spent_seconds=60,
        deployment="local-deployment",
        winner=dict(run_id="winner"),
    )
    unsupported = dict(
        id="toy-bf16-webgpu-default-ar",
        status="unsupported",
        context=dict(workload=shape),
        spent_seconds=1,
    )
    state = dict(
        identity="0123456789abcdef",
        created_at="2026-01-01T00:05:00Z",
        updated_at="2026-01-01T00:08:00Z",
        source=source,
        specification=spec,
        models=dict(toy=model),
        status="configuration_budgets_exhausted",
        spent_seconds=61,
        configuration_budget_seconds=60,
        entries=[entry, unsupported],
    )
    campaign = build / "campaign.json"
    write(campaign, state)
    write(
        build / entry["id"] / "campaign.json",
        dict(
            jobs=[
                dict(
                    recipe=recipe, status="passed", run_id="winner", elapsed_seconds=60
                ),
                dict(
                    recipe={**recipe, "id": "r2", "trials": 8},
                    status="failed",
                    run_id="rejected",
                    error="diagnostic changed",
                    elapsed_seconds=8,
                ),
            ],
            probes=[],
        ),
    )
    write(
        build / unsupported["id"] / "campaign.json",
        dict(
            jobs=[],
            probes=[dict(status="unsupported", errors=[dict(error="backend absent")])],
        ),
    )
    return campaign, history, build


def test_all_shapes_and_failed_baseline_values_are_visible(campaign_evidence):
    campaign, history, _ = campaign_evidence
    out = export(campaign, history=history)
    data = read(out / "report.json")
    assert len(data["profiles"]) == 4  # CUDA/WebGPU, vLLM/SGLang.
    assert sum(len(p["measurements"]) for p in data["profiles"]) == 8
    baseline = next(p for p in data["profiles"] if p["runtime"] == "vllm")
    assert (
        baseline["selected_run_id"] == "base-failed"
    )  # Ignore the overlapping future completion.
    assert baseline["qualifying_status"] == "unqualified"
    assert baseline["measurements"][0]["burst_output_tok_s"] == 123
    assert baseline["measurements"][0]["qualification"] == "unqualified"
    assert "123.0 †" in (out / "COMPARISON.md").read_text()
    unsupported = next(p for p in data["profiles"] if p["provider"] == "webgpu")
    assert all(
        r["burst_output_tok_s"] is None and r["qualification"] == "unsupported"
        for r in unsupported["measurements"]
    )
    assert "backend absent" in (out / "AKAS.md").read_text()
    assert "peak host process RSS GiB: 0.2" in (out / "AKAS.md").read_text()
    assert "peak host process RSS GiB: 0.2" in (out / "BASELINES.md").read_text()
    with (out / "measurements.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8 and sum(r["burst_output_tok_s"] == "" for r in rows) == 4
    assert (
        data["schedules"]["toy-bf16-cuda-default-ar"]["phases"][0]["profile"][
            "entries"
        ][0]["schedule"]["block_m"]
        == 32
    )
    winner = next(
        p
        for p in data["profiles"]
        if p["kind"] == "akas" and p["qualifying_status"] == "passed"
    )
    assert (
        winner["measurements"][0]["ttft_ms"] == 2
    )  # Median of measured bursts, not max.
    assert data["evidence"]["winner"]["environment"]["cpu_count"] == 4
    assert winner["measurements"][0]["phase_protocol"] == "native-token-phase-v1"
    with (out / "candidate-measurements.csv").open() as stream:
        candidates = list(csv.DictReader(stream))
    rejected = [r for r in candidates if r["run_id"] == "rejected"]
    assert len(rejected) == 2
    assert all(
        r["candidate_status"] == "failed" and r["qualification"] == "unqualified"
        for r in rejected
    )


def test_report_can_be_rendered_after_build_cleanup(campaign_evidence):
    campaign, history, build = campaign_evidence
    out = export(campaign, history=history)
    expected = {
        p.name: p.read_bytes() for p in out.iterdir() if p.suffix in (".md", ".csv")
    }
    shutil.rmtree(build)
    render(read(out / "report.json"), out, history)
    assert expected == {
        p.name: p.read_bytes() for p in out.iterdir() if p.suffix in (".md", ".csv")
    }
    for p in out.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            path = target.split("#")[0]
            assert not Path(path).is_absolute()
            assert (p.parent / path).exists(), (p, target)
    assert "full tables" in (history / "reports/README.md").read_text()


def test_report_rejects_changed_measurement_evidence(campaign_evidence):
    campaign, history, _ = campaign_evidence
    (history / "runs/winner/report.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        export(campaign, history=history)


def test_archive_check_is_read_only_and_independent_of_build(campaign_evidence):
    campaign, history, build = campaign_evidence
    export(campaign, history=history)
    shutil.rmtree(build)
    before = {p: p.read_bytes() for p in history.rglob("*") if p.is_file()}
    check(history)
    assert before == {p: p.read_bytes() for p in history.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "change, message",
    [
        ("edit_table", "Generated report differs"),
        ("delete_csv", "standard bundle"),
        ("extra_report", "standard bundle"),
        ("extra_directory", "Unexpected report entry"),
        ("extra_documentation", "Performance documentation belongs"),
        ("stale_index", "Report index differs"),
        ("changed_snapshot", "Generated report differs"),
        ("wrong_directory", "campaign identity"),
    ],
)
def test_archive_check_rejects_nonstandard_publication(
    campaign_evidence, change, message
):
    campaign, history, _ = campaign_evidence
    out = export(campaign, history=history)
    if change == "edit_table":
        with (out / "COMPARISON.md").open("a") as stream:
            stream.write("\nUnrecorded performance claim.\n")
    elif change == "delete_csv":
        (out / "measurements.csv").unlink()
    elif change == "extra_report":
        (out / "NOTES.md").write_text("Additional results")
    elif change == "extra_directory":
        (history / "reports/unstructured").mkdir()
    elif change == "extra_documentation":
        (history / "RESULTS-NEW.md").write_text("Additional results")
    elif change == "stale_index":
        (history / "reports/README.md").write_text("Missing campaign")
    elif change == "changed_snapshot":
        write(out / "campaign.json", {})
    elif change == "wrong_directory":
        out.rename(out.with_name("arbitrary"))
    with pytest.raises(ValueError, match=message):
        check(history)


@pytest.mark.parametrize(
    "change, message",
    [
        ("schema", "Unsupported report schema"),
        ("missing_field", "Missing report fields"),
        ("missing_shape", "workload shapes"),
        ("duplicate_shape", "workload shapes"),
        ("missing_metric", "Missing measurement fields"),
        ("invalid_metric", "Invalid measurement value"),
        ("missing_akas", "AKAS matrix"),
        ("missing_baseline", "baseline matrix"),
        ("duplicate_profile", "Duplicate report profile"),
    ],
)
def test_report_contract_rejects_incomplete_data(campaign_evidence, change, message):
    campaign, history, _ = campaign_evidence
    data = read(export(campaign, history=history) / "report.json")
    if change == "schema":
        data["schema"] = "unknown.v2"
    elif change == "missing_field":
        del data["evidence"]
    elif change == "missing_shape":
        data["profiles"][0]["measurements"].pop()
    elif change == "duplicate_shape":
        data["profiles"][0]["measurements"].append(
            data["profiles"][0]["measurements"][0]
        )
    elif change == "missing_metric":
        del data["profiles"][0]["measurements"][0]["prefill_tok_s"]
    elif change == "invalid_metric":
        data["profiles"][0]["measurements"][0]["prefill_tok_s"] = float("nan")
    elif change == "missing_akas":
        data["profiles"] = [p for p in data["profiles"] if p["kind"] != "akas"]
    elif change == "missing_baseline":
        data["profiles"] = [p for p in data["profiles"] if p["runtime"] != "vllm"]
    elif change == "duplicate_profile":
        data["profiles"].append(data["profiles"][0])
    with pytest.raises(ValueError, match=message):
        validate(data)


@pytest.mark.parametrize("interrupted", [False, True])
def test_matrix_exports_report_on_completion_and_interruption(
    campaign_evidence, monkeypatch, interrupted
):
    from tools import campaign as matrix
    from tools.campaign_report import report_directory

    original, history, build = campaign_evidence
    state = read(original)
    spec = state["specification"]
    spec["backends"] = ["cuda"]
    specification = build / "spec.json"
    write(specification, spec)
    monkeypatch.setattr(matrix, "declarations", lambda *args: state["models"])
    monkeypatch.setattr(
        matrix,
        "entries",
        lambda *args: [
            dict(
                id="toy-bf16-cuda-default-ar",
                context=dict(
                    checkpoint="weights",
                    definition="toy:Model",
                    workload=state["entries"][0]["context"]["workload"],
                    backend="cuda",
                ),
            )
        ],
    )

    def optimize(*args, **kwargs):
        if interrupted:
            raise KeyboardInterrupt()
        return dict(status="complete")

    monkeypatch.setattr(matrix, "optimize", optimize)
    out = build / "new-matrix"
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            matrix.run(specification, out, budget=60, slice_seconds=30, history=history)
    else:
        matrix.run(specification, out, budget=60, slice_seconds=30, history=history)
    final = read(out / "campaign.json")
    report = read(report_directory(final, history) / "report.json")
    assert report["campaign"]["status"] == (
        "interrupted" if interrupted else "complete"
    )
    assert len(report["profiles"]) == 3
