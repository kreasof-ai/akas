"""Run provenance, accounting, immutable history and safe incumbent reuse."""

import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from akas.benchmarking.history import Recorder, digest, export_profile, records, write


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "kernel.py").write_text("def kernel(x): return x + 1\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return root


def report(protocol="native-token-step-v2"):
    return dict(
        schema="akas.benchmark.v1",
        metadata=dict(
            device=dict(name="NVIDIA L4", provider="cuda"),
            model="qwen",
            runtime="akas",
            provider="cuda",
            target="sm_89",
            protocol=protocol,
        ),
        results=[
            dict(
                status="ok",
                case=dict(batch=4, prompt=32, generated=128, depth=0, mixed=False),
            )
        ],
    )


def test_dirty_source_snapshot_and_searchable_index_are_durable(tmp_path, source):
    (source / "kernel.py").write_text("def kernel(x): return x + 2\n")
    (source / "new.py").write_text("new = True\n")
    root = tmp_path / "benchmarks"
    run = Recorder(
        root,
        report()["metadata"],
        source_root=source,
        change="fused epilogue",
        parent="previous",
    )
    run.update(report(), final=True)
    record = json.loads((run.directory / "run.json").read_text())
    assert record["source"]["dirty"] is True
    assert record["source"]["revision"]
    assert record["change"] == dict(
        kind="configuration", description="fused epilogue", parent_run="previous"
    )
    assert (
        record["started_at"].endswith("Z")
        and record["completed_at"] >= record["started_at"]
    )
    with tarfile.open(run.directory / "source.tar.gz") as archive:
        assert (
            archive.extractfile("kernel.py").read()
            == (source / "kernel.py").read_bytes()
        )
        assert archive.extractfile("new.py").read() == (source / "new.py").read_bytes()
    for artifact in record["artifacts"]:
        assert digest(run.directory / artifact["path"]) == artifact["sha256"]
    assert [
        r["id"] for r in records(root, hardware="l4", model="qwen", status="passed")
    ] == [run.id]
    assert not list(records(root, hardware="L40S"))
    assert (
        json.loads((root / "index.json").read_text())["runs"][0]["workload"]
        == "standard-v1"
    )
    assert "cuda / unspecified / ar / unspecified" in (root / "RESULTS.md").read_text()
    assert "4/32/128" in (root / "RESULTS.md").read_text()
    with pytest.raises(ValueError, match="immutable"):
        run.update(report())


def test_changed_sources_invalidate_run_and_legacy_accounting_stays_separate(
    tmp_path, source
):
    run = Recorder(tmp_path / "history", report()["metadata"], source_root=source)
    (source / "kernel.py").write_text("changed = True\n")
    record = run.update(report("native-token-step-v1"), final=True)
    assert record["status"] == "invalidated"
    assert record["workload"]["shapes"][0]["output_tokens"] == 129
    assert record["workload"]["id"] == "custom"


def program(root):
    profile = dict(
        schema="tensor.schedule-profile.v1",
        provider="cuda",
        target="sm_89",
        entries=[
            dict(
                operation="linear",
                parameters=dict(r=4, k=256, o=256),
                schedule=dict(block_m=16, implementation="gemm"),
            )
        ],
    )
    write(
        root / "phase.json",
        dict(
            forward=dict(contract="bf16"),
            tuning=dict(
                profile=profile,
                operations=[
                    dict(
                        kind="linear",
                        parameters=dict(r=4, k=256, o=256),
                        selected=dict(
                            implementation="gemm",
                            schedule=dict(block_m=16),
                            score=0.8,
                            trials=4,
                        ),
                    )
                ],
            ),
        ),
    )
    write(
        root / "program.json",
        dict(
            provider="cuda",
            target="sm_89",
            phases=[dict(phase="decode", rows=4, chunk=1, path="phase.json")],
        ),
    )
    return profile


def test_qualified_phase_export_checks_geometry_target_and_evidence(tmp_path, source):
    expected = program(tmp_path / "program")
    root = tmp_path / "history"
    run = Recorder(root, report()["metadata"], source_root=source)
    run.program(tmp_path / "program")
    run.update(report(), final=True)
    options = dict(
        phase="decode",
        batch=4,
        chunk=1,
        provider="cuda",
        target="sm_89",
        out=tmp_path / "seed.json",
    )
    assert export_profile(root, run.id, **options) == expected
    assert json.loads(options["out"].read_text()) == expected
    with pytest.raises(ValueError, match="matching"):
        export_profile(root, run.id, **{**options, "target": "sm_90"})
    path = run.directory / run.record["search"][0]["qualification"]
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        export_profile(root, run.id, **options)


def test_installed_source_without_git_has_no_invented_revision(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    (root / "kernel.py").write_text("x = 1\n")
    run = Recorder(tmp_path / "history", report()["metadata"], source_root=root)
    run.update(report(), final=True)
    assert run.record["source"]["revision"] is None
    assert run.record["source"]["dirty"] is None
    assert "unavailable" in (tmp_path / "history/RESULTS.md").read_text()
