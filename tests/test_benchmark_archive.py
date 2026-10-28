"""Archive rotation, immutable navigation and recovery without touching HEAD/index."""

from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from akas.benchmarking import archive
from akas.benchmarking.history import (
    command,
    digest,
    load_record,
    rebuild_index,
    records,
    write,
)
from tools.benchmark_archive import check, file_manifest, prune, publish


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    archive.git(tmp_path, "config", "user.name", "Test")
    archive.git(tmp_path, "config", "user.email", "test@example.invalid")
    (tmp_path / "code.py").write_text("original\n")
    archive.git(tmp_path, "add", "code.py")
    archive.git(tmp_path, "commit", "-qm", "source")
    (tmp_path / "code.py").write_text("staged\n")
    archive.git(tmp_path, "add", "code.py")
    (tmp_path / "code.py").write_text("unstaged\n")
    return tmp_path


def run(root, identity):
    folder = root / "runs" / identity
    write(folder / "report.json", dict(results=[]))
    write(
        folder / "run.json",
        dict(
            schema="akas.benchmark-run.v1",
            id=identity,
            started_at="2026-10-25T00:00:00Z",
            completed_at="2026-10-25T00:01:00Z",
            status="passed",
            source=dict(revision="abc123", dirty=False),
            metadata=dict(
                device=dict(name="GPU"), runtime="akas", model="toy", provider="cuda"
            ),
            workload=dict(id="custom", shapes=[]),
            search=[],
            results=[],
            change={},
            artifacts=[dict(path="report.json", sha256=digest(folder / "report.json"))],
        ),
    )
    return folder


def test_rotation_preserves_index_head_source_bytes_and_retrieval(repo, capsys):
    root = repo / "benchmarks"
    folders = [run(root, f"run-{i}") for i in range(4)]
    size = max(sum(v["bytes"] for v in file_manifest(f).values()) for f in folders)
    head = archive.git(repo, "rev-parse", "HEAD")
    staged = archive.git(repo, "diff", "--cached")
    publish(
        root,
        repository_url="https://github.com/example/archive",
        max_commit_bytes=size,
        max_branch_bytes=size * 2,
    )
    data = check(root, objects=True)
    entries = list(data["runs"].values())
    assert len({e["branch"] for e in entries}) == 2
    assert len({e["commit"] for e in entries}) == 4
    for entry in entries:
        # Every commit has exactly its own bundle; later trees drop old payloads.
        names = archive.git(
            repo, "ls-tree", "-r", "--name-only", entry["commit"]
        ).decode()
        assert all(n.startswith(entry["path"] + "/") for n in names.splitlines())
    for branch in {e["branch"] for e in entries}:
        commits = archive.git(repo, "rev-list", branch).decode().splitlines()
        assert len(commits) == 2
        assert (
            len(archive.git(repo, "rev-list", "--max-parents=0", branch).splitlines())
            == 1
        )
        assert head.strip().decode() not in commits
    assert archive.git(repo, "rev-parse", "HEAD") == head
    assert archive.git(repo, "diff", "--cached") == staged
    assert (repo / "code.py").read_text() == "unstaged\n"
    prune(root)
    assert not list((root / "runs").glob("*/run.json"))
    assert load_record(root, "run-0")["status"] == "passed"
    assert len(list(records(root, hardware="gpu"))) == 4
    assert not list(records(root, provider="webgpu"))
    rebuild_index(root)
    summaries = json.loads((root / "index.json").read_bytes())["runs"]
    assert len(summaries) == 4 and all("/blob/" in r["path"] for r in summaries)
    cache = archive.directory(root, "runs", "run-0")
    assert cache.is_relative_to(repo / "build")
    assert (cache / "run.json").is_file()
    run(root, "new-run")
    rebuild_index(root)
    assert len(json.loads((root / "index.json").read_bytes())["runs"]) == 5
    # Listing summaries still works without archived Git objects being available.
    args = Namespace(
        root=root,
        action="list",
        json=False,
        **dict.fromkeys(
            (
                "hardware",
                "model",
                "runtime",
                "status",
                "workload",
                "provider",
                "mode",
                "purpose",
            )
        ),
    )
    command(args)
    assert "run-0 passed akas toy" in capsys.readouterr().out
    publish(root)
    assert len(archive.catalog(root)["runs"]) == 5


def test_prune_refuses_changed_or_unsealed_data(repo):
    root = repo / "benchmarks"
    folder = run(root, "sealed")
    publish(root, repository_url="https://github.com/example/archive")
    (folder / "report.json").write_text("changed")
    with pytest.raises(ValueError, match="refusing removal"):
        prune(root)
    with pytest.raises(ValueError, match="checksum"):
        archive.read_file(root, "runs", "sealed", "report.json")
    incomplete = run(root, "unfinished") / "run.json"
    data = json.loads(incomplete.read_bytes())
    data["completed_at"] = None
    write(incomplete, data)
    with pytest.raises(ValueError, match="seal or mark interrupted"):
        publish(root)


def test_fork_publication_keeps_historical_links(repo):
    root = repo / "benchmarks"
    run(root, "upstream")
    publish(root, repository_url="https://github.com/example/upstream")
    original = deepcopy(archive.catalog(root)["runs"]["upstream"])
    run(root, "fork")
    publish(root, repository_url="https://github.com/example/fork")
    data = archive.catalog(root)
    assert data["repository"] == "https://github.com/example/upstream"
    assert data["runs"]["upstream"] == original
    assert archive.url(data, original, "run.json").startswith(
        "https://github.com/example/upstream/"
    )
    current = data["runs"]["fork"]
    assert archive.url(data, current, "run.json").startswith(
        "https://github.com/example/fork/"
    )
    assert archive.url(data, current, "plot.svg", raw=True).startswith(
        "https://raw.githubusercontent.com/example/fork/"
    )
    check(root, objects=True)


def test_new_shard_starts_without_source_or_previous_archive_parent(repo):
    root = repo / "benchmarks"
    run(root, "first")
    publish(root, repository_url="https://github.com/example/archive")
    original = deepcopy(archive.catalog(root)["runs"]["first"])
    run(root, "fresh")
    publish(root, first_shard=9)
    data = archive.catalog(root)
    assert data["runs"]["first"] == original
    entry = data["runs"]["fresh"]
    assert entry["branch"] == "codex/artifacts/2026-10-009"
    assert archive.git(repo, "rev-list", "--count", entry["commit"]).strip() == b"1"
    check(root, objects=True)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "C:/escape", "a\\b", "a/../b"]
)
def test_paths_cannot_escape_archive(name):
    with pytest.raises(ValueError, match="unsafe archive path"):
        archive.safe_path(name)


def test_source_ci_rejects_tracked_payloads(repo):
    root = repo / "benchmarks"
    run(root, "run")
    archive.git(repo, "add", "benchmarks/runs")
    with pytest.raises(ValueError, match="must not be tracked"):
        check(root, tracked=True)


def test_catalog_rejects_corruption_and_report_links_are_pinned(repo):
    root = repo / "benchmarks"
    run(root, "run")
    publish(root, repository_url="https://github.com/example/archive")
    data = deepcopy(archive.catalog(root))
    source = root / "reports/report/README.md"
    rewritten = archive.rewrite_links(
        "[run](../../runs/run/run.json) ![plot](../../runs/run/report.json)",
        source,
        root,
    )
    commit = data["runs"]["run"]["commit"]
    assert f"/blob/{commit}/benchmarks/runs/run/run.json" in rewritten
    assert "raw.githubusercontent.com/example/archive/" + commit in rewritten
    data["runs"]["run"]["files"]["run.json"]["sha256"] = "0" * 64
    write(root / "archive.json", data)
    with pytest.raises(ValueError, match="checksum mismatch"):
        check(root, objects=True)


def test_single_branch_clone_lists_offline_and_fetches_only_requested_archive(
    repo, tmp_path, capsys
):
    root = repo / "benchmarks"
    run(root, "run")
    publish(root, repository_url="https://github.com/example/archive")
    prune(root)
    clone = repo.parent / (repo.name + "-clone")
    branch = archive.git(repo, "branch", "--show-current").decode().strip()
    archive.git(
        repo,
        "clone",
        "--quiet",
        "--no-local",
        "--single-branch",
        "--branch",
        branch,
        str(repo),
        str(clone),
    )
    import shutil

    shutil.copytree(root, clone / "benchmarks")
    with pytest.raises(ValueError, match="is unavailable; fetch"):
        load_record(clone / "benchmarks", "run")
    assert not list(records(clone / "benchmarks", available_only=True))
    directory = archive.directory(clone / "benchmarks", "runs", "run", fetch=True)
    assert json.loads((directory / "run.json").read_bytes())["id"] == "run"
    assert archive.git(
        clone, "rev-parse", "refs/remotes/origin/codex/artifacts/2026-10-001"
    )
    prune(clone / "benchmarks")
