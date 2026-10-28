"""Publish bounded benchmark snapshots, verify them, and retrieve one bundle."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import quote

from akas.benchmarking import archive
from akas.benchmarking.history import digest, index_entries, rebuild_index, write


def file_manifest(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"archive cannot contain symlinks: {path}")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            archive.safe_path(name)
            result[name] = dict(
                sha256=digest(path), bytes=path.stat().st_size, mode="100644"
            )
    return result


def report_summary(data):
    profiles = data["profiles"]
    counts = {}
    for kind in ("akas", "baseline"):
        rows = [p for p in profiles if p["kind"] == kind]
        counts[kind] = (
            f"{sum(p['qualifying_status'] == 'passed' for p in rows)}/{len(rows)}"
        )
    state = data["campaign"]
    return dict(
        started_at=state["created_at"],
        workload=data["workload"].get("id", "custom"),
        revision=state["source"].get("revision"),
        **counts,
    )


def overview(data):
    rows = {}
    for kind in ("runs", "reports"):
        for entry in data[kind].values():
            row = rows.setdefault(
                (entry["branch"], entry["commit"]), dict(runs=0, reports=0, bytes=0)
            )
            row[kind] += 1
            row["bytes"] += sum(f["bytes"] for f in entry["files"].values())
    lines = [
        "# Artifact archive",
        "",
        "Generated from [archive.json](archive.json). Find individual bundles in the",
        "[run index](RESULTS.md) and [report index](reports/README.md). Each row is one",
        "snapshot; a branch tip contains only its latest batch. Earlier snapshots remain",
        "reachable through the full commit hashes below.",
        "",
        "| Branch | Commit | Runs | Reports | MiB |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for (branch, commit), row in sorted(rows.items()):
        repositories = {
            entry.get("repository", data["repository"])
            for kind in ("runs", "reports")
            for entry in data[kind].values()
            if entry["branch"] == branch and entry["commit"] == commit
        }
        repo = next(iter(repositories))
        lines.append(
            f"| [{branch}]({repo}/tree/{quote(branch, safe='')}) | "
            f"[{commit}]({repo}/commit/{commit}) | {row['runs']} | {row['reports']} | "
            f"{row['bytes'] / 1024**2:.2f} |"
        )
    return "\n".join(lines) + "\n"


def write_overview(root, data):
    (Path(root) / "ARCHIVE.md").write_text(
        overview(data), encoding="utf-8", newline="\n"
    )


def _report_snapshot(source, target, root):
    shutil.copytree(source, target)
    for path in list(target.rglob("*.md")):
        original = path.read_bytes()
        content = archive.rewrite_links(
            original.decode("utf-8"), source / path.relative_to(target), root
        )
        if content.encode("utf-8") != original:
            saved = target / "originals" / path.relative_to(target)
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(original)
            path.write_text(content, encoding="utf-8", newline="\n")
    schedules = target / "SCHEDULES.json"
    if schedules.exists():
        original = schedules.read_bytes()
        data = json.loads(original)
        for entry in data.get("configurations", {}).values():
            if entry.get("run_id"):
                entry["record"] = archive.link(
                    root, "runs", entry["run_id"], "run.json"
                ) or entry.get("record")
        rendered = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
        if rendered != original:
            (target / "originals").mkdir(exist_ok=True)
            (target / "originals/SCHEDULES.json").write_bytes(original)
            schedules.write_bytes(rendered)


def publish(
    root="benchmarks",
    *,
    repository_url=None,
    branch_prefix="codex/artifacts",
    max_commit_bytes=192 * 1024**2,
    max_branch_bytes=512 * 1024**2,
    legacy_bootstrap=False,
    first_shard=1,
):
    """Use a temporary Git index: never switch HEAD or stage the user's work."""
    root = Path(root).resolve()
    repo = archive.repository(root)
    if root != repo / "benchmarks":
        raise ValueError("publish requires the repository's benchmarks directory")
    archive.safe_path(branch_prefix)
    archive.git(repo, "check-ref-format", "refs/heads/" + branch_prefix + "/check")
    if max_commit_bytes <= 0 or max_branch_bytes < max_commit_bytes:
        raise ValueError("branch limit must be at least the positive commit limit")
    if type(first_shard) is not int or first_shard < 1:
        raise ValueError("first shard must be a positive integer")
    data = deepcopy(archive.catalog(root))
    if legacy_bootstrap and (data["runs"] or data["reports"]):
        raise ValueError("legacy bootstrap is only allowed for an empty archive")
    publication_repository = (
        repository_url.rstrip("/") if repository_url else data["repository"]
    )
    if not data["repository"]:
        if not publication_repository:
            remote = archive.git(repo, "remote", "get-url", "origin").decode().strip()
            remote = re.sub(r"^git@github.com:", "https://github.com/", remote)
            publication_repository = remote.removesuffix(".git")
        data["repository"] = publication_repository
    if not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", publication_repository):
        raise ValueError("specify --repository https://github.com/OWNER/REPO")
    rebuild_index(root)
    summaries = {
        r["id"]: r for r in json.loads((root / "index.json").read_bytes())["runs"]
    }
    pending = []
    for kind, required in (("runs", "run.json"), ("reports", "report.json")):
        for folder in sorted((root / kind).glob("*")):
            if not folder.is_dir() or folder.name == ".ipynb_checkpoints":
                continue
            if folder.name in data[kind]:
                continue
            if not (folder / required).is_file():
                raise ValueError(f"incomplete artifact bundle: {folder}")
            record = json.loads((folder / required).read_bytes())
            if kind == "runs":
                if record.get("status") == "running" or not record.get("completed_at"):
                    raise ValueError(
                        f"seal or mark interrupted before archiving: {folder}"
                    )
                if record["id"] != folder.name:
                    raise ValueError("run ID differs from its directory")
                summary = summaries[folder.name]
            else:
                summary = report_summary(record)
            size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            if size > max_commit_bytes:
                raise ValueError(f"bundle exceeds commit size limit: {folder}")
            pending.append((kind, folder, summary, size))
    if any(kind == "reports" for kind, *_ in pending) and not legacy_bootstrap:
        # Publication must not let new reports bypass the generated-content gate
        # by immediately turning them into checksum-sealed historical snapshots.
        from tools.campaign_report import check as check_reports

        check_reports(root)
    # Runs precede reports so every rewritten report link has an immutable target.
    write(root / "archive.json", data, compact_entries=True)
    branches = {}
    for kind in ("runs", "reports"):
        for entry in data[kind].values():
            state = branches.setdefault(entry["branch"], dict(bytes=0, commits=set()))
            state["bytes"] += sum(f["bytes"] for f in entry["files"].values())
            state["commits"].add(entry["commit"])
    with tempfile.TemporaryDirectory(prefix="akas-archive-") as temporary:
        temporary = Path(temporary)
        env = dict(os.environ, GIT_INDEX_FILE=str(temporary / "index"))
        mode_output = archive.git(repo, "ls-files", "--stage", "-z")
        modes = {
            row.split(b"\t", 1)[1].decode(): row[:6].decode()
            for row in mode_output.split(b"\0")
            if row
        }
        batch = []
        batch_size = 0
        branch = None

        def commit_batch():
            nonlocal batch, batch_size
            if not batch:
                return
            ref = "refs/heads/" + branch
            state = branches.setdefault(branch, dict(bytes=0, commits=set()))
            try:
                parent = (
                    archive.git(repo, "rev-parse", "--verify", ref).decode().strip()
                )
            except ValueError:
                parent = None
            if parent and parent not in state["commits"]:
                raise ValueError(f"archive branch has an unrecognized tip: {branch}")
            archive.git(repo, "read-tree", "--empty", env=env)
            entries = []
            for kind, folder, summary, _ in batch:
                source_files = file_manifest(folder)
                snapshot = folder
                if kind == "reports":
                    snapshot = temporary / folder.name
                    _report_snapshot(folder, snapshot, root)
                files = file_manifest(snapshot)
                for name, info in files.items():
                    info["mode"] = modes.get(
                        f"benchmarks/{kind}/{folder.name}/{name}", "100644"
                    )
                entries.append((kind, folder, summary, snapshot, files, source_files))
            files_to_hash = [
                (snapshot / name, f"benchmarks/{kind}/{folder.name}/{name}", info)
                for kind, folder, _, snapshot, files, _ in entries
                for name, info in files.items()
            ]
            actual_size = sum(info["bytes"] for _, _, info in files_to_hash)
            if (
                actual_size > max_commit_bytes
                or state["bytes"] + actual_size > max_branch_bytes
            ):
                raise ValueError(
                    "snapshot exceeds size bounds (allow room for report originals)"
                )
            for path, _, _ in files_to_hash:
                if "\n" in str(path) or "\r" in str(path):
                    raise ValueError("newline in archive file name")
            blobs = archive.git(
                repo,
                "hash-object",
                "-w",
                "--no-filters",
                "--stdin-paths",
                input="".join(str(p) + "\n" for p, _, _ in files_to_hash).encode(),
            ).splitlines()
            index = b"".join(
                f"{info['mode']} {blob.decode()}\t{name}".encode() + b"\0"
                for (_, name, info), blob in zip(files_to_hash, blobs, strict=True)
            )
            archive.git(
                repo, "update-index", "-z", "--index-info", input=index, env=env
            )
            tree = archive.git(repo, "write-tree", env=env).decode().strip()
            for _, folder, _, _, _, source_files in entries:
                if file_manifest(folder) != source_files:
                    raise ValueError(f"artifact changed while archiving: {folder}")
            args = ["commit-tree", tree]
            if parent:
                args.extend(["-p", parent])
            message = f"Archive {len(batch)} benchmark bundles ({actual_size / 1024**2:.1f} MiB); replace previous batch\n"
            commit = archive.git(repo, *args, input=message.encode()).decode().strip()
            archive.git(repo, "update-ref", ref, commit, parent or "0" * 40)
            for kind, folder, summary, snapshot, files, source_files in entries:
                data[kind][folder.name] = dict(
                    branch=branch,
                    commit=commit,
                    path=f"benchmarks/{kind}/{folder.name}",
                    files=files,
                    summary=summary,
                )
                if publication_repository != data["repository"]:
                    data[kind][folder.name]["repository"] = publication_repository
                if kind == "reports":
                    data[kind][folder.name]["source_files"] = source_files
            write(root / "archive.json", data, compact_entries=True)
            state["bytes"] += actual_size
            state["commits"].add(commit)
            print(
                f"{branch} {commit} {len(batch)} bundles {actual_size / 1024**2:.1f} MiB",
                flush=True,
            )
            batch, batch_size = [], 0

        for item in pending:
            kind, folder, summary, size = item
            if batch and batch[-1][0] != kind:
                commit_batch()
            # Reserve a little room for rewritten report pages and their originals.
            budget = size * 2 if kind == "reports" else size
            month = summary["started_at"][:7]
            number = first_shard
            while True:
                candidate = f"{branch_prefix}/{month}-{number:03d}"
                used = branches.get(candidate, {}).get("bytes", 0)
                waiting = batch_size if branch == candidate else 0
                if used + waiting + budget <= max_branch_bytes:
                    break
                number += 1
            if branch != candidate or batch_size + budget > max_commit_bytes:
                commit_batch()
            branch = candidate
            batch.append(item)
            batch_size += budget
        commit_batch()
    rebuild_index(root)
    write_overview(root, data)
    from tools.campaign_report import index

    index(root)
    return data


def check(root="benchmarks", *, objects=False, fetch=False, tracked=False):
    root = Path(root).resolve()
    data = archive.catalog(root)
    if tracked:
        repo = archive.repository(root)
        paths = archive.git(
            repo, "ls-files", "-z", "--", "benchmarks/runs", "benchmarks/reports"
        )
        payloads = [
            p for p in paths.split(b"\0") if p and p != b"benchmarks/reports/README.md"
        ]
        if payloads:
            raise ValueError(
                "benchmark payloads must not be tracked on the source branch"
            )
    if (root / "archive.json").exists():
        index = json.loads((root / "index.json").read_bytes())
        if index != dict(schema="akas.benchmark-index.v1", runs=index_entries(root)):
            raise ValueError(
                "benchmark index differs from catalog and local runs; rebuild it"
            )
        if (root / "ARCHIVE.md").read_text(encoding="utf-8") != overview(data):
            raise ValueError(
                "archive overview differs from catalog; run publish to regenerate it"
            )
    repo = archive.repository(root) if objects else None
    trees = {}
    for kind in ("runs", "reports"):
        for identity, entry in data[kind].items():
            if not objects:
                continue
            commit = entry["commit"]
            if commit not in trees:
                archive.ensure_commit(root, entry, fetch=fetch)
                output = archive.git(repo, "ls-tree", "-r", "-z", commit)
                trees[commit] = {
                    row.split(b"\t", 1)[1].decode(): row.split(b"\t", 1)[0].split()
                    for row in output.split(b"\0")
                    if row
                }
            paths = {
                name.removeprefix(entry["path"] + "/")
                for name in trees[commit]
                if name.startswith(entry["path"] + "/")
            }
            if paths != entry["files"].keys():
                raise ValueError(f"archive tree differs from manifest: {identity}")
            # One cat-file process per bundle, with streaming payloads, avoids
            # thousands of Git processes and never holds a whole archive in RAM.
            names = list(entry["files"])
            import subprocess
            import threading

            process = subprocess.Popen(
                ["git", "-C", str(repo), "cat-file", "--batch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )

            def request():
                try:
                    for name in names:
                        process.stdin.write(
                            f"{commit}:{entry['path']}/{name}\n".encode()
                        )
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

            sender = threading.Thread(target=request)
            sender.start()
            try:
                for name in names:
                    header = process.stdout.readline().split()
                    if len(header) != 3 or header[1] != b"blob":
                        raise ValueError(f"missing archive blob: {identity}/{name}")
                    remaining = int(header[2])
                    info = entry["files"][name]
                    value = hashlib.sha256()
                    if remaining != info["bytes"]:
                        raise ValueError("archive byte count mismatch")
                    while remaining:
                        block = process.stdout.read(min(1024**2, remaining))
                        if not block:
                            raise ValueError("truncated archive blob")
                        value.update(block)
                        remaining -= len(block)
                    if (
                        process.stdout.read(1) != b"\n"
                        or value.hexdigest() != info["sha256"]
                    ):
                        raise ValueError(
                            f"archive checksum mismatch: {identity}/{name}"
                        )
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.terminate()
                process.wait()
                sender.join()
    return data


def prune(root="benchmarks"):
    """Remove only byte-verified local copies already sealed in reachable Git refs."""
    root = Path(root).resolve()
    data = check(root, objects=True)
    repo = archive.repository(root)
    checked = set()
    for kind in ("runs", "reports"):
        for entry in data[kind].values():
            key = entry["branch"], entry["commit"]
            if key in checked:
                continue
            for ref in (
                "refs/heads/" + entry["branch"],
                "refs/remotes/origin/" + entry["branch"],
            ):
                try:
                    archive.git(
                        repo, "merge-base", "--is-ancestor", entry["commit"], ref
                    )
                    checked.add(key)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(
                    f"archive branch must retain commit before pruning: {entry['branch']}"
                )
    for kind in ("runs", "reports"):
        for identity, entry in data[kind].items():
            folder = root / kind / identity
            if not folder.exists():
                continue
            if folder.is_symlink() or not folder.resolve().is_relative_to(root):
                raise ValueError("refusing to remove an artifact outside benchmarks")
            expected = entry.get("source_files", entry["files"])
            observed = file_manifest(folder)
            # Git executable modes are metadata; compare source bytes and names.
            if {k: (v["sha256"], v["bytes"]) for k, v in observed.items()} != {
                k: (v["sha256"], v["bytes"]) for k, v in expected.items()
            }:
                raise ValueError(f"local artifacts changed; refusing removal: {folder}")
            shutil.rmtree(folder)
    rebuild_index(root)
    write_overview(root, data)
    from tools.campaign_report import index

    index(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmarks"))
    sub = parser.add_subparsers(dest="action", required=True)
    publish_parser = sub.add_parser(
        "publish", help="create local archive commits; does not push"
    )
    publish_parser.add_argument("--repository", dest="repository_url")
    publish_parser.add_argument("--branch-prefix", default="codex/artifacts")
    publish_parser.add_argument(
        "--first-shard",
        type=int,
        default=1,
        help="start at this shard when publishing a fresh batch",
    )
    publish_parser.add_argument(
        "--legacy-bootstrap",
        action="store_true",
        help="initial import only: seal historical reports without regenerating them",
    )
    check_parser = sub.add_parser(
        "check", help="audit catalog; optionally hash archived blobs"
    )
    check_parser.add_argument("--objects", action="store_true")
    check_parser.add_argument("--fetch", action="store_true")
    check_parser.add_argument(
        "--tracked",
        action="store_true",
        help="also reject payloads in the source Git index",
    )
    sub.add_parser(
        "prune", help="verify archive objects then remove unchanged local bundles"
    )
    fetch_parser = sub.add_parser(
        "fetch", help="fetch archive branch and cache one verified bundle"
    )
    fetch_parser.add_argument("kind", choices=("runs", "reports"))
    fetch_parser.add_argument("identity")
    args = parser.parse_args()
    if args.action == "publish":
        publish(
            args.root,
            repository_url=args.repository_url,
            branch_prefix=args.branch_prefix,
            first_shard=args.first_shard,
            legacy_bootstrap=args.legacy_bootstrap,
        )
    elif args.action == "check":
        check(args.root, objects=args.objects, fetch=args.fetch, tracked=args.tracked)
        print(
            "Benchmark archive catalog"
            + (" and objects" if args.objects else "")
            + " verified."
        )
    elif args.action == "prune":
        prune(args.root)
    else:
        print(archive.directory(args.root, args.kind, args.identity, fetch=True))


if __name__ == "__main__":
    main()
