"""Commit-addressed benchmark storage; catalog reads never fetch implicitly."""

import hashlib
from functools import lru_cache
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import quote

SCHEMA = "akas.benchmark-archive.v1"


def git(root, *args, input=None, env=None):
    result = subprocess.run(
        ["git", "-C", str(root), *args], input=input, capture_output=True, env=env
    )
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def repository(root):
    return Path(git(root, "rev-parse", "--show-toplevel").decode().strip())


def safe_path(value):
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or "\0" in value
        or path.as_posix() != value
    ):
        raise ValueError(f"unsafe archive path: {value!r}")
    return path


def catalog(root):
    path = Path(root) / "archive.json"
    if not path.exists():
        return dict(schema=SCHEMA, repository=None, runs={}, reports={})
    stat = path.stat()
    return _catalog(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _catalog(path, modified, size):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError("unsupported benchmark archive catalog")
    if not re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", data["repository"]):
        raise ValueError("archive repository must be a GitHub HTTPS repository URL")
    locations = set()
    for kind in ("runs", "reports"):
        for identity, entry in data[kind].items():
            if "repository" in entry and not re.fullmatch(
                r"https://github\.com/[\w.-]+/[\w.-]+", entry["repository"]
            ):
                raise ValueError("archive entry repository must be a GitHub HTTPS URL")
            if len(safe_path(identity).parts) != 1:
                raise ValueError("archive ID must be one path component")
            safe_path(entry["branch"])
            if not re.fullmatch(r"[0-9a-f]{40}", entry["commit"]):
                raise ValueError("archive requires full immutable commit hashes")
            if entry["path"] != f"benchmarks/{kind}/{identity}":
                raise ValueError("archive path does not match its ID")
            if entry["path"] in locations:
                raise ValueError("duplicate archive path")
            locations.add(entry["path"])
            required = "run.json" if kind == "runs" else "report.json"
            if required not in entry["files"]:
                raise ValueError(f"archive missing {required}")
            for name, info in entry["files"].items():
                safe_path(name)
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", info["sha256"])
                    or not isinstance(info["bytes"], int)
                    or info["bytes"] < 0
                    or info.get("mode", "100644") not in {"100644", "100755"}
                ):
                    raise ValueError("invalid archive file checksum, size or mode")
            if kind == "runs" and entry["summary"]["id"] != identity:
                raise ValueError("archive summary does not match its ID")
            for name, original in entry.get("source_files", {}).items():
                safe_path(name)
                copies = (
                    entry["files"].get(name),
                    entry["files"].get("originals/" + name),
                )
                if not any(
                    copy
                    and copy["sha256"] == original["sha256"]
                    and copy["bytes"] == original["bytes"]
                    for copy in copies
                ):
                    raise ValueError(
                        "original report file is not preserved in the archive"
                    )
    return data


def url(data, entry, name="", *, raw=False):
    if name:
        safe_path(name)
    path = entry["path"] + ("/" + name if name else "")
    repository_url = entry.get("repository", data["repository"])
    if raw:
        repo = repository_url.removeprefix("https://github.com/")
        return (
            f"https://raw.githubusercontent.com/{repo}/{entry['commit']}/{quote(path)}"
        )
    view = "blob" if name else "tree"
    return f"{repository_url}/{view}/{entry['commit']}/{quote(path)}"


def link(root, kind, identity, name="", *, raw=False):
    data = catalog(root)
    entry = data[kind].get(identity)
    return url(data, entry, name, raw=raw) if entry else None


def ensure_commit(root, entry, *, fetch=False):
    repo = repository(root)
    try:
        git(repo, "cat-file", "-e", entry["commit"] + "^{commit}")
    except ValueError:
        if not fetch:
            raise ValueError(
                f"Archive commit {entry['commit']} is unavailable; fetch "
                f"origin {entry['branch']} (without --depth), or use "
                "python -m tools.benchmark_archive fetch runs RUN_ID"
            ) from None
        refspec = f"refs/heads/{entry['branch']}:refs/remotes/origin/{entry['branch']}"
        git(repo, "fetch", "--no-tags", "origin", refspec)
        try:
            git(repo, "cat-file", "-e", entry["commit"] + "^{commit}")
        except ValueError:
            # Recover an older snapshot even if this archive branch was itself
            # fetched shallowly earlier. Deepen only the requested archive ref.
            git(repo, "fetch", "--no-tags", "--depth=2147483647", "origin", refspec)
            git(repo, "cat-file", "-e", entry["commit"] + "^{commit}")
    return repo


def read_file(root, kind, identity, name, *, fetch=False):
    safe_path(name)
    if kind not in {"runs", "reports"} or len(safe_path(identity).parts) != 1:
        raise ValueError("invalid artifact kind or ID")
    data = catalog(root)
    entry = data[kind].get(identity)
    local = Path(root) / kind / identity / name
    if entry is None:
        return local.read_bytes()
    info = entry["files"].get(name)
    if info is None:
        raise ValueError(f"file is absent from archive manifest: {identity}/{name}")
    # A restored/local copy is usable only if it still matches the sealed archive.
    # Historical report originals can remain locally until prune; their navigable
    # archived pages deliberately have different links.
    payload = local.read_bytes() if local.exists() else None
    if payload is not None and hashlib.sha256(payload).hexdigest() != info["sha256"]:
        original = entry.get("source_files", {}).get(name)
        if original and hashlib.sha256(payload).hexdigest() == original["sha256"]:
            payload = None
    if payload is None:
        payload = git(
            ensure_commit(root, entry, fetch=fetch),
            "show",
            entry["commit"] + ":" + entry["path"] + "/" + name,
        )
    if (
        len(payload) != info["bytes"]
        or hashlib.sha256(payload).hexdigest() != info["sha256"]
    ):
        raise ValueError(f"archive checksum mismatch: {identity}/{name}")
    return payload


def directory(root, kind, identity, *, fetch=False):
    """Materialize one verified bundle in an ignored cache, never all history."""
    safe_path(identity)
    data = catalog(root)
    entry = data[kind].get(identity)
    if entry is None:
        return Path(root).resolve() / kind / identity
    repo = ensure_commit(root, entry, fetch=fetch)
    cache = repo / "build/benchmark-archive" / entry["commit"] / entry["path"]
    for name, info in entry["files"].items():
        destination = cache / name
        if destination.is_file():
            payload = destination.read_bytes()
            if (
                len(payload) == info["bytes"]
                and hashlib.sha256(payload).hexdigest() == info["sha256"]
            ):
                continue
        payload = read_file(root, kind, identity, name, fetch=fetch)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        if info.get("mode") == "100755":
            destination.chmod(0o755)
    return cache


def rewrite_links(content, source, root):
    """Pin Markdown artifact links while leaving other prose and assets intact."""
    root = Path(root).resolve()

    def replace(match):
        target = match["target"]
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
            return match[0]
        path, marker, anchor = target.partition("#")
        resolved = (Path(source).parent / path).resolve()
        if not resolved.is_relative_to(root):
            return match[0]
        parts = resolved.relative_to(root).parts
        if len(parts) < 3 or parts[0] not in {"runs", "reports"}:
            return match[0]
        destination = link(
            root, parts[0], parts[1], "/".join(parts[2:]), raw=bool(match["image"])
        )
        if destination:
            return f"{match['image'] or ''}[{match['label']}]({destination}{marker}{anchor})"
        return match[0]

    return re.sub(
        r"(?P<image>!)?\[(?P<label>[^\]\n]*)\]\((?P<target>[^)\n]+)\)", replace, content
    )
