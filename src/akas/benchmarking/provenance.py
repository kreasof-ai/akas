"""Durable identities and evidence serialization shared by benchmark producers."""

from datetime import datetime, timezone
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import uuid


def utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value, *, compact_entries=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if compact_entries:
        # One line per run/bundle keeps generated catalogs small and their diffs
        # useful without changing the ordinary evidence JSON serialization.
        encode = lambda item: json.dumps(item, allow_nan=False, separators=(",", ":"))
        fields = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                opening, closing = ("{", "}") if isinstance(item, dict) else ("[", "]")
                entries = (
                    [encode(k) + ":" + encode(v) for k, v in item.items()]
                    if isinstance(item, dict)
                    else [encode(v) for v in item]
                )
                body = (
                    opening
                    + "\n"
                    + ",\n".join("    " + e for e in entries)
                    + "\n  "
                    + closing
                )
            else:
                body = encode(item)
            fields.append("  " + encode(key) + ": " + body)
        content = "{\n" + ",\n".join(fields) + "\n}\n"
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def source_identity(root):
    """Hash the working source, including uncommitted and untracked source files."""
    root = Path(root).resolve()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, check=True
        ).stdout

    try:
        revision = git("rev-parse", "HEAD").decode().strip()
        paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
        names = paths.decode().strip("\0").split("\0")
        dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
        source_suffixes = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".ptx"}
        names = [
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.suffix in source_suffixes
        ]
    files = {}
    for name in sorted(set(names)):
        # Results must not recursively become part of the measured source.
        if not name or name.startswith("benchmarks/") or ".ipynb_checkpoints" in name:
            continue
        if revision is not None and not (
            name.startswith(
                ("src/", "tools/", "tests/", "examples/", "model/", "servers/")
            )
            or name in ("pyproject.toml", "uv.lock", "MANIFEST.in")
            or ("/" not in name and name.endswith(".py"))
        ):
            continue
        path = root / name
        if path.is_symlink():
            files[name] = hashlib.sha256(
                path.readlink().as_posix().encode()
            ).hexdigest()
        elif path.is_file():
            files[name] = digest(path)
    tree = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(
        revision=revision,
        dirty=dirty,
        tree_sha256=tree,
        files=files,
    )


def source_hashes(names):
    # Resolve source paths without importing operation factories or compilers.
    return {
        name: hashlib.sha256(Path(find_spec(name).origin).read_bytes()).hexdigest()
        for name in sorted(names)
    }
