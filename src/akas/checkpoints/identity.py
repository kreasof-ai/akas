"""Identify local model weights and cache directory digests."""

import hashlib
import json
from pathlib import Path


def checkpoint_digest(path):
    import os

    override = os.environ.get("AKAS_CHECKPOINT_SHA256")
    if override:
        return override
    path = Path(path)
    if path.is_file():
        from akas.benchmarking.provenance import digest

        return digest(path)
    files = (
        [path]
        if path.is_file()
        else [
            f
            for f in sorted([*path.glob("*.safetensors"), *path.glob("*.json")])
            if not f.name.startswith(".")
        ]
    )
    if not files:
        raise ValueError("checkpoint has no local weights/configuration")
    manifest = [[f.name, f.stat().st_size] for f in files]
    sidecar = path / ".akas-checkpoint-sha256"
    if sidecar.is_file():
        try:
            cached = json.loads(sidecar.read_text())
            if cached.get("files") == manifest and isinstance(
                cached.get("sha256"), str
            ):
                return cached["sha256"]
        except (OSError, ValueError):
            pass
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    result = digest.hexdigest()
    try:
        sidecar.write_text(
            json.dumps(dict(files=manifest, sha256=result), indent=2) + "\n"
        )
    except OSError:
        pass
    return result
