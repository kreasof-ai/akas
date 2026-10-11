"""Read and copy complete bundles with checked artifact paths and checksums."""

import json
from pathlib import Path
import shutil
from akas.compilation.build import checksum


def artifacts(records):
    for row in records.values():
        yield row
        for role in ("logical", "quantize", "merge"):
            if role in row:
                yield row[role]


def copy(source, out, filename="prefill.json"):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("output must be separate from the input bundle")
    manifest = json.loads((source / filename).read_text())
    rows = list(artifacts(manifest["kernels"]))
    # Validate everything before changing the destination.
    for row in rows:
        relative = Path(row.get("path", row.get("artifact", "")))
        if relative.is_absolute() or not (out / relative).resolve().is_relative_to(out):
            raise ValueError("source artifact must use a relative bundle path")
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or checksum(path) != row["sha256"]:
            raise ValueError("source artifact path or checksum mismatch")
    out.mkdir(parents=True, exist_ok=True)
    (out / filename).unlink(missing_ok=True)
    for row in rows:
        relative = row.get("path", row.get("artifact"))
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
        entry = (source / relative).with_suffix(".py")
        if entry.is_file():
            shutil.copyfile(entry, destination.with_suffix(".py"))
    return manifest
