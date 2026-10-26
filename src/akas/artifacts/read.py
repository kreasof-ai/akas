"""Validated loading of compiled program kernels."""

import hashlib
from pathlib import Path

def artifact_path(directory, record):
    """Resolve and check a physical artifact for either manifest format."""
    directory = Path(directory).resolve()
    descriptor = record.get("logical", record)
    path = (
        directory / descriptor.get("path", descriptor.get("artifact", ""))
    ).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("artifact path escapes its bundle or is missing")
    with path.open("rb") as image:
        checksum = hashlib.file_digest(image, "sha256").hexdigest()
    if checksum != descriptor["sha256"]:
        raise ValueError("artifact checksum/integrity mismatch")
    return path


def load_kernels(device, directory, records, *, requirements=None, cache=None):
    """Share integrity checks and cleanup across all executor/model adapters."""
    if requirements is not None and records.keys() != requirements.keys():
        raise ValueError("incomplete/incompatible kernel coverage")
    kernels, owned = {}, []
    try:
        for key, row in records.items():
            if requirements is not None:
                kind, parameters = requirements[key]
                if row["kind"] != kind or row["parameters"] != parameters:
                    raise ValueError("artifact identity mismatch")
            path = artifact_path(directory, row)
            checksum = row.get("logical", row)["sha256"]
            if cache is not None and checksum in cache:
                kernels[key] = cache[checksum]
            else:
                kernels[key] = device.load(path)
                if cache is None:
                    owned.append(kernels[key])
                else:
                    cache[checksum] = kernels[key]
        return kernels
    except BaseException:
        for kernel in owned:
            kernel.release()
        raise


def artifacts(records):
    for row in records.values():
        yield row
        for role in ("logical", "quantize", "merge"):
            if role in row:
                yield row[role]




def read_program(directory):
    """Read one inference program index and validate every phase manifest."""
    import json

    root = Path(directory).resolve()
    manifest = json.loads((root/"program.json").read_text())
    return validate_program(root, manifest)


def validate_program(directory, manifest):
    """Check an unpublished program using the same contract as its loader."""
    import json

    root = Path(directory).resolve()
    if manifest.get("schema") != "akas.program.v1":
        raise ValueError("unsupported inference program")
    if any(type(manifest.get(name)) is not int or manifest[name] < 1 for name in ("max_requests","max_context")):
        raise ValueError("invalid program capacity")
    records = manifest.get("phases")
    if not isinstance(records,list) or not records:
        raise ValueError("program has no compiled phases")
    identities = set()
    states = {}
    for row in records:
        identity = (row["phase"],row["rows"],row["chunk"])
        if identity in identities or any(type(n) is not int or n < 1 for n in identity[1:]):
            raise ValueError("invalid or duplicate program geometry")
        identities.add(identity)
        if row["rows"] > manifest["max_requests"] or row["chunk"] > manifest["max_context"]:
            raise ValueError("phase exceeds program capacity")
        path = (root/row["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("phase path escapes program or is missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("phase manifest checksum mismatch")
        phase = json.loads(path.read_text())
        if phase.get("provider") != manifest["provider"] or phase.get("target") != manifest["target"]:
            raise ValueError("phase target/provider mismatch")
        buffers = {spec["name"]:spec for spec in phase["forward"]["buffers"]}
        for name in (*row["inputs"].values(),row["prediction"], *row.get("features",{}).values()):
            if name not in buffers:
                raise ValueError("undeclared phase input/output binding")
        commit = row.get("commit")
        if commit is not None:
            commit_path = (root/commit["path"]).resolve()
            if not commit_path.is_relative_to(root) or not commit_path.is_file():
                raise ValueError("commit path escapes program or is missing")
            if hashlib.sha256(commit_path.read_bytes()).hexdigest() != commit["sha256"]:
                raise ValueError("commit phase checksum mismatch")
            commit_manifest = json.loads(commit_path.read_text())
            if commit_manifest.get("provider") != manifest["provider"] or commit_manifest.get("target") != manifest["target"]:
                raise ValueError("commit target/provider mismatch")
            for spec in commit_manifest["forward"]["buffers"]:
                if buffers.get(spec["name"]) != spec:
                    raise ValueError("commit buffer differs from verification phase")
        for name,spec in buffers.items():
            if spec["role"] == "state":
                previous = states.setdefault(name,spec)
                if previous != spec:
                    raise ValueError("shared state layout differs across phase geometries")
        for descriptor in row.get("journal",{}).values():
            journal = json.loads(artifact_path(root,descriptor).read_text())
            if journal.get("provider") != manifest["provider"] or journal.get("target") != manifest["target"]:
                raise ValueError("journal target/provider mismatch")
            for spec in journal["forward"]["buffers"]:
                if spec["role"] == "state" and spec.get("alias_of") is None:
                    previous = states.setdefault(spec["name"],spec)
                    if previous != spec:
                        raise ValueError("journal state differs across phases")
    reset = manifest.get("reset")
    if reset is not None:
        path = (root/reset["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("reset phase path escapes program or is missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != reset["sha256"]:
            raise ValueError("reset phase checksum mismatch")
        phase = json.loads(path.read_text())
        if phase.get("provider") != manifest["provider"] or phase.get("target") != manifest["target"]:
            raise ValueError("reset target/provider mismatch")
        buffers = {spec["name"]:spec for spec in phase["forward"]["buffers"]}
        if reset["mapping"] not in buffers:
            raise ValueError("undeclared reset mapping")
        for name,spec in buffers.items():
            if spec["role"] == "state" and spec.get("alias_of") is None and states.get(name) != spec:
                raise ValueError("reset state layout differs from token phases")
    rounds = set()
    for record in manifest.get("rounds", ()):
        identity = (record["rows"], record["window"])
        if identity in rounds or ("verify",*identity) not in identities:
            raise ValueError("round controls have no matching verification geometry")
        rounds.add(identity)
        path = artifact_path(root, record)
        controls = json.loads(path.read_text())
        if controls.get("provider") != manifest["provider"] or controls.get("target") != manifest["target"]:
            raise ValueError("round target/provider mismatch")
        if controls.get("schema") != "akas.forward.v1" or controls["forward"]["contract"] != "speculation":
            raise ValueError("invalid round controls")
    return manifest
