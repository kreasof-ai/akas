"""Link qualified decode and prefill phases with identical checkpoint ABIs.

This preserves each producer's immutable phase manifests, kernels and search
evidence. It does not compile or retime a kernel; benchmark the linked program
before claiming a model-level gain.
"""

import argparse
import copy
import json
from pathlib import Path
import re
import shutil

from akas.artifacts.read import artifact_path, read_program, validate_program
from akas.benchmarking.history import digest, write


def link(prefill, decode, output):
    prefill, decode, output = (Path(p).resolve() for p in (prefill, decode, output))
    if output.is_relative_to(prefill) or output.is_relative_to(decode):
        raise ValueError("linked output must be outside producer directories")
    a, b = read_program(prefill), read_program(decode)
    for key in ("provider", "target", "config", "max_requests", "max_context", "model_options", "upload_options"):
        if a.get(key) != b.get(key):
            raise ValueError("programs differ in " + key)
    if a.get("rounds") or b.get("rounds"):
        raise ValueError("link only autoregressive programs without speculation rounds")
    selected = [(prefill, row) for row in a["phases"] if row["phase"] != "decode"]
    selected += [(decode, row) for row in b["phases"] if row["phase"] == "decode"]
    if not selected or not any(row["phase"] == "decode" for _, row in selected):
        raise ValueError("missing linked token phases")
    inventory = {}
    generations = set()
    for root, row in selected:
        if row.get("commit") or row.get("journal"):
            raise ValueError("linked phases must be ordinary autoregressive calls")
        manifest = json.loads((root / row["path"]).read_text())
        if not manifest.get("tuning", {}).get("operations"):
            raise ValueError("phase lacks independent kernel qualification")
        for record in manifest["kernels"].values():
            artifact_path(root, record)
        for spec in manifest["forward"]["buffers"]:
            if spec["role"] in ("state", "weight") and spec.get("alias_of") is None:
                identity = spec["role"], spec["name"]
                previous = inventory.setdefault(identity, spec)
                if previous != spec:
                    raise ValueError("incompatible linked weight/state ABI: " + spec["name"])
        path = Path(row["path"])
        if len(path.parts) < 3 or path.parts[0] != "programs" or not re.fullmatch("[0-9a-f]{32}", path.parts[1]):
            raise ValueError("link requires immutable compiler generation paths")
        generations.add((root, Path(*path.parts[:2])))
    if output.exists() and any(output.iterdir()):
        raise ValueError("linked output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    reset = a.get("reset")
    if reset:
        path = Path(reset["path"])
        if len(path.parts) < 3 or path.parts[0] != "programs" or not re.fullmatch("[0-9a-f]{32}", path.parts[1]):
            raise ValueError("reset requires an immutable compiler generation path")
        generations.add((prefill, Path(*path.parts[:2])))
    for root, generation in sorted(generations):
        destination = output / generation
        if destination.exists():
            raise ValueError("compiler generation identity collision")
        shutil.copytree(root / generation, destination)
    result = copy.deepcopy(a)
    result["phases"] = [copy.deepcopy(row) for _, row in selected]
    result["policy"] = {**a.get("policy", {}), **b.get("policy", {})}
    result["linked_from"] = [dict(role=role, program_sha256=digest(root / "program.json"),
                                   policy=data.get("policy", {}))
                            for role, root, data in (("prefill", prefill, a), ("decode", decode, b))]
    validate_program(output, result)
    write(output / "program.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    link(args.prefill, args.decode, args.out)
