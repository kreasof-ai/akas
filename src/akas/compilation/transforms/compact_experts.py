"""Build an opt-in compact expert schedule beside an existing prefill bundle."""

from akas.compilation.build import build
from akas.compilation.plan import Bundle
from akas.compilation.plan import Kernel
from akas.compilation.plan import Source

import json
from pathlib import Path
import shutil
from akas.artifacts.kernels import identity
from akas.artifacts.fingerprints import (
    block_fp8_hashes as implementation_hashes,
)

import hashlib


def produce(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out:
        raise ValueError("compact output must be separate from the control bundle")
    out.mkdir(parents=True, exist_ok=True)
    for p in source.iterdir():
        if p.is_file():
            shutil.copy2(p, out / p.name)
    manifest = json.loads((source / "prefill.json").read_text())
    rows = manifest["slots"] * manifest["chunk"]
    m = manifest["schedule"]["block_m"]
    record = dict(max_tiles=(rows * 8 + m - 1) // m + 255, block_m=m, projections={})

    kernels = []

    def add(kind, parameters, module, factory, specialization):
        key = identity(kind, parameters)
        kernels.append(
            Kernel(
                key,
                kind,
                parameters,
                Source(module, factory, (specialization,)),
                target=manifest.get("target", "sm_89"),
            )
        )
        return key

    p = dict(rows=rows, block_m=m)
    record["tile_map"] = add(
        "expert_tile_map", p, "akas.ops.moe.cuda.compact", "tile_map_kernel", p
    )
    for logical, row in list(manifest["kernels"].items()):
        if row["kind"] != "fp8_experts":
            continue
        p = row["parameters"]
        schedule = dict(
            rows=rows,
            k=p["k"],
            o=p["o"],
            routed_input=p["routed_input"],
            block_m=m,
            threads=256 if m >= 32 else 128,
            compact=True,
        )
        selected = add(
            "compact_fp8_experts",
            p,
            "akas.ops.moe.cuda.prefill",
            "expert_kernel",
            schedule,
        )
        record["projections"][logical] = selected
    manifest["compact_experts"] = record
    manifest["compact_expert_implementation"] = implementation_hashes()
    build(Bundle(manifest, "prefill.json", kernels), out)
    return out
