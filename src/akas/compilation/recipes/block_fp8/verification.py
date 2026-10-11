"""Produce small-chunk greedy verification and recurrent rollback artifacts."""

from akas.compilation.build import build
from akas.compilation.plan import Bundle
from akas.compilation.plan import Kernel
from akas.compilation.plan import Source
from akas.execution.bindings.block_fp8.geometry import from_manifest

import hashlib
import json
import math
from pathlib import Path
import shutil
from akas.artifacts.kernels import identity
from akas.artifacts.fingerprints import (
    block_fp8_hashes as implementation_hashes,
)


def produce(prefill, out):
    prefill, out = Path(prefill).resolve(), Path(out).resolve()
    if out == prefill or out.is_relative_to(prefill):
        raise ValueError("use a separate verification output")
    manifest = json.loads((prefill / "prefill.json").read_text())
    slots, chunk = manifest["slots"], manifest["chunk"]
    geometry = from_manifest(manifest)
    if chunk < 2 or chunk > 128:
        raise ValueError("verification chunk must be between 2 and 128")
    out.mkdir(parents=True, exist_ok=True)
    for row in manifest["kernels"].values():
        source = (prefill / row["path"]).resolve()
        if (
            not source.is_relative_to(prefill)
            or hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]
        ):
            raise ValueError("prefill artifact checksum mismatch")
        shutil.copyfile(source, out / row["path"])

    kernels = []

    def add(kind, p, module, factory, *args):
        key = identity(kind, p)
        kernels.append(
            Kernel(
                key,
                kind,
                p,
                Source(module, factory, args),
                target=manifest.get("target", "sm_89"),
            )
        )
        return key

    for kind in ("gdn_scan", "gdn_conv"):
        p = dict(slots=slots, chunk=chunk)
        add(
            "spec_" + kind,
            p,
            "akas.compilation.factories.block_fp8",
            "verification_factory",
            kind,
            p,
        )
    for shape in (
        (geometry["value_heads"], geometry["key_dim"], geometry["value_dim"]),
        (
            2 * geometry["key_heads"] * geometry["key_dim"]
            + geometry["value_heads"] * geometry["value_dim"],
            geometry["conv_width"] - 1,
        ),
    ):
        p = dict(slots=slots, chunk=chunk, shape=list(shape))
        # Flat views compare bit counts through an int32 literal. A short
        # window also exceeds it when sufficiently many states are resident.
        if chunk > 8 or slots * (chunk - 1) * math.prod(shape) * 32 >= 2**31:
            add(
                "restore",
                p,
                "akas.ops.gdn.cuda.hopper_restore",
                "restore_kernel",
                p,
            )
        else:
            add(
                "restore",
                p,
                "akas.compilation.factories.block_fp8",
                "verification_factory",
                "restore",
                p,
            )
    p = dict(r=slots * chunk, k=geometry["width"], o=geometry["vocab"])
    add(
        "verify_head",
        p,
        "akas.ops.vocabulary.cuda.speculative",
        "head_kernel",
        dict(
            p,
            block_m=16 if slots * chunk <= 16 else (64 if slots * chunk >= 64 else 32),
            columns=128,
            depth=128,
            threads=128,
        ),
    )
    p = dict(r=slots * chunk, vocab=geometry["vocab"])
    add(
        "argmax",
        p,
        "akas.compilation.factories.block_fp8",
        "decode_factory",
        "argmax",
        p,
    )
    manifest.update(
        verification="greedy-prefix-snapshots",
        implementation=implementation_hashes(),
        speculative_verification=True,
        model_throughput_qualified=False,
        prefill_lineage=dict(
            path=str(prefill),
            manifest_sha256=hashlib.sha256(
                (prefill / "prefill.json").read_bytes()
            ).hexdigest(),
        ),
    )
    build(Bundle(manifest, "prefill.json", kernels), out)
    return out
