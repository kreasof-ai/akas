"""Add native MTP input adapters to a selected chunked prefill bundle."""

import hashlib
import json
from pathlib import Path

from akas.compilation.build import build
from akas.compilation.plan import Bundle
from akas.compilation.plan import Kernel
from akas.compilation.plan import Source
from akas.compilation.bundles import copy
from akas.profiles.fixed_slots.geometry import validate
from akas.artifacts.kernels import identity
from akas.artifacts.fingerprints import (
    fixed_slots_hashes as implementation_hashes,
)


def produce(checkpoint, prefill, out):
    validate(checkpoint.config)
    prefill, out = Path(prefill).resolve(), Path(out).resolve()
    if out == prefill or out.is_relative_to(prefill):
        raise ValueError("use a separate MTP prefill output")
    manifest = json.loads((prefill / "prefill.json").read_text())
    if manifest["schema"] != "tensor.qwen35-prefill.v1":
        raise ValueError("unsupported prefill schema")
    out.mkdir(parents=True, exist_ok=True)
    (out / "prefill.json").unlink(missing_ok=True)
    manifest = copy(prefill, out, "prefill.json")
    kernels = []
    r = manifest["slots"] * manifest["chunk"]
    for kind, p in (
        (
            "mtp_join",
            dict(r=r, c=checkpoint.config.width, eps=checkpoint.config.epsilon),
        ),
        ("mtp_cast", dict(r=r, c=checkpoint.config.width)),
        (
            "bf16_linear",
            dict(r=r, k=2 * checkpoint.config.width, o=checkpoint.config.width),
        ),
    ):
        key = identity(kind, p)
        module, factory, args = (
            "akas.ops.mtp.cuda",
            "make_kernel",
            (kind, p),
        )
        if kind == "bf16_linear":
            module, factory, args = (
                "akas.ops.vocabulary.cuda.bf16",
                "bf16_head_kernel",
                (dict(p, columns=128, depth=128),),
            )
        kernels.append(Kernel(key, kind, p, Source(module, factory, args)))
    manifest.update(
        draft="qwen35-mtp",
        implementation=implementation_hashes(),
        target_prefill_lineage=dict(
            path=str(prefill),
            manifest_sha256=hashlib.sha256(
                (prefill / "prefill.json").read_bytes()
            ).hexdigest(),
        ),
        speculative_verification=False,
        full_stress_target_reached=False,
    )
    build(Bundle(manifest, "prefill.json", kernels), out)
    return out
