"""Add split-context small-chunk attention to a selected verification/repair bundle."""

from akas.compilation.build import publish

import hashlib
import json
from pathlib import Path
import shutil
from akas.compilation.build import build_artifact
from akas.compilation.build import needs_build
from akas.compilation.build import export_source
from akas.artifacts.hybrid import identity
from akas.execution.optimizations.split_attention import implementation_hashes


def produce(
    prefill,
    out,
    *,
    splits=16,
    key_rows=64,
    query_tokens=None,
    kernel_target=None,
    pad_queries=False,
):
    prefill, out = Path(prefill).resolve(), Path(out).resolve()
    if out == prefill or out.is_relative_to(prefill):
        raise ValueError("use a separate output")
    manifest = json.loads((prefill / "prefill.json").read_text())
    if manifest.get("kv_dtype") != "fp8":
        raise ValueError("split kernel requires FP8 KV")
    out.mkdir(parents=True, exist_ok=True)
    for row in manifest["kernels"].values():
        source = (prefill / row["path"]).resolve()
        if (
            not source.is_relative_to(prefill)
            or hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]
        ):
            raise ValueError("source artifact checksum mismatch")
        shutil.copyfile(source, out / row["path"])
    p = {k: manifest[k] for k in ("slots", "chunk", "context")}
    p["capacity"] = p.pop("context")
    p["splits"] = splits
    for kind, factory in (("split_attention", "partial"), ("split_merge", "merge")):
        key = identity(kind, p)
        entry = out / (key + ".py")
        artifact = entry.with_suffix(".tbin")
        schedule = dict(p, packed_loads=True, key_rows=key_rows)
        module = "akas.kernels.attention.speculative_attention"
        target = manifest.get("target", "sm_89")
        if kind == "split_attention" and query_tokens is not None:
            module = "akas.kernels.attention.hopper_attention"
            schedule["query_tokens"] = query_tokens
            schedule["pad_queries"] = pad_queries
            target = kernel_target or target
        source = export_source(
            module,
            factory,
            schedule,
            dependencies=('tensor.compiler.frontend.source', "akas.kernels.elementwise"),
        )
        if needs_build(entry, artifact, source, target):
            entry.write_text(source)
            artifact.unlink(missing_ok=True)
            build_artifact(entry, artifact, target=target)
        manifest["kernels"][key] = dict(
            kind=kind,
            parameters=p,
            path=artifact.name,
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        )
        print(kind, schedule, flush=True)
    manifest.update(
        split_attention=dict(splits=splits, key_rows=key_rows),
        split_attention_implementation=implementation_hashes(),
    )
    if query_tokens is not None:
        manifest["split_attention"].update(
            query_tokens=query_tokens,
            kernel_target=kernel_target,
            pad_queries=pad_queries,
        )
    publish(out / "prefill.json", manifest)
    return out
