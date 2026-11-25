"""Compile kernel declarations through Tensor and emit checked artifacts."""

import os
from pathlib import Path

from akas.artifacts.schema import Kernel, Source
from akas.compilation.source import (
    compiler_identity,
    export_kernel,
)
from akas.benchmarking.provenance import digest as checksum


def needs_build(source, destination, text, target, *, provider="cuda"):
    """Reuse only matching sources, provider, target and compiler implementation."""
    from tensor.artifacts.format import read_artifact

    if not source.is_file() or not destination.is_file() or source.read_text() != text:
        return True
    try:
        manifest, _ = read_artifact(destination)
        return (
            manifest["target"] != target
            or manifest.get("provider", "cuda") != provider
            or any(
                manifest.get("compiler", {}).get(key) != value
                for key, value in compiler_identity(provider).items()
            )
        )
    except (ValueError, KeyError, OSError):
        return True


def build_artifact(source, destination, *, target, provider="cuda", cache_dir=None):
    """Compile through Tensor; compiler imports stay behind this call."""
    import tensor

    return tensor.build(
        source,
        destination,
        provider=provider,
        target=target,
        compiler="nvrtc" if provider == "cuda" else "wgsl",
        cache_dir=cache_dir,
        nvrtc_home=(
            os.environ.get("TENSOR_NVRTC_HOME", "build/nvrtc-12.9")
            if provider == "cuda"
            else None
        ),
    )


def compile_kernel(
    out,
    kernel: Kernel,
    *,
    target,
    provider="cuda",
    source_directory="",
    artifact_directory="",
    path_field="path",
):
    import hashlib
    import shutil

    out = Path(out)
    source = out / source_directory / (kernel.key + ".py")
    artifact = out / (
        kernel.artifact or str(Path(artifact_directory) / (kernel.key + ".tbin"))
    )
    if not source.resolve().is_relative_to(
        out.resolve()
    ) or not artifact.resolve().is_relative_to(out.resolve()):
        raise ValueError("kernel paths must remain inside the output bundle")
    text = (
        kernel.source.render() if isinstance(kernel.source, Source) else kernel.source
    )
    source.parent.mkdir(parents=True, exist_ok=True)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if needs_build(source, artifact, text, target, provider=provider):
        artifact.unlink(missing_ok=True)
        source.write_text(text)
        search_pool = out / "search-kernels"
        search_pool.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(text.encode()).hexdigest()
        pooled_source = search_pool / (digest + ".py")
        pooled_artifact = search_pool / (digest + ".tbin")
        if not needs_build(
            pooled_source, pooled_artifact, text, target, provider=provider
        ):
            shutil.copyfile(pooled_artifact, artifact)
        else:
            build_artifact(
                source,
                artifact,
                target=target,
                provider=provider,
                cache_dir=out / "compiler-cache",
            )
            pooled_source.write_text(text)
            if pooled_artifact.resolve() != artifact.resolve():
                shutil.copyfile(artifact, pooled_artifact)
    record = dict(
        kind=kernel.kind, parameters=kernel.parameters, sha256=checksum(artifact)
    )
    record[path_field] = artifact.relative_to(out).as_posix()
    if kernel.schedule is not None:
        record["schedule"] = kernel.schedule
    return record


def export_source(module, factory, *args, dependencies=()):
    """Lazy entry serializer for measured operation overlays."""
    return export_kernel(module, factory, *args, dependencies=dependencies)
