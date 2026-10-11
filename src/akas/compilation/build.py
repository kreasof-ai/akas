"""Shared source caching, compilation, checksums and atomic bundle publication."""

import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path

from akas.compilation.plan import Bundle, Kernel, Source


def _compiler_identity(provider):
    if provider == "cuda":
        path = Path(find_spec("tensor.compiler.cuda.lowering").origin)
        return {"lowering_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    from tensor.compiler.webgpu.identity import lowering_hash
    producer = Path(find_spec("tensor.compiler.webgpu.build").origin)
    return {
        "lowering_sha256": lowering_hash(),
        "producer_sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
    }


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
                for key, value in _compiler_identity(provider).items()
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
        nvrtc_home=os.environ.get("TENSOR_NVRTC_HOME", "build/nvrtc-12.9"),
    )


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path, value):
    """Expose a manifest only after every referenced artifact exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


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
        build_artifact(
            source,
            artifact,
            target=target,
            provider=provider,
            cache_dir=out / "compiler-cache",
        )
    record = dict(
        kind=kernel.kind, parameters=kernel.parameters, sha256=checksum(artifact)
    )
    record[path_field] = artifact.relative_to(out).as_posix()
    if kernel.schedule is not None:
        record["schedule"] = kernel.schedule
    return record


def build(bundle: Bundle, out, *, progress=None):
    """Build any profile's kernel declarations through one common pipeline."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / bundle.filename
    if not destination.resolve().is_relative_to(out.resolve()):
        raise ValueError("manifest path must remain inside the output bundle")
    if len({kernel.key for kernel in bundle.kernels}) != len(bundle.kernels):
        raise ValueError("duplicate kernel declaration")
    destination.unlink(missing_ok=True)
    manifest = dict(bundle.manifest)
    records = dict(manifest.get("kernels", {}))
    for kernel in bundle.kernels:
        records[kernel.key] = compile_kernel(
            out,
            kernel,
            target=kernel.target or bundle.target,
            provider=bundle.provider,
            source_directory=bundle.source_directory,
            artifact_directory=bundle.artifact_directory,
            path_field=bundle.path_field,
        )
        if progress:
            progress(kernel.kind, kernel.parameters)
    manifest["kernels"] = records
    if "status" in manifest:
        manifest["status"] = "built"
    publish(destination, manifest)
    return manifest


def export_source(module, factory, *args, dependencies=()):
    """Lazy entry serializer for measured operation overlays."""
    from tensor.compiler.frontend.source import export_source as export

    return export(module, factory, *args, dependencies=dependencies)
