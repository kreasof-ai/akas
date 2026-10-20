"""Compile kernel declarations through Tensor and emit checked artifacts."""

import os
from functools import lru_cache
from pathlib import Path

from akas.artifacts.fingerprint import _compiler_identity
from akas.artifacts.schema import Kernel, Source
from akas.artifacts.write import checksum


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
        digest = hashlib.sha256(text.encode()).hexdigest()
        pooled_source = search_pool / (digest + ".py")
        pooled_artifact = search_pool / (digest + ".tbin")
        if search_pool.is_dir() and not needs_build(
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
            if search_pool.is_dir():
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


def source_dependencies(module, *, dependencies=()):
    """Hash the selected operation and its transitive source imports."""
    return dependency_closure(module, dependencies)


@lru_cache(maxsize=512)
def _source_imports(package, text):
    """Cache parsed imports by content; edits cannot reuse an outdated scan."""
    import ast

    imports = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                base += "." + node.module if node.module else ""
            else:
                base = node.module or ""
            imported = [base, *(base + "." + alias.name for alias in node.names)]
        else:
            continue
        imports.extend(
            item for item in imported if item.startswith(("akas.", "tensor.compiler."))
        )
    return tuple(imports)


def dependency_closure(module, dependencies):
    """Hash the selected factory's imports rather than unrelated executors.

    Imports inside factories and macros count too. Codec discovery is dynamic,
    so local codec sources are included when a factory uses the format API.
    Explicit dependencies cover factories that import a module supplied as data.
    """
    from importlib.util import find_spec

    pending = [module, *dependencies]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        try:
            spec = find_spec(name)
        except (ModuleNotFoundError, AttributeError):
            continue  # An imported symbol, rather than a child module.
        if spec is None or spec.origin is None:
            continue
        path = Path(spec.origin)
        if path.suffix != ".py":
            continue
        visited.add(name)
        if name == "akas.quantization.formats":
            # Format lookup imports modules by name. Include local codec sources
            # so adding/editing a dynamically selected codec invalidates artifacts
            # without a second, manually maintained registration table.
            pending.extend(
                "akas.quantization."
                + ".".join(p.relative_to(path.parent).with_suffix("").parts)
                for p in path.parent.rglob("*.py")
                if p.stem != "__init__"
            )
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        pending.extend(_source_imports(package, path.read_text()))
    return tuple(sorted(visited - {module}))


def export_kernel(module, factory, *arguments, dependencies=()):
    from tensor.compiler.frontend.source import export_source

    return export_source(
        module,
        factory,
        *arguments,
        dependencies=source_dependencies(module, dependencies=dependencies),
    )
