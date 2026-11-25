"""Export kernel factories and track their source dependencies."""

from functools import lru_cache
import hashlib
from importlib.util import find_spec
from pathlib import Path


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


def compiler_identity(provider):
    if provider == "cuda":
        path = Path(find_spec("tensor.compiler.cuda.lowering").origin)
        result = {"lowering_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for module, key in (
            ("tensor.compiler.frontend.signature", "signature_sha256"),
            ("tensor.compiler.cuda.tensor_map", "tensor_map_sha256"),
        ):
            spec = find_spec(module)
            if spec is not None:
                result[key] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
        return result
    from tensor.compiler.webgpu.identity import lowering_hash

    producer = Path(find_spec("tensor.compiler.webgpu.build").origin)
    return {
        "lowering_sha256": lowering_hash(),
        "producer_sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
    }
