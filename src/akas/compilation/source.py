"""Serialize producer entry points with the complete operation dependency closure."""

from pathlib import Path


def operation_modules():
    root = Path(__file__).resolve().parents[1]
    return tuple(
        "akas." + str(path.relative_to(root).with_suffix("")).replace("/", ".")
        for folder in ("ops", "execution/bindings", "compilation/factories")
        for path in sorted((root / folder).rglob("*.py"))
        if path.name != "__init__.py"
    )


def source_dependencies():
    return tuple(
        sorted(
            {
                "akas.compilation.source",
                "akas.checkpoints.gguf",
                "tensor.compiler.frontend.source",
                "tensor.compiler.cuda.lowering",
                *(
                    "tensor.compiler.webgpu.schedules." + name
                    for name in (
                        "gemv",
                        "outer_product",
                        "packed_integer",
                        "packed_outer_product",
                        "partitioned",
                        "register_matmul",
                        "registers",
                    )
                ),
                *operation_modules(),
            }
        )
    )


def export_kernel(module, factory, *arguments, dependencies=()):
    from tensor.compiler.frontend.source import export_source

    return export_source(
        module,
        factory,
        *arguments,
        dependencies=tuple(sorted(set((*dependencies, *source_dependencies())))),
    )
