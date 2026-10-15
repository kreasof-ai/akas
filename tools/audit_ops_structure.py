"""Inventory operation structure and declared backend coverage without compiling.

Run with the project environment: python -m tools.audit_ops_structure.
The JSON records declarations and source structure, not verified device support.
"""

import argparse
import ast
from collections import Counter
from importlib import import_module
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "akas"
PROVIDERS = ("cuda", "webgpu")


def python_files(directory):
    return sorted(
        (
            path
            for path in directory.rglob("*.py")
            if not any(
                (part.startswith(".") for part in path.relative_to(SOURCE).parts)
            )
        )
    )


def is_kernel(node):
    if not isinstance(node, ast.FunctionDef):
        return False
    for decorator in node.decorator_list:
        name = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(name, ast.Name) and name.id == "kernel":
            return True
    return False


def audit():
    from akas.kernels.registry import definition_modules

    declarations = []
    for module in definition_modules():
        definitions = import_module(module)
        for item in getattr(definitions, "IMPLEMENTATIONS", ()):
            declarations.append(
                dict(
                    owner=module.removeprefix("akas.").removesuffix(".definition"),
                    kind=item.kind,
                    contract=item.contract,
                    provider=item.provider,
                    phase=item.phase,
                    module=item.module_name,
                    factory=item.factory,
                    targets=item.targets,
                    required=item.required,
                    has_predicate=item.accepts is not None,
                    defaults=dict(item.defaults),
                    tuning_axes=[name for name, _ in item.space],
                    semantic_keys=item.semantic_keys,
                    identifier=item.identifier,
                    priority=item.priority,
                    validation_registered=item.kind
                    in getattr(definitions, "VALIDATIONS", {}),
                )
            )
    sources = []
    imports = []
    for path in python_files(SOURCE):
        relative = path.relative_to(SOURCE)
        text = path.read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.append(
                    dict(
                        file=str(relative),
                        line=node.lineno,
                        module=node.module,
                        symbols=[alias.name for alias in node.names],
                    )
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        dict(
                            file=str(relative),
                            line=node.lineno,
                            module=alias.name,
                            symbols=[],
                        )
                    )
        if relative.parts[0] not in ("ops", "quantization"):
            continue
        sources.append(
            dict(
                file=str(relative),
                lines=len(text.splitlines()),
                functions=[
                    dict(name=node.name, line=node.lineno)
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ],
                kernels=[
                    dict(name=node.name, line=node.lineno)
                    for node in ast.walk(tree)
                    if is_kernel(node)
                ],
            )
        )
    families = []
    for directory in sorted((SOURCE / "ops").iterdir()):
        if not directory.is_dir() or directory.name.startswith((".", "_")):
            continue
        files = [s for s in sources if s["file"].startswith(f"ops/{directory.name}/")]
        records = [d for d in declarations if d["owner"] == f"ops.{directory.name}"]
        keys = {
            provider: {
                (d["kind"], d["contract"], d["phase"])
                for d in records
                if d["provider"] == provider
            }
            for provider in PROVIDERS
        }
        families.append(
            dict(
                family=directory.name,
                files=[s["file"] for s in files],
                lines=sum((s["lines"] for s in files)),
                kernels=sum((len(s["kernels"]) for s in files)),
                registrations=dict(Counter((d["provider"] for d in records))),
                kinds=sorted({d["kind"] for d in records}),
                missing_validations=sorted(
                    {d["kind"] for d in records if not d["validation_registered"]}
                ),
                both_keys=sorted(keys["cuda"] & keys["webgpu"]),
                cuda_only_keys=sorted(keys["cuda"] - keys["webgpu"]),
                webgpu_only_keys=sorted(keys["webgpu"] - keys["cuda"]),
            )
        )
    formats = []
    for path in sorted((SOURCE / "quantization").glob("*/*/definition.py")):
        module = "akas." + ".".join(path.relative_to(SOURCE).with_suffix("").parts)
        format = import_module(module).FORMAT
        formats.append(
            dict(
                scope=path.parents[1].name,
                name=format.name,
                file=str(path.relative_to(SOURCE)),
                providers=dict(format.providers),
            )
        )
    operation_records = [d for d in declarations if d["owner"].startswith("ops.")]
    kinds = {
        provider: {d["kind"] for d in operation_records if d["provider"] == provider}
        for provider in PROVIDERS
    }
    return dict(
        limitations=[
            "Declarations are not evidence of successful compilation or device execution.",
            "Kind names include internal stages and hardware algorithms, not just public features.",
            "Matching keys do not prove matching geometry, predicates, layouts or numerical behavior.",
            "A forward registration can also serve other phases; exact phase-key differences may overstate gaps.",
            "Source imports do not include dynamically constructed module paths or arbitrary getattr calls.",
            "Inline codec helpers and direct factory callers can exist without registry declarations.",
        ],
        summary=dict(
            sources={
                scope: dict(
                    files=sum((s["file"].startswith(scope + "/") for s in sources)),
                    lines=sum(
                        (
                            s["lines"]
                            for s in sources
                            if s["file"].startswith(scope + "/")
                        )
                    ),
                    kernels=sum(
                        (
                            len(s["kernels"])
                            for s in sources
                            if s["file"].startswith(scope + "/")
                        )
                    ),
                )
                for scope in ("ops", "quantization")
            },
            registrations=dict(Counter((d["provider"] for d in declarations))),
            operation_kinds=len(kinds["cuda"] | kinds["webgpu"]),
            both_kinds=sorted(kinds["cuda"] & kinds["webgpu"]),
            cuda_only_kinds=sorted(kinds["cuda"] - kinds["webgpu"]),
            webgpu_only_kinds=sorted(kinds["webgpu"] - kinds["cuda"]),
        ),
        families=families,
        formats=formats,
        declarations=declarations,
        sources=sources,
        imports=imports,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/ops-structure-audit.json"
    )
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("family          files  kernels   CUDA  WebGPU registrations")
    for family in result["families"]:
        counts = family["registrations"]
        print(
            f"{family['family']:16}{len(family['files']):5}{family['kernels']:9}{counts.get('cuda', 0):7}{counts.get('webgpu', 0):8}"
        )
    print(f"Full inventory: {args.output}")


if __name__ == "__main__":
    main()
