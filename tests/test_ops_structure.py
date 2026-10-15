"""The uniform authoring tree and routing are enforced, not just documented."""

import ast
from importlib import import_module
from pathlib import Path

from akas.kernels.registry import definition_modules, implementations


ROOT = Path(__file__).parents[1] / "src" / "akas"
FILES = {
    "__init__.py",
    "definition.py",
    "kernel.py",
    "cuda.py",
    "webgpu.py",
    "validation.py",
}


def test_every_operation_and_codec_has_the_same_runtime_files():
    families = sorted((ROOT / "ops").glob("*/definition.py"))
    codecs = sorted((ROOT / "quantization").glob("*/*/definition.py"))
    assert families and codecs
    assert {p.parent for p in families} == {
        p
        for p in (ROOT / "ops").iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    }
    for definition in families + codecs:
        package = definition.parent
        assert {p.name for p in package.glob("*.py")} == FILES, package
        assert not any(
            p.is_dir() and not p.name.startswith((".", "_")) for p in package.iterdir()
        ), package
        assert (package / "README.md").exists(), package
    assert not (ROOT / "ops" / "state").exists()
    assert not (ROOT / "ops" / "reductions").exists()


def test_discovery_reads_only_family_definitions_and_keeps_backend_inventories():
    assert all(name.endswith(".definition") for name in definition_modules())
    choices = implementations()
    assert choices
    for name in definition_modules():
        definition = import_module(name)
        for provider in ("cuda", "webgpu"):
            backend = import_module(name.removesuffix(".definition") + "." + provider)
            assert backend.IMPLEMENTATIONS == tuple(
                c for c in definition.IMPLEMENTATIONS if c.provider == provider
            )
    for choice in choices:
        assert callable(getattr(import_module(choice.module_name), choice.factory))


def test_flattening_does_not_shadow_a_specialized_kernel_with_a_shared_binding():
    from akas.ops.linear import cuda, webgpu, kernel

    assert cuda.linear is not kernel.linear
    assert webgpu.linear is not kernel.linear
    assert cuda.shared_linear is kernel.linear
    assert webgpu.shared_linear is kernel.linear


def test_no_duplicate_factories_or_checkpoint_imports_in_operations():
    for path in (ROOT / "ops").rglob("*.py"):
        tree = ast.parse(path.read_text())
        names = [
            n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        ]
        assert len(names) == len(set(names)), path
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("akas.checkpoints"), path


def test_gguf_consumes_quantization_owned_encoding_definitions():
    from akas.checkpoints.gguf import TYPES
    from akas.quantization.weights.ggml.definition import ENCODINGS

    assert TYPES is ENCODINGS
