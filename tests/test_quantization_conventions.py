"""Format packages declare storage locally and kernels share one factory contract."""

import inspect
import subprocess
import sys
from importlib import import_module
from types import SimpleNamespace
import pytest
from akas.memory.state import BufferSpec
from akas.quantization import Format, Storage, get


def test_new_activation_format_needs_only_its_own_declaration(monkeypatch):
    format = Format("example", lambda shape: Storage(BufferSpec(shape, "uint8")))
    monkeypatch.setitem(
        sys.modules,
        "akas.quantization.activations.example.definition",
        SimpleNamespace(FORMAT=format),
    )
    assert get("activations", "example") is format
    assert format.storage((2, 128)).values == BufferSpec((2, 128), "uint8")


def test_formats_expose_encoded_values_and_all_metadata_buffers():
    assert get("activations", "fp8").storage((3, 256)).metadata == (
        BufferSpec((3, 2), "float32"),
    )
    q16 = get("activations", "q16").storage((3, 64), provider="webgpu")
    assert q16.values == BufferSpec((96,), "uint32")
    assert q16.metadata == (BufferSpec((12,), "float32"), BufferSpec((12,), "int32"))
    assert get("weights", "ggml").storage(
        (64, 256), encoding=2, provider="webgpu"
    ).values == BufferSpec((2304,), "uint32")
    assert get("activations", "q16").storage((3, 64), provider="cuda") == q16


def test_every_registered_backend_uses_the_same_named_factory_signature():
    from akas.kernels.registry import implementations

    for declaration in implementations():
        factory = getattr(import_module(declaration.module_name), declaration.factory)
        assert tuple(inspect.signature(factory).parameters) == ("spec", "schedule"), (
            declaration
        )
        assert declaration.factory != "make_kernel"


def test_declarations_and_factory_modules_import_without_compiler_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "\nimport importlib, importlib.abc, sys\nclass Guard(importlib.abc.MetaPathFinder):\n    def find_spec(self, name, path=None, target=None):\n        if name.startswith('tensor.compiler') or name.split('.')[0] in {'tilelang', 'tvm', 'tvm_ffi', 'torch', 'triton'}:\n            raise ImportError(name)\nsys.meta_path.insert(0, Guard())\nfrom akas.kernels.registry import implementations\nfor declaration in implementations():\n    getattr(importlib.import_module(declaration.module_name), declaration.factory)\n",
        ],
        check=True,
    )
