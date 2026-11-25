"""Source identities shared by bundle producers and compiler-free consumers."""

import hashlib
from importlib.util import find_spec
from akas.benchmarking.provenance import source_hashes
from pathlib import Path


def _modules(*owners):
    root = Path(__file__).resolve().parents[1]
    names = set()
    for owner in owners:
        path = root / owner.replace(".", "/")
        if path.is_dir():
            names.update(
                "akas."
                + p.relative_to(root).with_suffix("").as_posix().replace("/", ".")
                for p in path.rglob("*.py")
                if p.name != "__init__.py"
                and all(
                    part.isidentifier()
                    for part in p.relative_to(root).with_suffix("").parts
                )
            )
        else:
            names.add("akas." + owner)
    return names


def backbone_hashes():
    return source_hashes(
        _modules(
            "artifacts.read",
            "artifacts.schema",
            "artifacts.fingerprint",
            "execution",
            "graph",
            "tensor",
            "memory",
            "models.blocks",
            "ops",
            "kernels",
            "quantization",
            "validation",
            "checkpoints.upload",
        )
    )


def runtime_hashes(provider="cuda"):
    names = _modules("models", "checkpoints", "generation", "text")
    names.update(
        "tensor.runtime." + name
        for name in (
            "abi",
            "signature",
            "buffer",
            "executable",
            "session",
            "event",
            "errors",
            "dtypes",
            "dlpack",
        )
    )
    names.update(("tensor.artifacts.format", "tensor.providers." + provider))
    if find_spec("tensor.runtime.tensor_map") is not None:
        names.add("tensor.runtime.tensor_map")
    if find_spec("tensor.runtime.domain") is not None:
        names.add("tensor.runtime.domain")
    if provider == "cuda":
        names.update(
            "tensor.providers.cuda." + name for name in ("device", "driver", "graph")
        )
        if find_spec("tensor.providers.cuda.tensor_map") is not None:
            names.add("tensor.providers.cuda.tensor_map")
    elif provider == "webgpu":
        names.update(
            "tensor.providers.webgpu." + name
            for name in ("device", "adapter", "buffer", "plan")
        )
        names.add("tensor.artifacts.webgpu")
    else:
        raise ValueError("unsupported fingerprint provider")
    result = backbone_hashes() | source_hashes(names)
    if provider == "webgpu":
        path = Path(find_spec("tensor").origin).parent / "native/webgpu_plan.c"
        result["tensor.native.webgpu_plan"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


def validate_implementation(manifest, expected):
    if manifest.get("implementation") != expected:
        raise ValueError("program implementation changed; regenerate the AOT bundle")


def compiler_identity(provider):
    from akas.compilation.source import compiler_identity as compiler

    return compiler(provider)


_compiler_identity = compiler_identity


_hashes = source_hashes
