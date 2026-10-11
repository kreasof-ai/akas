"""Source identities shared by bundle producers and compiler-free consumers."""

import hashlib
from importlib.util import find_spec
from pathlib import Path


def _modules(*owners):
    root = Path(__file__).resolve().parents[1]
    names = set()
    for owner in owners:
        path = root / owner.replace(".", "/")
        if path.is_dir():
            names.update(
                "akas." + str(p.relative_to(root).with_suffix("")).replace("/", ".")
                for p in path.rglob("*.py")
                if p.name != "__init__.py"
            )
        else:
            names.add("akas." + owner)
    return names


def _hashes(names):
    # Resolve source paths without importing operation factories or compilers.
    return {
        name: hashlib.sha256(Path(find_spec(name).origin).read_bytes()).hexdigest()
        for name in sorted(names)
    }


def backbone_hashes():
    return _hashes(
        _modules(
            "artifacts.kernels",
            "artifacts.fingerprints",
            "execution.plan",
            "tensor",
            "memory",
            "models.blocks",
            "ops",
            "profiles",
            "checkpoints.upload",
        )
    )


def _contract_hashes(contract, *, provider="cuda"):
    names = _modules(
        "execution." + contract, "models", "checkpoints", "generation", "text"
    )
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
    if provider == "cuda":
        names.update(
            "tensor.providers.cuda." + name for name in ("device", "driver", "graph")
        )
    elif provider == "webgpu":
        names.update(
            "tensor.providers.webgpu." + name
            for name in ("device", "adapter", "buffer", "plan")
        )
        names.add("tensor.artifacts.webgpu")
    else:
        raise ValueError("unsupported fingerprint provider")
    result = backbone_hashes() | _hashes(names)
    if provider == "webgpu":
        path = Path(find_spec("tensor").origin).parent / "native/webgpu_plan.c"
        result["tensor.native.webgpu_plan"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


def independent_hashes(provider="cuda"):
    return _contract_hashes("independent", provider=provider)


def fixed_slots_hashes():
    return _contract_hashes("fixed_slots")


def compact_batch_hashes():
    return _contract_hashes("compact_batch")


def validate_implementation(manifest, expected):
    if manifest.get("implementation") != expected:
        raise ValueError("dense implementation changed; regenerate the AOT bundle")
