"""Data describing kernel sources and a complete execution bundle."""

from dataclasses import dataclass, field
from importlib import import_module
import hashlib
import json


@dataclass(frozen=True)
class Source:
    module: str
    factory: str
    args: tuple = ()
    dependencies: tuple[str, ...] = (
        "tensor.compiler.frontend.source",
        "akas.ops.normalization.kernel",
    )
    export: bool = True

    def render(self):
        if not self.export:
            return getattr(import_module(self.module), self.factory)(*self.args)
        from akas.compilation.emit import export_kernel

        return export_kernel(
            self.module, self.factory, *self.args, dependencies=self.dependencies
        )


@dataclass(frozen=True)
class Kernel:
    key: str
    kind: str
    parameters: dict
    source: Source | str
    artifact: str | None = None
    schedule: dict | None = None
    target: str | None = None
    mapping: tuple | None = None


@dataclass
class Bundle:
    manifest: dict
    filename: str
    kernels: list[Kernel] = field(default_factory=list)
    provider: str = "cuda"
    source_directory: str = ""
    artifact_directory: str = ""
    path_field: str = "path"
    catalog: tuple | None = None

    @property
    def target(self):
        return self.manifest.get("target", "sm_89")


def identity(kind, parameters):
    return hashlib.sha256(
        json.dumps([kind, parameters], sort_keys=True).encode()
    ).hexdigest()[:24]
