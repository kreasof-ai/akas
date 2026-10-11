"""Data describing kernel sources and a complete execution bundle."""

from dataclasses import dataclass, field
from importlib import import_module


@dataclass(frozen=True)
class Source:
    module: str
    factory: str
    args: tuple = ()
    dependencies: tuple[str, ...] = (
        "tensor.compiler.entry",
        "akas.kernels.elementwise",
    )
    export: bool = True

    def render(self):
        if not self.export:
            return getattr(import_module(self.module), self.factory)(*self.args)
        from tensor.compiler.entry import export_source

        return export_source(
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


@dataclass
class Bundle:
    manifest: dict
    filename: str
    kernels: list[Kernel] = field(default_factory=list)
    provider: str = "cuda"
    source_directory: str = ""
    artifact_directory: str = ""
    path_field: str = "path"

    @property
    def target(self):
        return self.manifest.get("target", "sm_89")
