"""Lazy selection of compatible operation implementations.

Contracts identify buffer ABIs, not model names. Operation-owned registrations
can qualify a variant by target and geometry without importing its compiler.
"""

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Implementation:
    kind: str
    module: str
    contract: str
    provider: str = "cuda"
    phase: str = "forward"
    priority: int = 10
    targets: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    accepts: Callable[[dict], bool] | None = None
    kernel_kind: str | None = None
    defaults: tuple[tuple[str, object], ...] = ()
    # Axes are declared only where the kernel actually supports these choices.
    space: tuple[tuple[str, tuple], ...] = ()

    def compatible(self, kind, parameters, *, contract, provider, phase, target):
        return (
            (self.kind, self.contract, self.provider, self.phase)
            == (kind, contract, provider, phase)
            and (not self.targets or target in self.targets)
            and all(name in parameters for name in self.required)
            and (self.accepts is None or self.accepts(parameters))
        )

    def source_arguments(self, parameters):
        return self.kernel_kind or self.kind, {**dict(self.defaults), **parameters}

    def make(self, parameters):
        return import_module("akas.ops." + self.module).make_kernel(
            *self.source_arguments(parameters)
        )


def variants(kinds, module, contract, **options):
    return tuple(Implementation(kind, module, contract, **options) for kind in kinds)


@lru_cache(maxsize=1)
def implementations():
    """Discover local operation registrations; kernel modules remain unloaded."""
    root = Path(__file__).parent
    return tuple(
        item
        for path in sorted(root.glob("*/implementations.py"))
        for item in import_module(
            f"akas.ops.{path.parent.name}.implementations"
        ).IMPLEMENTATIONS
    )


def candidates(
    kind,
    parameters,
    *,
    contract,
    provider="cuda",
    phase="forward",
    target=None,
    catalog=None,
):
    catalog = implementations() if catalog is None else catalog
    return tuple(
        item
        for item in catalog
        if item.compatible(
            kind,
            parameters,
            contract=contract,
            provider=provider,
            phase=phase,
            target=target,
        )
    )


def select(kind, parameters, *, preferred=None, **options):
    eligible = candidates(kind, parameters, **options)
    if preferred is not None:
        eligible = tuple(item for item in eligible if item.module == preferred)
    if not eligible:
        raise ValueError(f"no compatible implementation for {kind}: {options}")
    priority = max(item.priority for item in eligible)
    winners = [item for item in eligible if item.priority == priority]
    if len(winners) != 1:
        raise ValueError(
            f"ambiguous implementations for {kind}: {[item.module for item in winners]}"
        )
    return winners[0]


def make_kernel(kind, parameters, **options):
    return select(kind, parameters, **options).make(parameters)
