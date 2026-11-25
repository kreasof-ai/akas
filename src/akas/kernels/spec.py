"""Compiler-free kernel inputs and implementation descriptors."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable


@dataclass(frozen=True)
class KernelSpec:
    kind: str
    parameters: Mapping = field(default_factory=dict)
    contract: str = "reference"
    phase: str = "forward"

    def __post_init__(self):
        if not self.kind or not self.contract or (not self.phase):
            raise ValueError("kernel kind, contract and phase must be nonempty")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("kernel parameters must be a mapping")

    def parameter(self, name):
        return self.parameters[name]

    def with_parameters(self, **updates):
        from dataclasses import replace

        return replace(self, parameters={**self.parameters, **updates})


@dataclass(frozen=True)
class Capabilities:
    provider: str = "cuda"
    target: str | None = None

    def __post_init__(self):
        if self.provider not in ("cuda", "webgpu"):
            raise ValueError(f"unsupported kernel provider {self.provider!r}")


@dataclass(frozen=True)
class Implementation:
    """One lazy named kernel factory and the conditions under which it is valid.

    Factories always accept (KernelSpec, schedule). Registration does not import
    the factory or its compiler. Names are stable schedule-profile identities.
    """

    kind: str
    module: str
    contract: str
    factory: str = field(kw_only=True)
    provider: str = "cuda"
    phase: str = "forward"
    priority: int = 10
    targets: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    accepts: Callable[[dict], bool] | None = None
    defaults: tuple[tuple[str, object], ...] = ()
    space: tuple[tuple[str, tuple], ...] = ()
    semantic_keys: tuple[str, ...] | None = None
    reference_keys: tuple[str, ...] = ()
    variant: str | None = None
    name: str | None = None

    @property
    def identifier(self):
        if self.name is not None:
            return self.name
        symbol = f"{self.module_name}:{self.factory}"
        return symbol if self.variant is None else f"{symbol}:{self.variant}"

    @property
    def module_name(self):
        return (
            self.module
            if self.module.startswith("akas.")
            else "akas.ops." + self.module
        )

    def compatible(self, kind, parameters, *, contract, provider, phase, target):
        return (
            (self.kind, self.provider) == (kind, provider)
            and self.contract in ("*", contract)
            and (self.phase in ("forward", phase))
            and (not self.targets or target in self.targets)
            and all((name in parameters for name in self.required))
            and (self.accepts is None or self.accepts(parameters))
        )

    def declaration(self):
        """Serializable routing; exported sources need no registry discovery."""
        return dict(
            identifier=self.identifier,
            module=self.module_name,
            factory=self.factory,
            kind=self.kind,
            contract=self.contract,
            provider=self.provider,
            phase=self.phase,
            targets=self.targets,
            defaults=dict(self.defaults),
            semantic_keys=self.semantic_keys,
            axes=tuple((name for name, _ in self.space)),
        )

    def make(self, parameters, schedule=None):
        schedule = dict(schedule or {})
        _check_schedule(
            parameters,
            schedule,
            self.semantic_keys,
            tuple((name for name, _ in self.space)),
        )
        spec = KernelSpec(
            self.kind, {**dict(self.defaults), **parameters}, self.contract, self.phase
        )
        return getattr(import_module(self.module_name), self.factory)(spec, schedule)


@dataclass(frozen=True)
class OutputABI:
    """Potential writes for ordered and named graph calls of an operation."""

    positions: tuple[int, ...]
    names: tuple[str, ...]


def _check_schedule(parameters, schedule, semantic_keys, axes):
    protected = set(parameters) if semantic_keys is None else set(semantic_keys)
    if set(schedule) & protected:
        raise ValueError("schedule cannot override semantic parameters")
    if (set(schedule) & set(parameters)) - set(axes):
        raise ValueError("schedule cannot override semantic parameters")
    unsupported = set(schedule) - set(axes)
    if unsupported:
        raise ValueError(
            f"schedule contains undeclared kernel axes: {sorted(unsupported)}"
        )
