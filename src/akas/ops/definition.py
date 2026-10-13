"""One kernel definition boundary for all providers, layouts and algorithms.

Models declare an operation and its semantic parameters. Schedules select an
implementation and its supported tuning axes; capabilities describe the target.
Leaf factories remain lazy and retain their existing arithmetic.
"""

from dataclasses import dataclass, field
from importlib import import_module
from collections.abc import Mapping


@dataclass(frozen=True)
class KernelSpec:
    kind: str
    parameters: Mapping = field(default_factory=dict)
    contract: str = "reference"
    phase: str = "forward"

    def __post_init__(self):
        if not self.kind or not self.contract or not self.phase:
            raise ValueError("kernel kind, contract and phase must be nonempty")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("kernel parameters must be a mapping")


@dataclass(frozen=True)
class Capabilities:
    provider: str = "cuda"
    target: str | None = None

    def __post_init__(self):
        if self.provider not in ("cuda", "webgpu"):
            raise ValueError(f"unsupported kernel provider {self.provider!r}")


def definition(spec, schedule=None, capabilities=None, *, preferred=None, catalog=None):
    """Resolve a definition without importing TileLang or a provider compiler."""
    from akas.ops.registry import select

    spec = spec if isinstance(spec, KernelSpec) else KernelSpec(**spec)
    capabilities = (capabilities if isinstance(capabilities, Capabilities)
                    else Capabilities(**(capabilities or {})))
    schedule = dict(schedule or {})
    requested = schedule.pop("implementation", preferred)
    if preferred is not None and requested != preferred:
        raise ValueError("conflicting implementation choices")
    implementation = select(
        spec.kind, {**spec.parameters, **schedule}, contract=spec.contract,
        provider=capabilities.provider, phase=spec.phase, target=capabilities.target,
        preferred=requested, catalog=catalog,
    )
    _check_schedule(spec.parameters, schedule, implementation.semantic_keys,
                    tuple(name for name, _ in implementation.space))
    return implementation, schedule


def _check_schedule(parameters, schedule, semantic_keys, axes):
    # Older registrations have not declared semantic keys. Such kernels may
    # accept schedule-only options, but a schedule may never overwrite their ABI.
    protected = set(parameters) if semantic_keys is None else set(semantic_keys)
    if set(schedule) & protected:
        raise ValueError("schedule cannot override semantic parameters")
    if (set(schedule) & set(parameters)) - set(axes):
        raise ValueError("schedule cannot override semantic parameters")
    unsupported = set(schedule) - set(axes)
    if unsupported:
        raise ValueError(f"schedule contains undeclared kernel axes: {sorted(unsupported)}")


def define(spec, schedule=None, capabilities=None, implementation=None):
    """Construct a kernel through the same API for CUDA and WebGPU.

    The compiler supplies a resolved, serializable implementation declaration.
    Direct callers may omit it and use the operation registry. Resolved choices
    keep exported sources independent of registry discovery and custom catalogs.
    """
    spec = spec if isinstance(spec, KernelSpec) else KernelSpec(**spec)
    capabilities = (capabilities if isinstance(capabilities, Capabilities)
                    else Capabilities(**(capabilities or {})))
    if implementation is None:
        selected, schedule = definition(spec, schedule, capabilities)
        implementation = selected.declaration()
    else:
        schedule = dict(schedule or {})
        requested = schedule.pop("implementation", implementation["identifier"])
        if requested != implementation["identifier"]:
            raise ValueError("resolved implementation differs from the schedule")
        if implementation["kind"] != spec.kind:
            raise ValueError("resolved implementation differs from the operation")
        if implementation["provider"] != capabilities.provider:
            raise ValueError("resolved implementation differs from the provider")
        if implementation["contract"] not in ("*", spec.contract):
            raise ValueError("resolved implementation differs from the buffer contract")
        if implementation["phase"] not in ("forward", spec.phase):
            raise ValueError("resolved implementation differs from the phase")
        targets = implementation["targets"]
        if targets and capabilities.target not in targets:
            raise ValueError("resolved implementation does not support the target")
        _check_schedule(spec.parameters, schedule, implementation["semantic_keys"],
                        implementation["axes"])
    parameters = {**implementation["defaults"], **spec.parameters, **schedule}
    factory = getattr(import_module(implementation["module"]), implementation["factory"])
    arguments = ((parameters,) if implementation["parameters_only"]
                 else (implementation["kernel_kind"], parameters))
    return factory(*arguments)
