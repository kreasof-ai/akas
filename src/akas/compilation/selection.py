"""Resolve operation-owned choices into serializable Tensor source declarations."""

from akas.artifacts.schema import Source
from akas.kernels.definition import definition


def source(
    kind,
    parameters,
    *,
    contract,
    provider="cuda",
    phase="forward",
    target=None,
    preferred=None,
    catalog=None,
    dependencies=None,
):
    spec = dict(kind=kind, parameters=parameters, contract=contract, phase=phase)
    capabilities = dict(provider=provider, target=target)
    implementation, schedule = definition(
        spec, {}, capabilities, preferred=preferred, catalog=catalog
    )
    return selected_source(
        spec, schedule, capabilities, implementation, dependencies=dependencies
    )


def selected_source(spec, schedule, capabilities, implementation, *, dependencies=None):
    """Serialize the common API and hash only the selected leaf and its imports."""
    return Source(
        "akas.kernels.definition",
        "define",
        (spec, schedule, capabilities, implementation.declaration()),
        dependencies=tuple(
            sorted(
                {
                    "tensor.compiler.frontend.source",
                    implementation.module_name,
                    *(dependencies or ()),
                }
            )
        ),
    )


def kernels(
    requirements,
    *,
    contract,
    provider="cuda",
    phase="forward",
    target=None,
    schedules=None,
    catalog=None,
    dependencies=None,
    record_implementation=False,
):
    """Resolve one requirement inventory, preserving each layout's semantic ABI."""
    from akas.artifacts.schema import Kernel

    result = []
    for key, (kind, parameters) in requirements.items():
        schedule = (
            {}
            if schedules is None
            else dict(
                schedules.select(kind, parameters, provider=provider, target=target)
            )
        )
        preferred = schedule.pop("implementation", None)
        spec = dict(kind=kind, parameters=parameters, contract=contract, phase=phase)
        capabilities = dict(provider=provider, target=target)
        variant, schedule = definition(
            spec, schedule, capabilities, preferred=preferred, catalog=catalog
        )
        selected = selected_source(
            spec, schedule, capabilities, variant, dependencies=dependencies
        )
        recorded = (
            {**schedule, "implementation": variant.identifier}
            if record_implementation
            else schedule or None
        )
        result.append(Kernel(key, kind, parameters, selected, schedule=recorded))
    return result
