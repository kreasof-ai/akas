"""Resolve operation-owned choices into serializable Tensor source declarations."""

from akas.artifacts.schema import Source
from akas.ops.registry import select


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
    implementation = select(
        kind,
        parameters,
        contract=contract,
        provider=provider,
        phase=phase,
        target=target,
        preferred=preferred,
        catalog=catalog,
    )
    options = {} if dependencies is None else {"dependencies": dependencies}
    return Source(
        implementation.module_name,
        "make_kernel",
        implementation.source_arguments(parameters),
        **options,
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
        variant = select(kind, {**parameters,**schedule}, contract=contract,provider=provider,
                         phase=phase,target=target,preferred=preferred,catalog=catalog)
        allowed = set(dict(variant.space)) if variant.semantic_keys is not None else set()
        if (set(schedule) & set(parameters)) - allowed:
            raise ValueError("schedule cannot override semantic parameters")
        selected = source(
            kind,
            {**parameters, **schedule},
            contract=contract,
            provider=provider,
            phase=phase,
            target=target,
            preferred=preferred,
            catalog=catalog,
            dependencies=dependencies,
        )
        recorded = (
            {**schedule, "implementation": variant.identifier}
            if record_implementation
            else schedule or None
        )
        result.append(Kernel(key, kind, parameters, selected, schedule=recorded))
    return result
