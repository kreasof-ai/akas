"""Compile composed models through operation contracts, independent of model names."""

from akas.compilation.plan import Bundle, Kernel
from akas.compilation.operations import source


def plan(
    forward,
    *,
    provider="cuda",
    target=None,
    filename="forward.json",
    schedules=None,
    catalog=None,
):
    target = target or ("sm_89" if provider == "cuda" else "webgpu-portable-v1")
    kernels = []
    for key, (kind, parameters) in forward.requirements().items():
        schedule = (
            {}
            if schedules is None
            else schedules.select(kind, parameters, provider=provider, target=target)
        )
        schedule = dict(schedule)
        preferred = schedule.pop("implementation", None)
        if set(schedule) & set(parameters):
            raise ValueError("schedule cannot override semantic parameters")
        configured = {**parameters, **schedule}
        selected = source(
            kind,
            configured,
            contract=forward.contract,
            provider=provider,
            phase=forward.phase,
            target=target,
            catalog=catalog,
            preferred=preferred,
        )
        kernels.append(
            Kernel(
                key,
                kind,
                parameters,
                selected,
                schedule={
                    **schedule,
                    "implementation": selected.module.removeprefix("akas.ops."),
                },
            )
        )
    return Bundle(
        dict(
            schema="akas.forward.v1",
            target=target,
            provider=provider,
            forward=forward.metadata(),
        ),
        filename,
        kernels,
        provider=provider,
    )


def produce(forward, out, **options):
    from akas.compilation.build import build

    return build(plan(forward, **options), out)
