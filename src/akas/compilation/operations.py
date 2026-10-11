"""Resolve operation-owned choices into serializable Tensor source declarations."""

from akas.compilation.plan import Source
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
        "akas.ops." + implementation.module,
        "make_kernel",
        implementation.source_arguments(parameters),
        **options,
    )
