"""Registry definitions for kernels."""

from functools import lru_cache
from importlib import import_module
from akas.ops.planning import (
    definition_modules,
)


@lru_cache(maxsize=1)
def implementations():
    """Discover local operation registrations; kernel modules remain unloaded."""
    return tuple(
        (
            item
            for name in definition_modules()
            for item in getattr(import_module(name), "IMPLEMENTATIONS", ())
        )
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
        (
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
    )


def select(kind, parameters, *, preferred=None, **options):
    eligible = candidates(kind, parameters, **options)
    if preferred is not None:
        eligible = tuple((item for item in eligible if item.identifier == preferred))
    if not eligible:
        raise ValueError(f"no compatible implementation for {kind}: {options}")
    priority = max((item.priority for item in eligible))
    winners = [item for item in eligible if item.priority == priority]
    if len(winners) != 1:
        raise ValueError(
            f"ambiguous implementations for {kind}: {[item.module for item in winners]}"
        )
    return winners[0]
