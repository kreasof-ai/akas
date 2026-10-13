"""Lazy selection of compatible operation implementations.

Contracts identify buffer ABIs, not model names. Operation-owned registrations
can qualify a variant by target and geometry without importing its compiler.
"""

from akas.ops.contracts import Implementation
from functools import lru_cache
from importlib import import_module
from pathlib import Path



def variants(kinds, module, contract, **options):
    return tuple(Implementation(kind, module, contract, **options) for kind in kinds)


def definition_modules():
    """Operations and codecs own their registrations at the same boundary."""
    operations = Path(__file__).parent
    root = operations.parent
    return tuple(f"akas.ops.{path.parent.name}.operation" for path in sorted(operations.glob("*/operation.py"))) + tuple(
        "akas." + ".".join(path.relative_to(root).with_suffix("").parts)
        for path in sorted(root.glob("quantization/*/*.py")) if path.name != "__init__.py")


@lru_cache(maxsize=1)
def implementations():
    """Discover local operation registrations; kernel modules remain unloaded."""
    return tuple(
        item
        for name in definition_modules()
        for item in getattr(import_module(name), "IMPLEMENTATIONS", ())
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
        eligible = tuple(item for item in eligible if item.identifier == preferred)
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


@lru_cache(maxsize=1)
def planning_rules():
    """Discover compiler-free operation semantics and graph rewrites."""
    return tuple(
        import_module(name) for name in definition_modules()
    )


def row_independent(call):
    return any(
        call.kind in getattr(rules, "ROW_INDEPENDENT", ()) for rules in planning_rules()
    )


def rewrites():
    return tuple(
        rule for rules in planning_rules() for rule in getattr(rules, "REWRITES", ())
    )


def validation(kind, parameters, signature):
    """Resolve the numerical recipe by operation, never by model family."""
    matches = []
    for name in definition_modules():
        module = import_module(name)
        recipe = getattr(module, "VALIDATIONS", {}).get(kind)
        if recipe is not None:
            matches.append(recipe)
    if len(matches) != 1:
        raise ValueError(f"operation {kind!r} needs exactly one independent validation recipe")
    return matches[0](kind, parameters, signature)
