"""Registry definitions for kernels."""

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from akas.kernels.contracts import Implementation


def definition_modules():
    """Operations and codecs own their registrations at the same boundary."""
    root = Path(__file__).parent.parent
    operations = root / "ops"
    return tuple(
        (
            f"akas.ops.{path.parent.name}.definition"
            for path in sorted(operations.glob("*/definition.py"))
        )
    ) + tuple(
        (
            "akas." + ".".join(path.relative_to(root).with_suffix("").parts)
            for path in sorted(root.glob("quantization/*/*/definition.py"))
        )
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


@lru_cache(maxsize=1)
def planning_rules():
    """Discover compiler-free operation semantics and graph rewrites."""
    return tuple(import_module(name) for name in definition_modules())


def row_independent(call):
    return any(
        (
            call.kind in getattr(rules, "ROW_INDEPENDENT", ())
            for rules in planning_rules()
        )
    )


def rewrites():
    return tuple(
        (rule for rules in planning_rules() for rule in getattr(rules, "REWRITES", ()))
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
        raise ValueError(
            f"operation {kind!r} needs exactly one independent validation recipe"
        )
    return matches[0](kind, parameters, signature)


def graph_outputs(call, *, contract):
    """Return operation-declared outputs; unknown ABIs keep the full graph.

    Positional indexes and named ABI fields are owned by operation registrations,
    so compiler passes do not infer output locations from model or kernel names.
    """
    declarations = []
    for module in planning_rules():
        owned = getattr(module, "OUTPUT_ARGUMENTS", {})
        declarations.append(owned.get(contract, owned.get("*", {})).get(call.kind))
    declarations = [item for item in declarations if item is not None]
    if not declarations:
        return None
    if len(declarations) != 1:
        raise ValueError(f"operation {call.kind!r} has ambiguous output declarations")
    outputs = declarations[0]
    if callable(outputs):
        outputs = outputs(call)
    if outputs is None:
        return None
    from akas.kernels.contracts import OutputABI

    if isinstance(outputs, OutputABI):
        outputs = outputs.names if call.bindings else outputs.positions
    if call.bindings:
        if not all(
            (isinstance(name, str) and name in call.bindings for name in outputs)
        ):
            return None
        return tuple((call.bindings[name] for name in outputs))
    if not all(
        (
            type(index) is int and -len(call.arguments) <= index < len(call.arguments)
            for index in outputs
        )
    ):
        raise ValueError(
            f"operation {call.kind!r} does not match its declared output ABI"
        )
    return tuple((call.arguments[index] for index in outputs))
