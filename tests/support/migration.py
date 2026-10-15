"""Resolve frozen pre-refactor test factories to their current named kernels.

Historical arithmetic fixtures and numerical probes keep their original inputs.
Production code uses named factories directly and does not use this adapter.
"""

import json
from importlib import import_module
from pathlib import Path

_RECORDS = json.loads(
    (Path(__file__).parents[1] / "fixtures/kernel-migration.json").read_text()
)
_ROUTES = {(r["old_module"], r["old_factory"], r["kind"]): r for r in _RECORDS}


def resolve(module, name, args):
    route = _ROUTES.get((module, name, None))
    if route is None and args and isinstance(args[0], str):
        route = _ROUTES.get((module, name, args[0]))
    return route


def build(module, name, kind, parameters):
    from akas.kernels.definition import KernelSpec

    return getattr(import_module(module), name)(KernelSpec(kind, parameters), {})


def legacy_arguments(route, args):
    """Translate frozen helper calls; current computation builders use flags."""
    module, name = (
        route.get("previous_module", route["module"]),
        route.get("previous_factory", route["factory"]),
    )
    if not args or not isinstance(args[0], str):
        return args
    kind, *rest = args
    if module.startswith("akas.ops.linear.") and name in (
        "_packed",
        "_fused",
        "_integer",
        "cuda_abi",
        "webgpu_abi",
        "_epilogue",
    ):
        return (*rest, kind in ("ffn", "ffn_q16"), kind == "linear_add")
    if module == "akas.ops.normalization.kernel" and name in ("_parallel", "_serial"):
        return (*rest, kind == "add_rms") if name == "_parallel" else tuple(rest)
    if module == "akas.ops.attention.kernel" and name == "_normalization":
        return (*rest, kind == "qnorm")
    return args


def factory(module, name, *args, **kwargs):
    if (
        module == "akas.ops.moe.pipeline"
        and name == "make_kernel"
        and (args[0] == "expert_partial")
    ):
        from akas.kernels.registry import select

        p = args[1]
        return select(
            "expert_partial",
            p,
            contract="block-fp8",
            target="sm_90a" if p.get("hopper") else "sm_89",
        ).make(p)
    route = resolve(module, name, args)
    if route is None:
        return getattr(import_module(module), name)(*args, **kwargs)
    if route["uniform"]:
        parameters = args[0] if route["kind"] is None else args[1]
        return build(
            route["module"], route["factory"], route["kind"] or name, parameters
        )
    return getattr(import_module(route["module"]), route["factory"])(
        *legacy_arguments(route, args), **kwargs
    )


def source(module, name, *args, dependencies=()):
    from tensor.compiler.frontend.source import export_source

    if (
        module == "akas.ops.moe.pipeline"
        and name == "make_kernel"
        and (args[0] == "expert_partial")
    ):
        from akas.compilation.selection import source as selected

        p = args[1]
        return selected(
            "expert_partial",
            p,
            contract="block-fp8",
            target="sm_90a" if p.get("hopper") else "sm_89",
        ).render()
    migrated = set()
    for dependency in dependencies:
        choices = {r["module"] for r in _RECORDS if r["old_module"] == dependency}
        migrated.update(choices or (dependency,))
    if module == "akas.compilation.transforms.state" and name == "make_kernel":
        old_module, old_name, arguments, *rest = args
        route = resolve(old_module, old_name, arguments)
        if route and route["uniform"]:
            parameters = arguments[0] if route["kind"] is None else arguments[1]
            inner = (
                route["module"],
                route["factory"],
                route["kind"] or old_name,
                parameters,
            )
            return export_source(
                module,
                name,
                __name__,
                "build",
                inner,
                *rest,
                dependencies=tuple(sorted(migrated | {route["module"]})),
            )
    route = resolve(module, name, args)
    if route:
        migrated.add(route["module"])
        if route["uniform"]:
            parameters = args[0] if route["kind"] is None else args[1]
            return export_source(
                __name__,
                "build",
                route["module"],
                route["factory"],
                route["kind"] or name,
                parameters,
                dependencies=tuple(sorted(migrated)),
            )
        args = legacy_arguments(route, args)
        module, name = (
            route["module"],
            route["factory"],
        )
    return export_source(module, name, *args, dependencies=tuple(sorted(migrated)))


def historical_source(choice, parameters):
    """Compare frozen source inventories independently of a factory's new home."""
    if not isinstance(choice, dict):
        choice = choice.declaration()
    identity = choice["identifier"].split(":", 1)[0]
    module = identity if identity.startswith("akas.") else "akas.ops." + identity
    candidates = [
        r
        for r in _RECORDS
        if r["old_module"] == module
        and r["module"] == choice["module"]
        and (r["factory"] == choice["factory"])
    ]
    exact = [r for r in candidates if r["kind"] == choice["kind"]]
    route = (exact or candidates)[0]
    values = {**choice["defaults"], **parameters}
    args = [values] if route["kind"] is None else [route["kind"], values]
    return dict(module=module, factory=route["old_factory"], args=args)
