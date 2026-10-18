"""Definition definitions for kernels."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module


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


@dataclass(frozen=True)
class Capabilities:
    provider: str = "cuda"
    target: str | None = None

    def __post_init__(self):
        if self.provider not in ("cuda", "webgpu"):
            raise ValueError(f"unsupported kernel provider {self.provider!r}")


def definition(spec, schedule=None, capabilities=None, *, preferred=None, catalog=None):
    """Resolve a definition without importing TileLang or a provider compiler."""
    from akas.kernels.registry import select

    spec = spec if isinstance(spec, KernelSpec) else KernelSpec(**spec)
    capabilities = (
        capabilities
        if isinstance(capabilities, Capabilities)
        else Capabilities(**capabilities or {})
    )
    schedule = dict(schedule or {})
    requested = schedule.pop("implementation", preferred)
    if preferred is not None and requested != preferred:
        raise ValueError("conflicting implementation choices")
    implementation = select(
        spec.kind,
        {**spec.parameters, **schedule},
        contract=spec.contract,
        provider=capabilities.provider,
        phase=spec.phase,
        target=capabilities.target,
        preferred=requested,
        catalog=catalog,
    )
    _check_schedule(
        spec.parameters,
        schedule,
        implementation.semantic_keys,
        tuple((name for name, _ in implementation.space)),
    )
    return (implementation, schedule)


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


def define(spec, schedule=None, capabilities=None, implementation=None):
    """Construct a kernel through the same API for CUDA and WebGPU.

    The compiler supplies a resolved, serializable implementation declaration.
    Direct callers may omit it and use the operation registry. Resolved choices
    keep exported sources independent of registry discovery and custom catalogs.
    """
    spec = spec if isinstance(spec, KernelSpec) else KernelSpec(**spec)
    capabilities = (
        capabilities
        if isinstance(capabilities, Capabilities)
        else Capabilities(**capabilities or {})
    )
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
        _check_schedule(
            spec.parameters,
            schedule,
            implementation["semantic_keys"],
            implementation["axes"],
        )
    from dataclasses import replace

    bound = replace(spec, parameters={**implementation["defaults"], **spec.parameters})
    factory = getattr(
        import_module(implementation["module"]), implementation["factory"]
    )
    prim = factory(bound, schedule)
    if (
        capabilities.provider == "webgpu"
        and hasattr(prim, "params")
        and hasattr(prim, "buffer_map")
    ):
        from tvm import tirx

        reserved = {"active", "target"}
        if any(
            str(prim.buffer_map[p].data.name) in reserved
            for p in prim.params
            if p in prim.buffer_map
        ):
            new_map, vmap, bmap = {}, {}, {}
            for p in prim.params:
                if p not in prim.buffer_map:
                    continue
                b = prim.buffer_map[p]
                if str(b.data.name) in reserved:
                    nd = tirx.Var(str(b.data.name) + "_buf", b.data.type_annotation)
                    nb = tirx.decl_buffer(
                        b.shape,
                        b.dtype,
                        name=str(b.name),
                        data=nd,
                        strides=b.strides,
                        elem_offset=b.elem_offset,
                        scope=b.scope(),
                    )
                    vmap[b.data] = nd
                    bmap[b] = nb
                    new_map[p] = nb
                else:
                    new_map[p] = b

            def _replace(op):
                if isinstance(op, tirx.BufferLoad) and op.buffer in bmap:
                    return tirx.BufferLoad(bmap[op.buffer], op.indices)
                if isinstance(op, tirx.BufferStore) and op.buffer in bmap:
                    return tirx.BufferStore(bmap[op.buffer], op.value, op.indices)
                return None

            body = tirx.stmt_functor.ir_transform(
                prim.body, None, _replace, ["tirx.BufferLoad", "tirx.BufferStore"]
            )
            body = tirx.stmt_functor.substitute(body, vmap)
            sig = getattr(prim, "__akas_signature__", None)
            prim = tirx.PrimFunc(
                prim.params, body, prim.ret_type, new_map, prim.attrs
            )
            if sig is not None:
                prim.__akas_signature__ = sig
    return prim
