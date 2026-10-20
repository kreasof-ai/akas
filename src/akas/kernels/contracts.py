"""Contracts definitions for kernels."""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable


@dataclass(frozen=True)
class Implementation:
    """One lazy named kernel factory and the conditions under which it is valid.

    Factories always accept (KernelSpec, schedule). Registration does not import
    the factory or its compiler. Names are stable schedule-profile identities.
    """

    kind: str
    module: str
    contract: str
    factory: str = field(kw_only=True)
    provider: str = "cuda"
    phase: str = "forward"
    priority: int = 10
    targets: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    accepts: Callable[[dict], bool] | None = None
    defaults: tuple[tuple[str, object], ...] = ()
    space: tuple[tuple[str, tuple], ...] = ()
    semantic_keys: tuple[str, ...] | None = None
    reference_keys: tuple[str, ...] = ()
    variant: str | None = None
    name: str | None = None

    @property
    def identifier(self):
        if self.name is not None:
            return self.name
        symbol = f"{self.module_name}:{self.factory}"
        return symbol if self.variant is None else f"{symbol}:{self.variant}"

    @property
    def module_name(self):
        return (
            self.module
            if self.module.startswith("akas.")
            else "akas.ops." + self.module
        )

    def compatible(self, kind, parameters, *, contract, provider, phase, target):
        return (
            (self.kind, self.provider) == (kind, provider)
            and self.contract in ("*", contract)
            and (self.phase in ("forward", phase))
            and (not self.targets or target in self.targets)
            and all((name in parameters for name in self.required))
            and (self.accepts is None or self.accepts(parameters))
        )

    def declaration(self):
        """Serializable routing; exported sources need no registry discovery."""
        return dict(
            identifier=self.identifier,
            module=self.module_name,
            factory=self.factory,
            kind=self.kind,
            contract=self.contract,
            provider=self.provider,
            phase=self.phase,
            targets=self.targets,
            defaults=dict(self.defaults),
            semantic_keys=self.semantic_keys,
            axes=tuple((name for name, _ in self.space)),
        )

    def make(self, parameters, schedule=None):
        from akas.kernels.definition import KernelSpec, _check_schedule

        schedule = dict(schedule or {})
        _check_schedule(
            parameters,
            schedule,
            self.semantic_keys,
            tuple((name for name, _ in self.space)),
        )
        spec = KernelSpec(
            self.kind, {**dict(self.defaults), **parameters}, self.contract, self.phase
        )
        return getattr(import_module(self.module_name), self.factory)(spec, schedule)


@dataclass(frozen=True)
class OutputABI:
    """Potential writes for ordered and named graph calls of an operation."""

    positions: tuple[int, ...]
    names: tuple[str, ...]


@dataclass(frozen=True)
class ValidationCase:
    """Initial argument arrays and expected outputs/state, independent of a kernel.

    Every written buffer must have an expectation; immutable arguments are also
    checked to detect accidental writes. References use the actual input storage
    precision. No model-specific tuning callback is needed.
    """

    arguments: dict
    expected: dict
    rtol: float = 0.02
    atol: float = 0.002
    comparators: dict = field(default_factory=dict)


def sample_arguments(signature, *, seed=17):
    """Deterministic finite inputs; operation recipes constrain indexes/masks."""
    import math
    import numpy as np
    from tensor.runtime.dtypes import decode_bfloat16, encode_bfloat16

    rng = np.random.default_rng(seed)
    result = {}
    for spec in signature:
        if spec.get("kind", "buffer") != "buffer":
            raise ValueError("validation requires an operation scalar input recipe")
        shape, dtype = (tuple(spec["shape"]), spec["dtype"])
        if any((type(n) is not int or n <= 0 for n in shape)):
            raise ValueError("validation requires resolved positive buffer shapes")
        total = math.prod(shape)
        period = 1048573
        if dtype == "bfloat16":
            if total <= period:
                values = rng.normal(0, 0.15, shape).astype("float32")
                result[spec["name"]] = decode_bfloat16(encode_bfloat16(values))
            else:
                tile = decode_bfloat16(
                    encode_bfloat16(rng.normal(0, 0.15, period).astype("float32"))
                )
                result[spec["name"]] = np.resize(tile, shape)
        elif np.dtype(dtype).kind in "iu":
            if total <= period:
                result[spec["name"]] = rng.integers(3, 10, shape, dtype=dtype)
            else:
                tile = rng.integers(3, 10, period, dtype=dtype)
                result[spec["name"]] = np.resize(tile, shape)
        elif total <= period:
            result[spec["name"]] = rng.normal(0, 0.15, shape).astype(dtype)
        else:
            tile = rng.normal(0, 0.15, period).astype(dtype)
            result[spec["name"]] = np.resize(tile, shape)
    return result


def mapped_validation(kind, parameters, signature, mapping):
    """Validate a lifted kernel using its original operation's reference.

    Gather local state into reversed, nonadjacent pool slots and scatter the
    reference result back. Unselected slots remain exact sentinels, so the
    qualification checks both arithmetic and request isolation.
    """
    import numpy as np
    from akas.kernels.registry import validation

    indexes, rows, capacity = mapping
    original = [dict(spec) for spec in signature if spec["name"] != "mapping"]
    names = []
    for index in indexes:
        spec = original[index]
        shape = list(spec["shape"])
        shape[0] = shape[0] // capacity * rows
        spec["shape"] = shape
        names.append(spec["name"])
    local = validation(kind, parameters, original)
    written_local = {
        name
        for name in local.expected
        if local.arguments[name] is not local.expected[name]
        and not np.array_equal(local.arguments[name], local.expected[name])
    }
    arguments = sample_arguments(
        [
            spec
            for spec in signature
            if spec["name"] in names or spec["name"] == "mapping"
        ]
    )
    expected = {
        name: value.copy() if name in written_local else value
        for name, value in arguments.items()
    }
    slots = np.linspace(0, capacity - 1, rows, dtype="int32")[::-1].copy()
    arguments["mapping"][:] = slots
    for name in list(local.arguments):
        if name in names:
            initial = arguments[name].reshape(capacity, -1)
            initial[slots] = local.arguments[name].reshape(rows, -1)
            if name in written_local:
                result = expected[name].reshape(capacity, -1)
                result[slots] = local.expected[name].reshape(rows, -1)
        else:
            arguments[name] = local.arguments[name]
            expected[name] = local.expected[name]
        local.arguments.pop(name, None)
        local.expected.pop(name, None)
    return ValidationCase(
        arguments, expected, local.rtol, local.atol, local.comparators
    )
