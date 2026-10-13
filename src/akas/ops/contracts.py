"""Kernel eligibility and independent validation data shared by all operations."""

from dataclasses import dataclass, field
from importlib import import_module
from typing import Callable

@dataclass(frozen=True)
class Implementation:
    kind: str
    module: str
    contract: str
    provider: str = "cuda"
    phase: str = "forward"
    priority: int = 10
    targets: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    accepts: Callable[[dict], bool] | None = None
    kernel_kind: str | None = None
    defaults: tuple[tuple[str, object], ...] = ()
    # Axes are declared only where the kernel actually supports these choices.
    space: tuple[tuple[str, tuple], ...] = ()
    semantic_keys: tuple[str, ...] | None = None
    variant: str | None = None
    factory: str = "make_kernel"
    parameters_only: bool = False

    @property
    def identifier(self):
        return self.module if self.variant is None else f"{self.module}:{self.variant}"

    @property
    def module_name(self):
        return self.module if self.module.startswith("akas.") else "akas.ops." + self.module

    def compatible(self, kind, parameters, *, contract, provider, phase, target):
        return (
            (self.kind, self.provider) == (kind, provider)
            and self.contract in ("*", contract)
            and self.phase in ("forward", phase)
            and (not self.targets or target in self.targets)
            and all(name in parameters for name in self.required)
            and (self.accepts is None or self.accepts(parameters))
        )

    def source_arguments(self, parameters):
        options = {**dict(self.defaults), **parameters}
        return (options,) if self.parameters_only else (self.kernel_kind or self.kind, options)

    def declaration(self):
        """Serializable routing; the exported source needs no registry discovery."""
        return dict(identifier=self.identifier, module=self.module_name, kind=self.kind,
                    kernel_kind=self.kernel_kind or self.kind, contract=self.contract,
                    provider=self.provider, phase=self.phase, targets=self.targets,
                    defaults=dict(self.defaults), semantic_keys=self.semantic_keys,
                    axes=tuple(name for name, _ in self.space),
                    factory=self.factory, parameters_only=self.parameters_only)

    def make(self, parameters):
        return getattr(import_module(self.module_name), self.factory)(
            *self.source_arguments(parameters)
        )



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
    import numpy as np
    from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16

    rng = np.random.default_rng(seed)
    result = {}
    for spec in signature:
        if spec.get("kind", "buffer") != "buffer":
            raise ValueError("validation requires an operation scalar input recipe")
        shape, dtype = tuple(spec["shape"]), spec["dtype"]
        if any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("validation requires resolved positive buffer shapes")
        if dtype == "bfloat16":
            values = rng.normal(0, 0.15, shape).astype("float32")
            result[spec["name"]] = decode_bfloat16(encode_bfloat16(values))
        elif np.dtype(dtype).kind in "iu":
            result[spec["name"]] = rng.integers(3, 10, shape, dtype=dtype)
        else:
            result[spec["name"]] = rng.normal(0, 0.15, shape).astype(dtype)
    return result


def mapped_validation(kind, parameters, signature, mapping):
    """Validate a lifted kernel using its original operation's reference.

    Gather local state into reversed, nonadjacent pool slots and scatter the
    reference result back. Unselected slots remain exact sentinels, so the
    qualification checks both arithmetic and request isolation.
    """
    import numpy as np
    from akas.ops.registry import validation

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
    arguments = sample_arguments(signature)
    expected = {name: value.copy() for name, value in arguments.items()}
    slots = np.linspace(0, capacity - 1, rows, dtype="int32")[::-1].copy()
    arguments["mapping"][:] = slots
    expected["mapping"][:] = slots
    for name in local.arguments:
        if name in names:
            initial = arguments[name].reshape(capacity, -1)
            result = expected[name].reshape(capacity, -1)
            initial[slots] = local.arguments[name].reshape(rows, -1)
            result[slots] = local.expected[name].reshape(rows, -1)
        else:
            arguments[name] = local.arguments[name]
            expected[name] = local.expected[name]
    return ValidationCase(arguments, expected, local.rtol, local.atol, local.comparators)
