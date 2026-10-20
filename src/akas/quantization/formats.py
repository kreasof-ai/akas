"""Compiler-free storage contracts shared by activation, weight and KV codecs."""

from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from akas.memory.state import BufferSpec


@dataclass(frozen=True)
class Storage:
    values: BufferSpec
    metadata: tuple[BufferSpec, ...] = ()
    metadata_names: tuple[str, ...] = ()

    def __post_init__(self):
        if (
            len(self.metadata_names) != len(self.metadata)
            or len(set(self.metadata_names)) != len(self.metadata_names)
            or "values" in self.metadata_names
        ):
            raise ValueError("metadata names must uniquely name every metadata buffer")

    @property
    def buffers(self):
        """Physical argument order: payload, followed by its declared metadata."""
        return (self.values, *self.metadata)

    @property
    def named(self):
        return dict(zip(("values", *self.metadata_names), self.buffers))


@dataclass(frozen=True)
class Codec:
    """Compile-time binding; inline functions receive an encoded-buffer tuple.

    ``read(encoded, *coordinates)`` returns F32. Matrix ``read_tile`` fills a
    caller-owned tile; its element type selects staging precision. Specialized
    capabilities keep their documented operation-specific argument contracts.
    ``dot_half(encoded, x, offset, tile, lane, row, previous)`` carries an F32
    lane accumulator through FP16 operand products; tile sums must not reset it.
    ``dot_float(encoded, x, offset, tile, lane, row, previous)`` adds a packed
    256-channel F32 lane product to ``previous`` without rounding operands to
    FP16. The consumer owns the warp reduction; tile grouping can change F32
    addition order and must pass an independent numerical reference.
    No codec dispatch or container allocation survives macro expansion.
    """

    storage: Storage
    read: Callable
    read_tile: Callable | None = None
    read_pair: Callable | None = None
    dot: Callable | None = None
    dot_half: Callable | None = None
    dot_float: Callable | None = None
    write: Callable | None = None
    write_pair: Callable | None = None
    read_packed: Callable | None = None
    packed_dtype: str | None = None
    read_fields: tuple[str, ...] | None = None

    @property
    def read_buffers(self):
        """Only buffers consumed by scalar/tile access, excluding dot auxiliaries."""
        if self.read_fields is None:
            return self.storage.buffers
        return tuple(self.storage.named[name] for name in self.read_fields)

    def require(self, capability):
        if capability not in (
            "read",
            "read_tile",
            "read_pair",
            "dot",
            "dot_half",
            "dot_float",
            "write",
            "write_pair",
            "read_packed",
        ):
            raise ValueError(f"unknown inline codec capability: {capability}")
        function = getattr(self, capability)
        if function is None:
            raise ValueError(f"codec binding does not support {capability}")
        return function


@dataclass(frozen=True)
class Format:
    name: str
    layout: Callable[[tuple[int, ...]], Storage]
    providers: tuple[tuple[str, str], ...] = ()

    layouts: tuple[tuple[str, Callable], ...] = ()

    def storage(self, shape, *, provider=None, **options):
        if (
            provider is not None
            and self.providers
            and provider not in dict(self.providers)
        ):
            raise ValueError(f"codec {self.name} does not support {provider}")
        shape = tuple(shape)
        if len(shape) < 2 or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError(
                "quantization storage requires positive dimensions including channels"
            )
        return dict(self.layouts).get(provider, self.layout)(shape, **options)

    def backend(self, provider):
        module = dict(self.providers).get(provider)
        if module is None:
            raise ValueError(f"codec {self.name} does not support {provider}")
        return import_module(module)

    def capabilities(self, provider):
        """Compiler-free inventory; availability is per capability and provider."""
        return self.backend(provider).CAPABILITIES

    def supports(self, capability, shape, *, provider, **options):
        """Check an inline access specialization without importing its compiler."""
        if capability not in self.capabilities(provider):
            return False
        check = getattr(self.backend(provider), "supports", None)
        return check(capability, tuple(shape), **options) if check else True

    def bind(self, shape, *, provider, access=None, **options):
        """Resolve storage and inline access before constructing a consumer."""
        storage = self.storage(shape, provider=provider, **options)
        binding = self.backend(provider).bind(tuple(shape), **options, **(access or {}))
        if not isinstance(binding, Codec) or binding.storage != storage:
            raise TypeError("codec bind must return its declared physical storage")
        return binding


def get(scope, name):
    """Load a format from its own package without a central format registry."""
    if scope not in ("activations", "weights", "kv"):
        raise ValueError(f"unknown quantization scope: {scope}")
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"invalid codec name: {name}")
    package = f"akas.quantization.{scope}.{name}"
    module = package + ".definition"
    try:
        result = import_module(module).FORMAT
    except ModuleNotFoundError as error:
        if error.name not in (package, module):
            raise
        raise ValueError(f"unknown {scope} codec: {name}") from error
    except AttributeError as error:
        raise ValueError(
            f"codec {scope}/{name} must declare FORMAT in definition.py"
        ) from error
    if not isinstance(result, Format) or result.name != name:
        raise ValueError(f"invalid codec definition: {scope}/{name}")
    return result
