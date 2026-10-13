"""Compiler-free storage contracts and lazy codec discovery."""

from dataclasses import dataclass
from importlib import import_module
from typing import Callable
from akas.memory.state import BufferSpec


@dataclass(frozen=True)
class Storage:
    values: BufferSpec
    metadata: tuple[BufferSpec, ...] = ()


@dataclass(frozen=True)
class Format:
    name: str
    layout: Callable[[tuple[int, ...]], Storage]
    providers: tuple[tuple[str, str], ...] = ()

    layouts: tuple[tuple[str, Callable], ...] = ()

    def storage(self, shape, *, provider=None):
        shape = tuple(shape)
        if len(shape) < 2 or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError(
                "KV storage requires positive dimensions including channels"
            )
        return dict(self.layouts).get(provider, self.layout)(shape)

    def pair_layout(self, shape):
        storage = self.storage(shape)
        # Persistent-state ABI: K, V, then K metadata and V metadata.
        return (storage.values, storage.values, *storage.metadata, *storage.metadata)

    def backend(self, provider):
        module = dict(self.providers).get(provider)
        if module is None:
            raise ValueError(f"KV codec {self.name} does not support {provider}")
        backend = import_module(module)
        return backend.backend(provider) if hasattr(backend, "backend") else backend


def get(name):
    """A new format supplies FORMAT in its own module; no central format table."""
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"invalid KV codec name: {name}")
    module = "akas.quantization.kv." + name
    try:
        result = import_module(module).FORMAT
    except ModuleNotFoundError as error:
        if error.name != module:
            raise
        raise ValueError(f"unknown KV codec: {name}") from error
    except AttributeError as error:
        raise ValueError(f"KV codec module has no FORMAT: {name}") from error
    if not isinstance(result, Format) or result.name != name:
        raise ValueError(f"invalid KV codec definition: {name}")
    return result


def is_quantized(name):
    if name in ("float16", "bfloat16"):
        return False
    get(name)
    return True


def validate(name, *, unquantized=("bfloat16",)):
    if name not in unquantized:
        get(name)
    return name
