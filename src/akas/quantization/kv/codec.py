"""KV codec lookup using the shared quantization storage contract."""

from akas.quantization.formats import Format as Format
from akas.quantization.formats import Storage as Storage
from akas.quantization.formats import get as _get


def get(name):
    return _get("kv", name)


def pair_layout(name, shape, *, provider=None):
    """Persistent-state order: K, V, then K metadata and V metadata."""
    storage = get(name).storage(shape, provider=provider)
    return (storage.values, storage.values, *storage.metadata, *storage.metadata)


def is_quantized(name):
    if name in ("float16", "float32", "bfloat16"):
        return False
    get(name)
    return True


def validate(name, *, unquantized=("bfloat16",)):
    if name not in unquantized:
        get(name)
    return name
