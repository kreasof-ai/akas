"""Inference and optional model loading without compiler imports."""

from akas.engine import Engine
from akas.config import EngineConfig
from akas.requests import Request
from akas.checkpoints.gguf import GGUF, GGUFError

_EXPORTS = {
    "Executor": ("akas.execution.executor", "Executor"),
    "Definition": ("akas.models.definition", "Definition"),
    "SpeculativeDecoder": (
        "akas.generation.speculative.decoding",
        "SpeculativeDecoder",
    ),
    "Tokenizer": ("akas.text.tokenizer", "Tokenizer"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        from akas.models import resolve_export
        import warnings

        value = resolve_export(name)
        warnings.warn(
            f"Import {name} from its akas.models family package instead of akas",
            DeprecationWarning,
            stacklevel=2,
        )
        globals()[name] = value
        return value
    from importlib import import_module

    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


__all__ = ["Engine", "EngineConfig", "Request", "GGUF", "GGUFError", *_EXPORTS]
