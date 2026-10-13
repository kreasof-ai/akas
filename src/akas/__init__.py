"""Inference and optional model loading without compiler imports."""

from akas.engine import Engine
from akas.config import EngineConfig
from akas.requests import Request
from akas.checkpoints.gguf import GGUF, GGUFError

_EXPORTS = {
    "Executor": ("akas.execution.executor", "Executor"),
    "Definition": ("akas.models.definition", "Definition"),
    "MTPHead": ("akas.models.heads.mtp", "MTPHead"),
    "SpeculativeDecoder": ("akas.generation.speculative.decoding", "SpeculativeDecoder"),
    "Tokenizer": ("akas.text.tokenizer", "Tokenizer"),
    "LFM2": ("akas.models.lfm2.model", "LFM2"),
    "LFM2Reference": ("akas.models.lfm2.model", "LFM2Reference"),
    "LFM2Batched": ("akas.models.lfm2.model", "LFM2Batched"),
    "Qwen35Batch": ("akas.models.qwen35.model", "Qwen35Batch"),
    "DenseEngine": ("akas.models.qwen35.model", "DenseEngine"),
    "Qwen35Checkpoint": ("akas.models.qwen35.weights", "Qwen35Checkpoint"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from importlib import import_module
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


__all__ = ["Engine", "EngineConfig", "Request", "GGUF", "GGUFError", *_EXPORTS]
