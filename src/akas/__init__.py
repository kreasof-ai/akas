"""Optional model loading and inference without framework/compiler imports."""

from akas.engine import Engine
from akas.config import EngineConfig
from akas.requests import Request
from akas.checkpoints.gguf import GGUF
from akas.checkpoints.gguf import GGUFError


def __getattr__(name):
    if name in ("LFM2", "LFM2Request"):
        from akas.models.lfm2.model import LFM2
        from akas.execution.bindings.packed.request import LFM2Request

        return {"LFM2": LFM2, "LFM2Request": LFM2Request}[name]
    if name == "Tokenizer":
        from akas.text.tokenizer import Tokenizer

        return Tokenizer
    if name == "Qwen35Batch":
        from akas.models.qwen35.model import Qwen35Batch

        return Qwen35Batch
    if name == "Qwen35Prefill":
        from akas.execution.bindings.block_fp8.prefill import Qwen35Prefill

        return Qwen35Prefill
    if name == "Qwen35Checkpoint":
        from akas.models.qwen35.weights import Qwen35Checkpoint

        return Qwen35Checkpoint
    if name == "Qwen35MTP":
        from akas.execution.bindings.block_fp8.draft import Qwen35MTP

        return Qwen35MTP
    if name == "Qwen35Verifier":
        from akas.execution.bindings.block_fp8.verify import Qwen35Verifier

        return Qwen35Verifier
    if name == "Qwen35Speculative":
        from akas.execution.bindings.block_fp8.speculative import Qwen35Speculative

        return Qwen35Speculative
    raise AttributeError(name)


__all__ = [
    "Engine",
    "EngineConfig",
    "Request",
    "GGUF",
    "GGUFError",
    "LFM2",
    "LFM2Request",
    "Tokenizer",
    "Qwen35Batch",
    "Qwen35Prefill",
    "Qwen35Checkpoint",
    "Qwen35MTP",
    "Qwen35Verifier",
    "Qwen35Speculative",
]
