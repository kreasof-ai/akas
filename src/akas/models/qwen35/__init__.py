"""Public model declarations; new models export their names here."""

from akas.models.qwen35.model import Qwen35Dense, Qwen35MoE
from akas.models.qwen35.mtp import MTPHead
from akas.models.qwen35.weights import Qwen35DenseCheckpoint, Qwen35MoECheckpoint

# Compatibility for historical family and root imports.
Qwen35Batch = Qwen35MoE
DenseEngine = Qwen35Dense
Qwen35Checkpoint = Qwen35MoECheckpoint

__all__ = [
    "Qwen35Dense",
    "Qwen35MoE",
    "Qwen35DenseCheckpoint",
    "Qwen35MoECheckpoint",
    "MTPHead",
    "Qwen35Batch",
    "DenseEngine",
    "Qwen35Checkpoint",
]
