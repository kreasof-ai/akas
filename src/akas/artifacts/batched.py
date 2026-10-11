"""Implementation identity shared by the AOT producer and native consumer."""

import hashlib
import importlib.util
from pathlib import Path


def implementation():
    modules = [
        "akas.artifacts.batched",
        "akas.models.qwen35.config",
        "akas.models.qwen35.weights",
        "akas.models.qwen35.model",
        "akas.execution.batched.model",
        "akas.execution.batched.batch",
        "akas.execution.batched.plan",
        "akas.execution.batched.workspace",
        "akas.execution.batched.draft",
    ]
    modules.extend(
        ('akas.generation.batched_mtp', 'akas.execution.batched.speculative_graph', 'akas.memory.recurrent_journal', 'akas.tensor.executor', 'akas.tensor.graph', 'akas.artifacts.kernels', 'akas.checkpoints.safetensors', 'akas.memory.state', 'akas.ops.hybrid_state', 'akas.ops.decoder', 'akas.ops.attention', 'akas.ops.gdn', 'akas.ops.shortconv', 'akas.ops.ffn', 'akas.checkpoints.upload', 'akas.ops.linear', 'akas.ops.normalization', 'akas.ops.quantization', 'akas.execution.plan', 'akas.kernels.elementwise', 'akas.kernels.hybrid.hybrid_batch', 'akas.kernels.linear.split_linear', 'akas.kernels.linear.head', 'akas.kernels.recurrent.recurrent', 'akas.kernels.speculation.speculation', 'akas.kernels.hybrid.hybrid', 'akas.kernels.speculation.mtp',)
    )
    return {
        name: hashlib.sha256(
            Path(importlib.util.find_spec(name).origin).read_bytes()
        ).hexdigest()
        for name in modules
    }


def validate_implementation(manifest, expected):
    if manifest.get("implementation") != expected:
        raise ValueError("dense implementation changed; regenerate the AOT bundle")
