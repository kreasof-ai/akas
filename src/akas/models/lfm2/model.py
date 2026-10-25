"""LFM2 declarations following the common composition contract."""

from akas.models.definition import Composition, Definition
from akas.models.lfm2.weights import (
    LFM2Checkpoint,
    LFM2MoECheckpoint,
    ReferenceCheckpoint,
)


def compose(
    checkpoint,
    *,
    rows,
    chunk,
    max_requests,
    max_context,
    provider="cuda",
    phase=None,
    kv_dtype="float16",
    kv_padding=0,
    max_weight_bytes=64 * 1024 * 1024,
    optimized=True,
    activation_quantization=None,
    expert_weight_layout="raw",
    dense_weight_layout="raw",
    auxiliary_layers=(),
):
    from akas.models.lfm2._composition import program

    if phase not in (None, "decode", "prefill", "verify"):
        raise ValueError("LFM2 supports decode, prefill and verification phases")
    result = program(
        checkpoint,
        rows=rows,
        chunk=chunk,
        max_requests=max_requests,
        max_context=max_context,
        provider=provider,
        kv_dtype=kv_dtype,
        kv_padding=kv_padding,
        max_weight_bytes=max_weight_bytes,
        optimized=optimized,
        activation_quantization=activation_quantization,
        expert_weight_layout=expert_weight_layout,
        dense_weight_layout=dense_weight_layout,
        auxiliary_layers=auxiliary_layers,
        phase=phase,
    )
    return Composition(*result) if isinstance(result, tuple) else Composition(result)


class LFM2(Definition):
    checkpoint_architectures = ("lfm2",)
    checkpoint_type = LFM2Checkpoint
    model_defaults = {"kv_dtype": "float16", "optimized": True}


class LFM2MoE(Definition):
    checkpoint_architectures = ("lfm2moe",)
    checkpoint_type = LFM2MoECheckpoint
    model_defaults = {"kv_dtype": "float16", "optimized": True}
    optimization_chunks = (512, 1024, 2048, 4096, 256, 128, 32, 1)
    optimization_space = (("expert_weight_layout", ("planar",)),)


class LFM2Reference(Definition):
    checkpoint_type = ReferenceCheckpoint
    model_defaults = {"kv_dtype": "float16", "optimized": False}


class LFM2Batched(Definition):
    checkpoint_type = ReferenceCheckpoint
    model_defaults = {"kv_dtype": "float16", "optimized": True}
