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
    max_weight_bytes=64 * 1024 * 1024,
    optimized=True,
    activation_quantization=None,
):
    from akas.models.lfm2._composition import program

    if phase not in (None, "decode", "prefill"):
        raise ValueError("LFM2 supports decode and prefill phases")
    return Composition(
        program(
            checkpoint,
            rows=rows,
            chunk=chunk,
            max_requests=max_requests,
            max_context=max_context,
            provider=provider,
            kv_dtype=kv_dtype,
            max_weight_bytes=max_weight_bytes,
            optimized=optimized,
            activation_quantization=activation_quantization,
        )
    )


class LFM2(Definition):
    checkpoint_architectures = ("lfm2",)
    checkpoint_type = LFM2Checkpoint
    model_defaults = {"kv_dtype": "float16", "optimized": True}


class LFM2MoE(Definition):
    checkpoint_architectures = ("lfm2moe",)
    checkpoint_type = LFM2MoECheckpoint
    model_defaults = {"kv_dtype": "float16", "optimized": True}
    optimization_chunks = (256, 512, 1024, 128, 32, 1)


class LFM2Reference(Definition):
    checkpoint_type = ReferenceCheckpoint
    model_defaults = {"kv_dtype": "float16", "optimized": False}


class LFM2Batched(Definition):
    checkpoint_type = ReferenceCheckpoint
    model_defaults = {"kv_dtype": "float16", "optimized": True}
