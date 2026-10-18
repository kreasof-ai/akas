"""Qwen3.5 declarations following the common composition contract."""

from akas.models.definition import Composition, Definition
from akas.models.qwen35.config import Qwen35MoEConfig
from akas.models.qwen35.weights import Qwen35DenseCheckpoint, Qwen35MoECheckpoint


def compose(
    checkpoint,
    *,
    rows,
    chunk,
    max_requests,
    max_context,
    provider="cuda",
    phase=None,
    **model_options,
):
    from akas.models.qwen35._composition import compose_dense, compose_moe_program

    if (
        isinstance(checkpoint.config, Qwen35MoEConfig)
        and getattr(checkpoint, "storage_format", None) != "ggml"
    ):
        result = compose_moe_program(
            checkpoint,
            rows=rows,
            chunk=chunk,
            max_requests=max_requests,
            max_context=max_context,
            provider=provider,
            phase=phase,
            **model_options,
        )
    else:
        if getattr(checkpoint, "storage_format", None) == "ggml":
            model_options.pop("splits", None)
            if model_options.get("fused_head"):
                raise ValueError(
                    "GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option"
                )
        if provider not in ("cuda", "webgpu"):
            raise ValueError("Qwen3.5 requires CUDA or WebGPU")
        if provider == "webgpu" and model_options.get("fused_head"):
            raise ValueError("fused_head is a native CUDA BF16 option")
        kv_dtype = model_options.pop("kv_dtype", "bfloat16")
        if provider == "webgpu" and kv_dtype == "bfloat16":
            kv_dtype = "float16"
        if (provider == "cuda" and kv_dtype not in ("bfloat16", "fp8")) or (
            provider == "webgpu" and kv_dtype not in ("float16", "fp8")
        ):
            raise ValueError(f"unsupported Qwen KV dtype {kv_dtype!r} for {provider}")
        if phase not in (None, "decode", "prefill", "verify", "proposal"):
            raise ValueError("unknown Qwen phase")
        result = compose_dense(
            checkpoint,
            slots=rows,
            chunk=chunk,
            pool=max_requests,
            context=max_context,
            provider=provider,
            kv_dtype=kv_dtype,
            verify=phase == "verify",
            draft=phase == "proposal",
            **model_options,
        )
    if isinstance(result, tuple):
        return Composition(*result)
    return Composition(result)


class Qwen35MoE(Definition):
    """Qwen3.5 MoE text model."""

    checkpoint_type = Qwen35MoECheckpoint
    checkpoint_architectures = ("qwen35moe", "qwen3_5_moe_text")
    optimization_space = (("splits", (4, 8, 32)),)
    speculative_head = "akas.models.qwen35.mtp:MTPHead"
    model_defaults = {"splits": 16, "kv_dtype": "bfloat16"}


class Qwen35Dense(Definition):
    """Qwen3.5 dense text model."""

    checkpoint_type = Qwen35DenseCheckpoint
    checkpoint_architectures = ("qwen35", "qwen3_5_text")
    optimization_space = (("fused_head", (True,)),)
    speculative_head = "akas.models.qwen35.mtp:MTPHead"
    model_defaults = {"fused_head": False, "kv_dtype": "bfloat16"}


# Compatibility for existing imports and stored definition paths.
Qwen35Batch = Qwen35MoE
DenseEngine = Qwen35Dense
