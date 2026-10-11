"""Block-scaled FP8 hybrid decode for a loaded checkpoint's execution geometry."""

from dataclasses import asdict
from akas.artifacts.hybrid import implementation_hashes, requirements
from akas.compilation.build import build
from akas.compilation.plan import Bundle, Kernel, Source
from akas.compilation.profiles.hybrid.geometry import validate


def plan(
    checkpoint,
    *,
    slots=8,
    context=48000,
    target="sm_89",
    splits=16,
    kv_dtype="bfloat16",
):
    config = checkpoint.config
    validate(config)
    config.state_bytes(slots=slots, context=context)
    if slots not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("unsupported slot count")
    kernels = []
    for key, (kind, parameters) in requirements(
        config, slots, context, splits=splits, kv_dtype=kv_dtype
    ).items():
        source = Source(
            "akas.kernels.hybrid.hybrid", "source", (kind, parameters), export=False
        )
        if kind == "fp8_linear" and slots > 8:
            source = Source(
                "akas.kernels.linear.matmul",
                "make_kernel",
                (
                    "fp8_linear_mma",
                    dict(parameters, block_m=16, columns=64, threads=128),
                ),
                (
                    "tensor.compiler.entry",
                    "tensor.compiler.cuda_lowering",
                    "akas.kernels.elementwise",
                ),
            )
        kernels.append(Kernel(key, kind, parameters, source))
    return Bundle(
        dict(
            schema="tensor.qwen35-batch.v1",
            status="building",
            slots=slots,
            context=context,
            target=target,
            splits=splits,
            kv_dtype=kv_dtype,
            config=asdict(config),
            implementation=implementation_hashes(),
        ),
        "inference.json",
        kernels,
    )


def produce(checkpoint, out, **options):
    build(plan(checkpoint, **options), out)
    from pathlib import Path

    return Path(out)
