"""Block-scaled FP8 hybrid decode for a loaded checkpoint's execution geometry."""

from dataclasses import asdict
from akas.artifacts.fingerprints import block_fp8_hashes as implementation_hashes
from akas.execution.bindings.block_fp8.requirements import requirements
from akas.compilation.build import build
from akas.compilation.plan import Bundle
from akas.compilation.plan import Kernel
from akas.compilation.operations import source
from akas.execution.bindings.block_fp8.geometry import validate


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
        selected = source(
            kind,
            parameters,
            contract="block-fp8",
            phase="decode",
            target=target,
            dependencies=(
                "tensor.compiler.frontend.source",
                "tensor.compiler.cuda.lowering",
                "akas.ops.normalization.kernels",
            ),
        )
        kernels.append(Kernel(key, kind, parameters, selected))
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
