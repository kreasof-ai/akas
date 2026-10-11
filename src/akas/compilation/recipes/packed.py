"""Compile the packed-weight execution contract from a loaded checkpoint."""

from dataclasses import asdict
import json
from pathlib import Path

from akas.artifacts.fingerprints import packed_hashes as implementation_hashes
from akas.execution.bindings.packed.requirements import requirements
from akas.compilation.build import build
from akas.compilation.plan import Bundle
from akas.compilation.plan import Kernel
from akas.compilation.operations import source
from akas.ops.schedules.packed_selection import valid_rows
from akas.ops.schedules.packed_parameters import CUDA_PROFILES


def plan(
    checkpoint,
    *,
    context=8448,
    rows=None,
    target=None,
    provider="cuda",
    webgpu_profile="portable",
    cuda_profile="default",
    schedule_profile=None,
):
    if provider not in ("cuda", "webgpu"):
        raise ValueError("requires CUDA or WebGPU")
    rows = rows or ((1, 32) if provider == "webgpu" else (1, 128))
    target = target or ("webgpu-portable-v1" if provider == "webgpu" else "sm_86")
    config, gguf = checkpoint.config, checkpoint.gguf
    if cuda_profile not in CUDA_PROFILES:
        raise ValueError("unsupported CUDA kernel profile")
    profile = cuda_profile if provider == "cuda" else webgpu_profile
    if (
        not valid_rows(provider, rows, profile)
        or type(context) is not int
        or not 1 <= context <= config.max_context
    ):
        raise ValueError(
            "requires the provider row profile and a supported positive context"
        )
    capacity = (context + max(rows) + 63) // 64 * 64
    wanted = requirements(
        gguf,
        capacity,
        rows,
        provider=provider,
        webgpu_profile=webgpu_profile,
        cuda_profile=cuda_profile,
        config=config,
    )
    selected = None
    if (
        schedule_profile is None
        and provider == "cuda"
        and cuda_profile == "optimized"
        and target == "sm_86"
    ):
        schedule_profile = Path(__file__).resolve().parent / "data/cuda-sm86.json"
    if schedule_profile is not None:
        from tensor.compiler.search import ScheduleProfile

        selected = ScheduleProfile(json.loads(Path(schedule_profile).read_text()))
        selected.select("", {}, provider=provider, target=target)
    contract = (
        "packed-baseline"
        if provider == "cuda" and cuda_profile != "optimized"
        else "packed"
    )
    kernels = []
    for key, (kind, parameters) in wanted.items():
        schedule = (
            selected.select(kind, parameters, provider=provider, target=target)
            if selected
            else {}
        )
        kernels.append(
            Kernel(
                key,
                kind,
                parameters,
                source(
                    kind,
                    {**parameters, **schedule},
                    contract=contract,
                    provider=provider,
                    target=target,
                ),
                schedule=schedule,
            )
        )
    manifest = dict(
        schema="tensor.lfm2-inference.v1",
        config=asdict(config),
        capacity=capacity,
        rows=rows,
        target=target,
        provider=provider,
        implementation=implementation_hashes(provider),
    )
    manifest["webgpu_profile" if provider == "webgpu" else "cuda_profile"] = profile
    if selected:
        manifest["schedule_profile"] = {**selected.data, "sha256": selected.sha256}
    return Bundle(
        manifest,
        "inference.json",
        kernels,
        provider=provider,
        source_directory="src",
        artifact_directory="artifacts",
        path_field="artifact",
    )


def produce(checkpoint, out, **options):
    return build(plan(checkpoint, **options), out)
