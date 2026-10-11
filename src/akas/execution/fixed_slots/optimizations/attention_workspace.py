"""Own a transient BF16 attention workspace while retaining FP8 KV storage."""

import json
from pathlib import Path
from akas.tensor.graph import CudaGraph
from akas.execution.plan import bind_named
from akas.execution.plan import call_bindings
from akas.tensor.executor import submit as submit_calls
from akas.artifacts.kernels import artifact_path
from tensor.runtime.cuda_target import matches_device
from akas.artifacts.fingerprints import (
    fixed_slots_hashes as implementation_hashes,
)


def install(executor, bundle):
    executor.model._check()
    if executor.closed or getattr(executor, "pending_verification", False):
        raise RuntimeError("executor must be idle")
    if getattr(executor, "_attention_workspace_installed", False):
        raise ValueError("attention workspace already installed")
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "attention-workspace.json").read_text())
    if manifest["implementation"] != implementation_hashes():
        raise ValueError("attention workspace implementation mismatch; rebuild bundle")
    if (manifest["slots"], manifest["chunk"], manifest["capacity"]) != (
        executor.model.slots,
        executor.chunk,
        executor.model.context,
    ):
        raise ValueError("attention workspace capacity mismatch")
    if not matches_device(manifest["target"], executor.device.info["arch"]):
        raise ValueError("attention workspace target mismatch")
    d = executor.device
    loaded = {}
    buffers = {}
    try:
        for name, row in manifest["kernels"].items():
            path = artifact_path(bundle, row)
            loaded[name] = d.load(path)
        shape = (manifest["slots"], 2, manifest["capacity"], 256)
        buffers = {
            name: d.empty(shape, "bfloat16") for name in ("decoded_k", "decoded_v")
        }

        plan = []
        replaced = 0
        for kernel, bound in executor.plan:
            identity = kernel.manifest["files"]["kernel.cubin"]
            if identity != manifest["original_cubin_sha256"]:
                plan.append((kernel, bound))
                continue
            values = call_bindings(kernel, bound)
            plan.append(bind_named(d, loaded["decode"], dict(values, **buffers)))
            plan.append(
                bind_named(
                    d,
                    loaded["attention"],
                    dict(values, kc=buffers["decoded_k"], vc=buffers["decoded_v"]),
                )
            )
            replaced += 1
        expected = manifest.get("expected_attention_calls")
        if replaced == 0 or (expected is not None and replaced != expected):
            raise ValueError("attention workspace source plan mismatch")
        graph = CudaGraph(
            d,
            lambda: submit_calls(d, plan),
            resources=(*executor.graph.resources, *loaded.values(), *buffers.values()),
        )
    except BaseException:
        for kernel in loaded.values():
            kernel.release()
        for buffer in buffers.values():
            buffer.release()
        raise
    executor.graph.close()
    executor.graph = graph
    executor.plan = plan
    executor.kernels.update({"_attention_workspace_" + k: v for k, v in loaded.items()})
    executor.buffers.update(
        {"_attention_workspace_" + k: v for k, v in buffers.items()}
    )
    executor._attention_workspace_installed = True
