"""Install compact expert tile scheduling without changing projection math."""

from akas.artifacts.kernels import backbone_hashes
import hashlib
import json
from pathlib import Path
from akas.tensor.graph import CudaGraph
from akas.execution.plan import bind_named, call_bindings
from akas.execution.plan import submit as submit_calls
from akas.artifacts.kernels import artifact_path
from akas.artifacts.hybrid import identity


def implementation_hashes():
    import akas.kernels.moe.compact_experts as compact_experts
    import akas.kernels.hybrid.hybrid_prefill as prefill

    return (
        {
            m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (compact_experts, prefill)
        }
        | {__name__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        | backbone_hashes()
    )


def install(executor, bundle):
    executor.model._check()
    if executor.closed or getattr(executor, "pending_verification", False):
        raise RuntimeError("prefill executor must be idle")
    if getattr(executor, "_compact_experts_installed", False):
        raise ValueError("compact experts are already installed")
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "prefill.json").read_text())
    if manifest.get("compact_expert_implementation") != implementation_hashes():
        raise ValueError("compact expert source mismatch; rebuild the bundle")
    if (
        manifest["slots"],
        manifest["chunk"],
        manifest["context"],
        manifest.get("target"),
    ) != (
        executor.model.slots,
        executor.chunk,
        executor.model.context,
        executor.device.info["arch"],
    ):
        raise ValueError("compact expert capacity/target mismatch")
    d = executor.device
    loaded, buffers = {}, {}
    try:
        record = manifest["compact_experts"]
        for key in {record["tile_map"], *record["projections"].values()}:
            row = manifest["kernels"][key]
            path = artifact_path(bundle, row)
            loaded[key] = d.load(path)
        buffers = {
            name: d.empty((record["max_tiles"],), "int32")
            for name in ("tile_experts", "tile_offsets")
        }
        replacements = {
            id(executor.kernels[logical]): loaded[selected]
            for logical, selected in record["projections"].items()
        }
        route = executor.kernels[
            identity(
                "expert_routes", dict(slots=executor.model.slots, chunk=executor.chunk)
            )
        ]

        plan = []
        for kernel, bound in executor.plan:
            if kernel is route:
                plan.append((kernel, bound))
                plan.append(
                    bind_named(
                        d,
                        loaded[record["tile_map"]],
                        dict(counts=executor.buffers["counts"], **buffers),
                    )
                )
            elif id(kernel) in replacements:
                values = call_bindings(kernel, bound)
                plan.append(
                    bind_named(d, replacements[id(kernel)], dict(**values, **buffers))
                )
            else:
                plan.append((kernel, bound))
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
    executor.graph, executor.plan = graph, plan
    # Distinct keys preserve ownership of the originals as well as replacements.
    executor.kernels.update({"_compact_" + k: v for k, v in loaded.items()})
    executor.buffers.update({"_compact_" + k: v for k, v in buffers.items()})
    executor._compact_experts_installed = True
