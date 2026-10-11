"""Install selected Split-K projections into an existing native chunk graph."""

from akas.artifacts.kernels import backbone_hashes
import hashlib
import json
from pathlib import Path
from akas.execution.plan import bind_named, call_bindings
from akas.execution.plan import submit as submit_calls
from akas.artifacts.kernels import artifact_path
from akas.tensor.graph import CudaGraph
from akas.artifacts.hybrid import identity


def implementation_hashes():
    import akas.kernels.linear.speculative_linear as speculative_linear
    import akas.kernels.linear.matmul as matmul

    return (
        {
            m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (speculative_linear, matmul)
        }
        | {__name__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        | backbone_hashes()
    )


def install(executor, bundle):
    executor.model._check()
    if executor.closed or getattr(executor, "pending_verification", False):
        raise RuntimeError("executor must be idle")
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "prefill.json").read_text())
    if manifest.get("split_linear_implementation") != implementation_hashes():
        raise ValueError("Split-K implementation mismatch")
    if (manifest["slots"], manifest["chunk"], manifest["context"]) != (
        executor.model.slots,
        executor.chunk,
        executor.model.context,
    ):
        raise ValueError("Split-K capacity mismatch")
    d = executor.device
    loaded = {}
    buffers = {}
    replacements = {}
    try:
        if "fixed_routes" in manifest:
            key = manifest["fixed_routes"]
            row = manifest["kernels"][key]
            path = artifact_path(bundle, row)
            loaded[key] = d.load(path)
            buffers["groups"] = d.empty((executor.rows * 8,), "int32")
            buffers["group_routes"] = d.empty(
                (executor.rows * 8, executor.rows), "int32"
            )
        for logical, record in manifest["split_linear"].items():
            old = executor.kernels[logical]
            for key in (record["matmul"], record.get("merge")):
                if key is None:
                    continue
                if key in loaded:
                    continue
                row = manifest["kernels"][key]
                path = artifact_path(bundle, row)
                loaded[key] = d.load(path)
            if "partial_shape" in record:
                buffers[logical] = d.empty(tuple(record["partial_shape"]))
            replacements[id(old)] = (logical, record)
        plan = []

        for kernel, bound in executor.plan:
            if "fixed_routes" in manifest and kernel is executor.kernels.get(
                identity(
                    "expert_routes",
                    dict(slots=executor.model.slots, chunk=executor.chunk),
                )
            ):
                values = call_bindings(kernel, bound)
                plan.append(
                    bind_named(
                        d,
                        loaded[manifest["fixed_routes"]],
                        dict(
                            ids=values["ids"],
                            experts=buffers["groups"],
                            routes=buffers["group_routes"],
                        ),
                    )
                )
                continue
            record = replacements.get(id(kernel))
            if record is None:
                plan.append((kernel, bound))
                continue
            logical, record = record
            values = call_bindings(kernel, bound)
            if "merge" not in record:
                plan.append(bind_named(d, loaded[record["matmul"]], values))
                continue
            if record.get("fixed_rows"):
                values.pop("counts")
                values.update(experts=buffers["groups"], routes=buffers["group_routes"])
            out = values["out"]
            values["out"] = buffers[logical]
            plan.append(bind_named(d, loaded[record["matmul"]], values))
            plan.append(
                bind_named(
                    d, loaded[record["merge"]], dict(partial=buffers[logical], out=out)
                )
            )
        graph = CudaGraph(
            d,
            lambda: submit_calls(d, plan),
            resources=(*executor.graph.resources, *buffers.values(), *loaded.values()),
        )
    except BaseException:
        for value in buffers.values():
            value.release()
        for kernel in loaded.values():
            kernel.release()
        raise
    executor.graph.close()
    executor.graph = graph
    executor.plan = plan
    executor.kernels.update(loaded)
    executor.buffers.update({"_split_linear_" + k: v for k, v in buffers.items()})
