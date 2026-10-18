"""Infer legal program choices from declarations, operation policies and hardware."""

from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess

from akas.optimization.context import identity, symbol


def hardware(provider, ordinal):
    import os
    import tensor

    if provider == "webgpu" and "VK_ICD_FILENAMES" not in os.environ:
        root = Path(__file__).resolve().parents[3]
        icd = root / "build/baselines/nvidia-graphics/local_egl_icd.json"
        if icd.is_file():
            os.environ["VK_ICD_FILENAMES"] = str(icd)
            graphics = str(icd.parent)
            ld = os.environ.get("LD_LIBRARY_PATH", "")
            if graphics not in ld.split(":"):
                os.environ["LD_LIBRARY_PATH"] = f"{graphics}:{ld}" if ld else graphics
            runtime = root / "build/baselines/runtime"
            runtime.mkdir(exist_ok=True)
            runtime.chmod(0o700)
            os.environ.setdefault("XDG_RUNTIME_DIR", str(runtime))

    with tensor.Device(ordinal, provider=provider) as device:
        result = dict(device.info)
    try:
        rows = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,name,memory.free,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            .strip()
            .splitlines()
        )
        # Refuse ambiguous device mapping rather than apply another GPU's memory limit.
        if len(rows) == 1 and result["name"] in rows[0]:
            uuid, _, free, total, driver = [
                value.strip() for value in rows[0].split(",")
            ]
            result.update(
                uuid=uuid,
                free_bytes=int(free) * 1024**2,
                total_bytes=int(total) * 1024**2,
                driver=driver,
            )
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def software():
    result = {}
    for name in ("tensor-workspace", "tilelang", "apache-tvm-ffi", "wgpu", "numpy"):
        try:
            d = importlib.metadata.distribution(name)
            result[name] = dict(
                version=d.version, record=identity(d.read_text("RECORD"))
            )
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def estimate(graph):
    """Conservative raw buffer bounds, before scratch reuse and shared phases."""
    from tensor.runtime.dtypes import dtype

    sizes = [
        math.prod(spec.shape) * dtype(spec.dtype).itemsize
        for spec in graph.buffers.values()
        if spec.alias_of is None
    ]
    return dict(graph_bytes=sum(sizes), largest_buffer_bytes=max(sizes, default=0))


def memory_bound(graphs):
    from akas.compilation.buffers import scratch_slots
    from tensor.runtime.dtypes import dtype

    persistent = {}
    private = 0
    for graph in graphs:
        size = lambda spec: math.prod(spec.shape) * dtype(spec.dtype).itemsize
        for name, spec in graph.buffers.items():
            if spec.alias_of is None and spec.role in ("weight", "state"):
                persistent[name] = max(persistent.get(name, 0), size(spec))
        groups = scratch_slots(graph)
        private += sum(size(graph.buffers[names[0]]) for names in groups)
        private += sum(
            size(spec)
            for spec in graph.buffers.values()
            if spec.alias_of is None and spec.role not in ("weight", "state", "scratch")
        )
    return dict(
        persistent_bytes=sum(persistent.values()),
        cached_frame_upper_bytes=private,
        total_upper_bytes=sum(persistent.values()) + private,
    )


def probe(context, provider):
    from akas.compilation.policy import candidates
    from akas.models.definition import Composition

    model = symbol(context["definition"])
    loaded = model.checkpoint_type(context["checkpoint"])
    device = hardware(provider, context["device"])
    data = context["workload"]
    capacity = max(data["batches"])
    limit = max(data["prefix_tokens"]) + data["output_tokens"]
    options = dict(getattr(model, "model_defaults", {}))
    if context.get("kv_dtype") is not None:
        options["kv_dtype"] = context["kv_dtype"]
    elif provider == "webgpu" and options.get("kv_dtype") == "bfloat16":
        options["kv_dtype"] = "float16"
    chunks, failures, policies = [], [], [{}]
    for chunk in dict.fromkeys(min(limit, c) for c in model.optimization_chunks):
        try:
            composed = model.compose(
                loaded,
                rows=min(capacity, 32),
                chunk=chunk,
                max_requests=capacity,
                max_context=limit,
                provider=provider,
                **options,
            )
            graph = composed.graph if isinstance(composed, Composition) else composed
            chunks.append(chunk)
            if len(chunks) == 1:
                policies = candidates(graph, provider=provider, target=device["arch"])
        except (ValueError, TypeError, NotImplementedError) as error:
            failures.append(dict(chunk=chunk, error=str(error)))
    if not chunks:
        return dict(
            status="unsupported", device=device, errors=failures, provider=provider
        )
    head = getattr(model, "speculative_head", None)
    speculation = []
    deferred_windows = []
    if head is not None and context.get("mode", "auto") != "ar":
        for window in (2, 4, 8):
            try:
                verify = model.compose(
                    loaded,
                    rows=1,
                    chunk=window,
                    max_requests=capacity,
                    max_context=limit + window,
                    provider=provider,
                    phase="verify",
                    **options,
                )
                model.compose(
                    loaded,
                    rows=1,
                    chunk=chunks[0],
                    max_requests=capacity,
                    max_context=limit + window,
                    provider=provider,
                    phase="proposal",
                    **options,
                )
                speculation.append(window)
                if isinstance(verify, Composition):
                    from akas.compilation.rewrites import deferred_state

                    try:
                        if deferred_state(verify.graph) is not None:
                            deferred_windows.append(window)
                    except ValueError:
                        pass
                if isinstance(verify, Composition):
                    for policy in candidates(
                        verify.graph,
                        commit=verify.commit,
                        provider=provider,
                        target=device["arch"],
                    ):
                        if policy not in policies:
                            policies.append(policy)
            except (
                ValueError,
                TypeError,
                NotImplementedError,
                AttributeError,
            ) as error:
                failures.append(dict(window=window, error=str(error)))
    if context.get("mode") == "mtp" and not speculation:
        return dict(
            status="unsupported",
            device=device,
            errors=failures
            + [dict(error="no supported speculative head for this checkpoint/backend")],
            provider=provider,
        )
    variants = []
    for name, values in model.optimization_space:
        for value in values:
            alternative = {**options, name: value}
            try:
                composed = model.compose(
                    loaded,
                    rows=min(capacity, 32),
                    chunk=chunks[0],
                    max_requests=capacity,
                    max_context=limit,
                    provider=provider,
                    **alternative,
                )
                default = model.compose(
                    loaded,
                    rows=min(capacity, 32),
                    chunk=chunks[0],
                    max_requests=capacity,
                    max_context=limit,
                    provider=provider,
                    **options,
                )
                if composed.graph.metadata() != default.graph.metadata():
                    variants.append(alternative)
            except (ValueError, TypeError, NotImplementedError) as error:
                failures.append(dict(option=name, value=value, error=str(error)))
    return dict(
        status="ok",
        device=device,
        provider=provider,
        chunks=chunks,
        policies=policies,
        model_options=options,
        option_variants=variants,
        deferred_windows=deferred_windows,
        windows=speculation,
        head=head,
        errors=failures,
        model_config=asdict(loaded.config),
        software=software(),
    )


def recipes(context, probes):
    """Finite, ordered search; runtime choices are generated rather than requested."""
    capacity = max(context["workload"]["batches"])
    plans = []

    def add(
        probe,
        *,
        chunk=None,
        group=None,
        graphs=True,
        trials=1,
        policy=None,
        window=None,
        model_options=None,
        deferred=False,
    ):
        recipe = dict(
            provider=probe["provider"],
            target=probe["device"]["arch"],
            chunk=chunk or probe["chunks"][0],
            prefill_requests=group or min(capacity, 32),
            graphs=graphs,
            trials=trials,
            policy=policy or {},
            model_options=probe["model_options"]
            if model_options is None
            else model_options,
            window=window,
            deferred=deferred,
            head=probe["head"] if window else None,
        )
        recipe["id"] = identity(recipe)[:16]
        if recipe not in plans:
            plans.append(recipe)

    legal = [p for p in probes if p["status"] == "ok"]
    for p in legal:
        add(p, graphs=False, trials=1)
    for p in legal:
        add(p)
    for p in legal:
        for chunk in p["chunks"]:
            add(p, chunk=chunk)
        for group in dict.fromkeys(min(capacity, n) for n in (1, 8, 64)):
            add(p, group=group)
        for policy in p["policies"]:
            add(p, policy=policy)
        for options in p.get("option_variants", ()):
            add(p, model_options=options)
        for window in p["windows"]:
            add(p, window=window)
            if window in p.get("deferred_windows", ()):
                add(p, window=window, deferred=True)
            for policy in p["policies"]:
                if "gdn" in policy:
                    add(p, window=window, policy=policy)
    if context.get("mode") == "mtp":
        plans = plans[:1] + [p for p in plans if p["window"] is not None]
    return plans


def refine(recipe, trials):
    """Deepen the current winner while retaining its program and serving choices."""
    if trials <= recipe["trials"]:
        return None
    result = {**{k: v for k, v in recipe.items() if k != "id"}, "trials": trials}
    result["id"] = identity(result)[:16]
    return result


def worker_allowance(budget, remaining, jobs, *, reference, provider):
    """A larger workload must get enough time to repeat its measured checks."""
    import os

    if os.environ.get("AKAS_SKIP_AUTOTUNE") == "1":
        return max(5, remaining)
    observed = max(
        (
            j.get("elapsed_seconds", 0)
            for j in jobs
            if j.get("status") == "passed" and j["recipe"]["provider"] == provider
        ),
        default=0,
    )
    return min(
        remaining, max(5, budget * (0.45 if reference is None else 0.2), observed * 1.5)
    )


def geometries(data, recipe):
    """Match proposal buffers to target prefill frames and verifier frames."""
    batches = sorted({1, *data["batches"]})
    groups = sorted({1, min(max(data["batches"]), recipe["prefill_requests"])})
    chunk, window = recipe["chunk"], recipe["window"]
    result = [(b, 1) for b in batches]
    if chunk > 1:
        result.extend((g, chunk) for g in groups)
    if window is not None:
        result.extend((b, window, "verify") for b in batches)
        result.extend((b, 1, "proposal") for b in batches)
        result.extend((b, window, "proposal") for b in batches)
        if chunk > 1:
            result.extend((g, chunk, "proposal") for g in groups)
    return list(dict.fromkeys(result))


def score(report, reference, context):
    """Whole-engine speedup, with every requested shape and latency bound checked."""
    import os

    skip_autotune = os.environ.get("AKAS_SKIP_AUTOTUNE") == "1"
    rows = report["results"]
    expected = {
        (b, p, context["workload"]["output_tokens"])
        for b in context["workload"]["batches"]
        for p in context["workload"]["prefix_tokens"]
    }
    keys = lambda row: (
        row["case"]["batch"],
        row["case"]["prompt"],
        row["case"]["generated"],
    )
    if {keys(row) for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("candidate omitted or duplicated workload shapes")
    if not any(row["status"] == "ok" for row in rows):
        raise ValueError("candidate has no completed workload shapes")
    baselines = {keys(row): row for row in reference["results"]}
    ratios = []
    for row in rows:
        if row["status"] == "timeout":
            continue
        if row["status"] != "ok" or not row["correctness"]["correct"]:
            raise ValueError("candidate has unqualified workload results")
        values = row["serving"]
        for field, bound in (
            ("ttft_ms", context.get("ttft_ms")),
            ("tpot_ms", context.get("tpot_ms")),
        ):
            if bound is not None and values[field] > bound:
                return None
        base_row = baselines.get(keys(row), {})
        if base_row.get("status") != "ok":
            continue
        ratio = base_row["serving"]["seconds"] / values["seconds"]
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("invalid serving duration")
        ratios.append(ratio)
    if not ratios:
        return 1.0 if skip_autotune else None
    if min(ratios) < 0.95 and not skip_autotune:
        return None
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))
