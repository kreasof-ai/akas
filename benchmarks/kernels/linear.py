"""Qualify floating projection schedules across row counts on one target.

Run as `python -m benchmarks.kernels.linear --out build/linear-search`.
The profile selects only measured geometry; it never covers unseen row counts.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import tensor
from akas.compilation.emit import build_artifact
from akas.compilation.emit import export_kernel
from akas.compilation.tuning import Measurement, Workload, tune
from akas.ops.registry import candidates
from akas.ops.linear.floating import projection_algorithm
from akas.ops.linear.webgpu.floating import MINIMUM_MMA_GROUPS as WEBGPU_MINIMUM
from akas.ops.linear.cuda.floating import MINIMUM_MMA_GROUPS as CUDA_MINIMUM
from akas.ops.linear.cuda.floating import MAXIMUM_GEMV_ROWS, mma_block_m


def prepared_timing(kernel, buffers, replays):
    """Amortize host submission using the same prepared path as model execution."""
    from akas.execution.plan import ExecutionPlan

    device = kernel.device
    plan = ExecutionPlan(device)
    plan.bind(kernel, tuple(buffers))
    plan *= replays
    graph = plan.capture(resources=(kernel, *buffers))
    try:
        for _ in range(3):
            graph.launch()
            device.synchronize()
        samples = []
        for _ in range(15):
            start = time.perf_counter()
            graph.launch()
            device.synchronize()
            samples.append((time.perf_counter() - start) / replays)
        return dict(
            launches_per_replay=replays,
            median_prepared_kernel_seconds=statistics.median(samples),
            min_prepared_kernel_seconds=min(samples),
        )
    finally:
        graph.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("cuda", "webgpu"), default="cuda")
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--o", type=int, default=4608)
    parser.add_argument(
        "--geometries", nargs="+", help="KxO pairs, e.g. 1024x4608 4608x1024"
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 256, 128, 512, 2048, 8192, 32768],
    )
    parser.add_argument("--budget", type=int, default=9)
    parser.add_argument("--launches-per-replay", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    geometries = (
        [tuple(map(int, value.split("x"))) for value in args.geometries]
        if args.geometries
        else [(args.k, args.o)]
    )
    if any(len(geometry) != 2 or min(geometry) <= 0 for geometry in geometries):
        parser.error("geometries must be positive KxO pairs")
    if min(args.k, args.o, args.launches_per_replay, *args.rows) <= 0:
        parser.error("geometry must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    records, measured = [], {}
    largest = max(max(k * o * 2, max(args.rows) * max(k, o) * 4) for k, o in geometries)
    options = (
        dict(max_buffer_size=max(largest, 128 * 1024 * 1024))
        if args.provider == "webgpu"
        else {}
    )
    with tensor.Device(provider=args.provider, **options) as device:
        target = (
            device.info.get("arch") if args.provider == "cuda" else "webgpu-portable-v1"
        )
        parameters = dict(
            optimized=True,
            weight_dtype="float16",
        )
        eligible = candidates(
            "linear",
            {**parameters, "r": 1, "k": geometries[0][0], "o": geometries[0][1]},
            contract="reference",
            provider=args.provider,
            target=target,
            phase="forward",
        )
        catalog = tuple(
            item
            for item in eligible
            if item.priority == max(i.priority for i in eligible)
        )

        def measure(item, p, workload):
            effective = dict(p)
            algorithm = projection_algorithm(
                {
                    **p,
                    "maximum_gemv_rows": p.get("maximum_gemv_rows", MAXIMUM_GEMV_ROWS),
                }
                if args.provider == "cuda"
                else p,
                minimum_groups=p.get(
                    "minimum_mma_groups",
                    WEBGPU_MINIMUM if args.provider == "webgpu" else CUDA_MINIMUM,
                ),
                default_tile=(16, 32, 64)
                if args.provider == "webgpu"
                else (32, 64, 32),
            )
            effective["algorithm"] = algorithm
            effective.pop("minimum_mma_groups", None)
            effective.pop("maximum_gemv_rows", None)
            if algorithm == "gemv":
                for name in (
                    "block_m",
                    "block_n",
                    "block_k",
                    "schedule",
                    "outer_shared_dtype",
                ):
                    effective.pop(name, None)
            elif args.provider == "cuda":
                effective.setdefault("block_m", mma_block_m(p))
                effective.setdefault("block_n", 64)
                effective.setdefault("block_k", 32)
            elif args.provider == "webgpu":
                effective.setdefault("schedule", "register")
                if effective["schedule"] == "register":
                    effective.pop("outer_shared_dtype", None)
            identity = json.dumps([item.module, effective], sort_keys=True)
            if identity in measured:
                return measured[identity]
            tag = hashlib.sha256(identity.encode()).hexdigest()[:16]
            rng = np.random.default_rng(41)
            x = rng.normal(size=(p["r"], p["k"])).astype("float32")
            w = rng.normal(0, 0.02, size=(p["o"], p["k"])).astype(
                parameters["weight_dtype"]
            )
            checked = sorted(set([0, p["r"] // 2, p["r"] - 1]))
            half = algorithm == "mma"
            expected = (
                x[checked].astype("float16").astype("float32") if half else x[checked]
            ) @ w.astype("float16" if half else "float32").astype("float32").T
            source, artifact = args.out / f"{tag}.py", args.out / f"{tag}.tbin"
            source.write_text(
                export_kernel(
                    "akas.ops." + item.module, "make_kernel", *item.source_arguments(p)
                )
            )
            artifact.unlink(missing_ok=True)
            build_artifact(source, artifact, target=target, provider=args.provider)
            buffers = [
                device.from_numpy(x.reshape(-1)),
                device.from_numpy(w.reshape(-1)),
                device.empty((p["r"] * p["o"],), "float32"),
            ]
            try:
                with device.load(artifact) as kernel:
                    kernel.launch(*buffers)
                    device.synchronize()
                    actual = buffers[-1].to_numpy().reshape(p["r"], p["o"])
                    error = float(
                        np.linalg.norm(actual[checked] - expected)
                        / np.linalg.norm(expected)
                    )
                    correct = bool(np.isfinite(actual).all() and error < 0.0001)
                    timing = {
                        **tensor.bench(kernel, tuple(buffers), warmup=3, iters=15),
                        **prepared_timing(kernel, buffers, args.launches_per_replay),
                    }
                value = Measurement(timing["median_prepared_kernel_seconds"], correct)
                records.append(
                    dict(
                        implementation=item.module,
                        parameters=p,
                        checked_rows=checked,
                        relative_rms=error,
                        correct=correct,
                        **timing,
                    )
                )
                (args.out / "measurements.json").write_text(
                    json.dumps(dict(device=device.info, measurements=records), indent=2)
                    + "\n"
                )
                print(
                    workload.name, p.get("block_m"), p.get("block_n"), value, flush=True
                )
                measured[identity] = value
                return value
            finally:
                for buffer in buffers:
                    buffer.release()

        workloads = [
            Workload(f"{k}x{o}/rows-{r}", dict(r=r, k=k, o=o))
            for k, o in geometries
            for r in dict.fromkeys(args.rows)
        ]
        result = tune(
            "linear",
            parameters,
            workloads,
            measure,
            contract="reference",
            provider=args.provider,
            target=target,
            catalog=catalog,
            budget=args.budget,
            width=4,
        )
        profile = result.profile(
            "linear", parameters, provider=args.provider, target=target
        ).data
        entry = profile["entries"][0]
        profile["entries"] = [
            {**entry, "parameters": {**parameters, **workload.parameters}}
            for workload in workloads
        ]
        (args.out / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
        (args.out / "result.json").write_text(
            json.dumps(asdict(result), indent=2) + "\n"
        )
        print(result, flush=True)


if __name__ == "__main__":
    main()
