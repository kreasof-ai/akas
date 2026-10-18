"""One disposable GPU process per probe or compilation/serving candidate."""

import argparse
import asyncio
from functools import partial
import json
import os
from pathlib import Path
import resource
import traceback

from akas.benchmarking.history import Recorder, digest, write, utc
from akas.optimization.context import symbol


def status(path, phase, **details):
    write(
        Path(path).parent / "progress.json",
        dict(phase=phase, updated_at=utc(), **details),
    )


def candidate(job, output):
    import numpy as np
    import tensor
    from akas.compilation import compile_checkpoint
    from akas.compilation.tuning import SearchSettings
    from akas.optimization.deployment import describe, service
    from akas.optimization.evaluation import evaluate, http_benchmark
    from akas.optimization.planning import (
        estimate,
        memory_bound,
        geometries as plan_geometries,
    )

    context, recipe = job["context"], job["recipe"]
    root = Path(output).parent
    program = root / "program"
    program.mkdir(exist_ok=True)
    # Bundles own their paths. Link immutable files rather than symlinking
    # outside the loader's integrity boundary. Atomic cache writes and artifact
    # replacement leave the preceding attempt intact.
    import shutil

    def link(source, target):
        Path(target).unlink(missing_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    for previous in job.get("cache_sources", ()):
        for name in ("search-cache", "search-kernels", "compiler-cache"):
            cache = Path(previous) / name
            if cache.is_dir():
                shutil.copytree(
                    cache, program / name, copy_function=link, dirs_exist_ok=True
                )
    model = symbol(context["definition"])
    data = context["workload"]
    capacity = max(data["batches"])
    chunk = recipe["chunk"]
    limit = max(data["prefix_tokens"]) + data["output_tokens"] + (recipe["window"] or 0)
    windows = () if recipe["window"] is None else (recipe["window"],)
    geometries = plan_geometries(data, recipe)
    skip_autotune = os.environ.get("AKAS_SKIP_AUTOTUNE") == "1"
    metadata = dict(
        runtime="akas",
        purpose="optimization",
        protocol="native-token-step-v2",
        model=context["definition"],
        definition=context["definition"],
        provider=recipe["provider"],
        target=recipe["target"],
        mode="mtp" if windows else "ar",
        kv_dtype=recipe["model_options"].get("kv_dtype"),
        graphs=recipe["graphs"],
        checkpoint=context["checkpoint"],
        checkpoint_sha256=job["checkpoint_sha256"],
        campaign_identity=job["identity"],
        recipe=recipe,
        device=job["device"],
        qualification="independent operation references; model outputs preserved against campaign baseline",
        warmup=int(
            os.environ.get("AKAS_BENCHMARK_WARMUP", "0" if skip_autotune else "1")
        ),
        repeats=int(
            os.environ.get("AKAS_BENCHMARK_REPEATS", "1" if skip_autotune else "3")
        ),
        seed=17,
    )
    recorder = Recorder(
        job["history"],
        metadata,
        parent=job.get("parent_run"),
        kind="configuration",
        change="Automatic compilation and serving optimization",
    )
    metadata["run_id"] = recorder.id
    write(
        root / "history.json", dict(directory=str(recorder.directory), id=recorder.id)
    )
    report = dict(
        schema="akas.benchmark.v1",
        metadata=metadata,
        results=[
            dict(
                case=dict(
                    batch=b,
                    prompt=p,
                    generated=data["output_tokens"],
                    depth=0,
                    mixed=False,
                ),
                status="pending",
            )
            for b in data["batches"]
            for p in data["prefix_tokens"]
        ],
    )
    recorder.update(report)
    result = dict(status="failed", run_id=recorder.id)
    try:
        status(output, "memory-planning", recipe=recipe["id"])
        loaded = model.checkpoint_type(context["checkpoint"])
        gguf = getattr(loaded, "gguf", None)
        if gguf is not None:
            preset = gguf.metadata.get("general.file_type")
            metadata.update(
                weight_format={0: "f32", 1: "f16", 2: "q4_0", 15: "q4_k_m"}.get(
                    preset, "gguf"
                ),
                gguf_file_type=preset,
                weight_encodings=sorted(
                    {info.encoding for info in gguf.tensors.values()}
                ),
            )
        else:
            config_path = Path(context["checkpoint"]) / "config.json"
            config = (
                json.loads(config_path.read_text()) if config_path.is_file() else {}
            )
            config = config.get("text_config", config)
            precision = config.get("dtype", config.get("torch_dtype"))
            metadata["weight_format"] = {"bfloat16": "bf16", "float16": "f16"}.get(
                precision, "safetensors"
            )
        bounds, graphs = [], []
        for b, t, *phase in geometries:
            composed = model.compose(
                loaded,
                rows=b,
                chunk=t,
                max_requests=capacity,
                max_context=limit,
                provider=recipe["provider"],
                phase=phase[0] if phase else None,
                **recipe["model_options"],
            )
            from akas.compilation.policy import lower

            graph, _ = lower(
                composed.graph,
                composed.commit,
                policy=recipe["policy"],
                provider=recipe["provider"],
                target=recipe["target"],
            )
            bounds.append(estimate(graph))
            graphs.append(
                (phase[0] if phase else "decode" if t == 1 else "prefill", graph)
            )
        needed = max(bound["largest_buffer_bytes"] for bound in bounds)
        memory = context.get("memory_bytes")
        # Raw graph sizes are upper estimates; actual allocations are measured below.
        metadata["memory_estimate"] = bounds
        metadata["memory_bound"] = memory_bound([g for _, g in graphs])
        available = (
            memory
            if memory is not None
            else job["device"].get("free_bytes", float("inf")) * 0.9
        )
        if metadata["memory_bound"]["persistent_bytes"] > available:
            raise MemoryError(
                "requested cache/request capacity cannot fit the available memory"
            )
        if memory is not None and needed > memory:
            raise MemoryError(
                "a required single buffer exceeds the requested GPU memory ceiling"
            )
        max_buffer_size = (
            needed if recipe["provider"] == "webgpu" and needed > 134217728 else None
        )
        status(output, "compiling", recipe=recipe["id"])
        from akas.optimization.incumbents import seed

        schedules, origins = seed(
            job["history"],
            graphs,
            provider=recipe["provider"],
            target=recipe["target"],
            checkpoint_sha256=job["checkpoint_sha256"],
            hardware=job["device"]["name"],
        )
        metadata["incumbents"] = origins
        with tensor.Device(
            context["device"],
            provider=recipe["provider"],
            max_buffer_size=max_buffer_size,
        ) as device:
            compile_checkpoint(
                model,
                context["checkpoint"],
                program,
                geometries=geometries,
                max_requests=capacity,
                max_context=limit,
                provider=recipe["provider"],
                target=recipe["target"],
                model_options=recipe["model_options"],
                policy=recipe["policy"],
                round_windows=windows,
                deferred=recipe["deferred"],
                search=not skip_autotune,
                search_settings=(
                    SearchSettings(budget=1, width=1, warmup=0, samples=1, repeats=1)
                    if recipe["trials"] == 1
                    else SearchSettings(budget=recipe["trials"])
                ),
                schedules=schedules,
                device=device,
            )
            status(output, "validating-and-measuring", recipe=recipe["id"])
            previous = (
                None
                if not job.get("reference")
                else json.loads(Path(job["reference"]).read_text())
            )
            evidence = (
                None
                if not job.get("evidence")
                else np.load(job["evidence"], allow_pickle=False)
            )
            try:
                with model(
                    context["checkpoint"], program, device, graphs=recipe["graphs"]
                ) as executor:
                    executor.prefill_group_size = recipe["prefill_requests"]
                    report["results"], features, statistics = evaluate(
                        executor, recipe, data, reference=previous, evidence=evidence
                    )
                metadata.update(statistics)
            finally:
                if evidence is not None:
                    evidence.close()
            metadata["device"] = dict(device.info)
        np.savez_compressed(root / "diagnostics.npz", **features)
        metadata["diagnostics_sha256"] = digest(root / "diagnostics.npz")
        resident = max(row["allocated_bytes"] for row in report["results"])
        if memory is not None and resident > memory:
            raise MemoryError(
                "measured resident allocations exceed the requested GPU memory ceiling"
            )
        descriptor = describe(
            context,
            recipe,
            program,
            checkpoint_sha256=job["checkpoint_sha256"],
            campaign_identity=job["identity"],
            run_id=recorder.id,
            max_buffer_size=max_buffer_size,
            hardware=job["device"],
        )
        write(root / "candidate.json", descriptor)
        status(output, "http-validation", recipe=recipe["id"])
        metadata["http"] = asyncio.run(
            http_benchmark(
                partial(service, root / "candidate.json", candidate=True),
                report["results"],
                vocab=loaded.config.vocab,
            )
        )
        report["metadata"]["cpu_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        descriptor["status"] = "qualified"
        write(root / "candidate.json", descriptor)
        result.update(
            status="passed",
            report=str(root / "report.json"),
            evidence=str(root / "diagnostics.npz"),
            deployment=str(root / "candidate.json"),
        )
    except BaseException as error:
        result.update(
            status="oom"
            if isinstance(error, MemoryError) or "out of memory" in str(error).lower()
            else "failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        report["metadata"]["error"] = result["error"]
        for row in report["results"]:
            row.update(status="error", error=result["error"])
        print(result["traceback"], flush=True)
    finally:
        if (root / "telemetry.json").exists():
            metadata["memory"] = json.loads((root / "telemetry.json").read_text())
            write(root / "memory.json", metadata["memory"])
            recorder.artifact(root / "memory.json", "memory.json")
        write(root / "report.json", report)
        recorder.artifact(root / "job.json", "job.json")
        if (root / "worker.log").exists():
            recorder.artifact(root / "worker.log", "worker.log")
        if (program / "program.json").exists():
            try:
                recorder.program(program)
            except (OSError, ValueError, KeyError) as error:
                result.update(status="failed", error="program evidence: " + str(error))
        record = recorder.update(
            report,
            final=True,
            status="passed" if result["status"] == "passed" else "failed",
        )
        if record["status"] == "invalidated" or (
            record["status"] != "passed" and result["status"] == "passed"
        ):
            result["status"] = record["status"]
        result["history_record"] = str(recorder.directory / "run.json")
    if result["status"] == "passed":
        for name in ("report", "evidence", "deployment"):
            result[name + "_sha256"] = digest(result[name])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    if os.name == "posix" and Path("/proc").exists():
        import ctypes
        import signal

        # A terminated supervisor must not leave its GPU worker running.
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)
        if os.getppid() != job.get("supervisor_pid", os.getppid()):
            raise SystemExit("supervisor exited before worker initialization")
    output = args.job.parent / "result.json"
    try:
        if job["action"] == "probe":
            from akas.optimization.planning import probe

            result = probe(job["context"], job["provider"])
        else:
            result = candidate(job, output)
    except BaseException as error:
        result = dict(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        print(result["traceback"], flush=True)
    write(output, result)


if __name__ == "__main__":
    main()
