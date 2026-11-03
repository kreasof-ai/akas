"""Run a declared model matrix serially, with cumulative campaign and per-model budgets."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from akas.benchmarking.history import source_identity, utc, write
from akas.optimization.context import (
    Context,
    discover,
    duration,
    identity,
    workload,
    measurement_limits,
)
from akas.optimization.supervisor import lock, optimize


def declarations(spec, root):
    return {
        name: json.loads((root / "benchmarks/models" / (name + ".json")).read_text())
        for name in spec["models"]
    }


def entries(spec, root):
    result = []
    for name, model in declarations(spec, root).items():
        for precision in spec["formats"]:
            path = (root / model["weights"][precision]["path"]).resolve()
            try:
                definition, error = discover(path), None
            except (ValueError, OSError) as failure:
                definition, error = None, f"{type(failure).__name__}: {failure}"
            for backend in spec["backends"]:
                for cache in spec["caches"]:
                    for mode in model["modes"]:
                        key = "-".join((name, precision, backend, cache, mode))
                        if error:
                            result.append(
                                dict(
                                    id=key,
                                    status="failed",
                                    error=error,
                                    checkpoint=str(path),
                                )
                            )
                            continue
                        context = Context(
                            str(path),
                            definition,
                            workload(spec["workload"]),
                            backend=backend,
                            kv_dtype=None if cache == "default" else cache,
                            mode=mode,
                        )
                        result.append(dict(id=key, context=asdict(context)))
    return result


def run(specification, out, *, budget, slice_seconds, history):
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(Path(specification).read_text())
    models = declarations(spec, root)
    frozen = dict(
        spec=spec,
        models=models,
        workload=workload(spec["workload"]),
        measurement_limits=measurement_limits(),
        source=source_identity(root)["tree_sha256"],
    )
    baseline_lock = None
    if spec.get("baseline_lock"):
        baseline_lock = json.loads((root / spec["baseline_lock"]).read_text())
        if baseline_lock.get("schema") != "akas.baseline-lock.v1":
            raise ValueError("unsupported baseline lock schema")
        frozen["baseline_lock"] = baseline_lock
    fingerprint = identity(frozen)
    cap = duration(spec.get("configuration_budget", "1h"))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Campaign directory: {out}", flush=True)
    with lock(out / ".lock"):
        path = out / "campaign.json"
        if path.exists():
            state = json.loads(path.read_text())
            if state["identity"] != fingerprint or state["budget_seconds"] != budget:
                raise ValueError(
                    "campaign source, specification, model declarations, measurement limits or total budget changed; use a new output directory"
                )
        else:
            print(
                "Indexing checkpoints and building the requested matrix...", flush=True
            )
            state = dict(
                schema="akas.matrix-campaign.v1",
                identity=fingerprint,
                specification=spec,
                models=models,
                measurement_limits=frozen["measurement_limits"],
                source=source_identity(root),
                created_at=utc(),
                budget_seconds=budget,
                configuration_budget_seconds=cap,
                spent_seconds=0,
                entries=entries(spec, root),
            )
            if baseline_lock is not None:
                state["baseline_lock"] = baseline_lock
        # A hard-killed coordinator cannot know how much of its last grant was
        # used. Charge the reserved grant rather than silently reset that time.
        active = state.pop("active", None)
        if active:
            state["spent_seconds"] = max(
                state["spent_seconds"], active["spent_before"] + active["allowance"]
            )
            entry = next(e for e in state["entries"] if e["id"] == active["id"])
            entry["spent_seconds"] = entry.get("spent_seconds", 0) + active["allowance"]
            entry["recovered_interruption"] = utc()
        previous_spent = state["spent_seconds"]

        def save():
            state.update(
                spent_seconds=previous_spent + time.monotonic() - started,
                updated_at=utc(),
            )
            write(path, state)

        state["status"] = "running"
        save()
        try:
            import os

            skip_autotune = os.environ.get("AKAS_SKIP_AUTOTUNE") == "1"
            while skip_autotune or (
                time.monotonic() - started + previous_spent < budget
            ):
                if (
                    source_identity(root)["tree_sha256"] != frozen["source"]
                    or declarations(spec, root) != models
                    or json.loads(Path(specification).read_text()) != spec
                    or workload(spec["workload"]) != frozen["workload"]
                ):
                    state["status"] = "invalidated"
                    raise ValueError("campaign inputs changed during execution")
                terminal = (
                    ("complete", "unsupported", "failed", "budget_exhausted")
                    if skip_autotune
                    else ("complete", "unsupported", "failed")
                )
                unfinished = [
                    e for e in state["entries"] if e.get("status") not in terminal
                ]
                pending = [
                    e
                    for e in unfinished
                    if skip_autotune or e.get("spent_seconds", 0) < cap
                ]
                if not pending:
                    state["status"] = (
                        "configuration_budgets_exhausted" if unfinished else "complete"
                    )
                    break
                # Give each configuration a first pass before deepening one.
                entry = min(pending, key=lambda e: e.get("spent_seconds", 0))
                remaining = (
                    budget
                    if skip_autotune
                    else budget - (time.monotonic() - started + previous_spent)
                )
                allowance = (
                    slice_seconds
                    if skip_autotune
                    else min(
                        slice_seconds, remaining, cap - entry.get("spent_seconds", 0)
                    )
                )
                print(entry["id"], f"allowance={allowance:.1f}s", flush=True)
                save()
                state["active"] = dict(
                    id=entry["id"],
                    allowance=allowance,
                    spent_before=state["spent_seconds"],
                    started_at=utc(),
                )
                write(path, state)
                tick = time.monotonic()
                try:
                    result = optimize(
                        Context(**entry["context"]),
                        out / entry["id"],
                        budget=allowance,
                        history=history,
                    )
                    entry.update(
                        status=result["status"],
                        deployment=result.get("deployment"),
                        winner=result.get("winner"),
                    )
                except KeyboardInterrupt:
                    entry["status"] = "interrupted"
                    raise
                except Exception as error:
                    entry.update(
                        status="failed", error=f"{type(error).__name__}: {error}"
                    )
                finally:
                    entry["spent_seconds"] = (
                        entry.get("spent_seconds", 0) + time.monotonic() - tick
                    )
                    entry["updated_at"] = utc()
                    state.pop("active", None)
                    save()
            else:
                state["status"] = "budget_exhausted"
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            raise
        finally:
            save()
            import sys

            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from tools.campaign_report import export

            destination = export(path, history=history)
            print("Campaign report:", destination / "README.md", flush=True)
        return state


def modal_command(
    spec,
    out,
    *,
    budget,
    slice_duration="30m",
    history="/data/standard-benchmarks",
    no_autotune=False,
    shape_timeout=None,
    python="python",
):
    command = [
        python,
        "-u",
        "-m",
        "tools.campaign",
        "--spec",
        str(spec),
        "--out",
        str(out),
        "--budget",
        str(budget),
        "--slice",
        str(slice_duration),
        "--history",
        str(history),
    ]
    if no_autotune:
        command.append("--no-autotune")
    if shape_timeout is not None:
        command.extend(["--shape-timeout", str(shape_timeout)])
    return command


def run_modal(args):
    import subprocess
    import modal

    root = Path(__file__).resolve().parents[1]
    tensor_root = root.parent / "tensor"
    volume = modal.Volume.from_name(args.modal_volume)
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12"
        )
        .apt_install("git", "g++", "curl", "libnuma1")
        .env(
            {
                "HF_HOME": "/data/hf",
                "CUDA_HOME": "/usr/local/cuda",
                "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "1",
                "MAX_JOBS": "4",
                "PYTHONPATH": "/workspace/akas/src:/workspace/akas:/workspace/tensor/src",
                "TENSOR_NVRTC_HOME": "/usr/local/cuda",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        .uv_pip_install(
            "numpy==2.5.3",
            "tilelang==0.1.14",
            "apache-tvm-ffi==0.1.12",
            "torch",
            "safetensors",
            "huggingface-hub",
            "aiohttp",
            "regex",
        )
        .add_local_dir(root / "src", "/workspace/akas/src")
        .add_local_dir(root / "tools", "/workspace/akas/tools")
        .add_local_dir(tensor_root / "src", "/workspace/tensor/src")
        .add_local_dir(
            root / "benchmarks",
            "/workspace/akas/benchmarks",
            ignore=["**/*.tar.gz", "**/*.npz", "**/*.json.gz"],
        )
    )
    app = modal.App("akas-campaign")
    zero_budget = str(args.budget).strip() in ("0", "0s", "0m", "0h")
    effective_budget = args.slice if zero_budget else args.budget
    cmd = modal_command(
        args.spec,
        args.out,
        budget=effective_budget,
        slice_duration=args.slice,
        history=args.history,
        no_autotune=args.no_autotune or zero_budget,
        shape_timeout=args.shape_timeout,
    )
    timeout = max(600, int(duration(effective_budget)) + 300)

    @app.function(
        image=image,
        gpu=args.modal_gpu,
        volumes={"/data": volume},
        timeout=timeout,
        cpu=4,
        memory=65536,
        scaledown_window=2,
        max_containers=1,
        serialized=True,
    )
    def execute_remote(command: list[str]):
        import os

        for rel in ("build/checkpoint", "build/checkpoints/lfm2.5-24b-a2b-fp8"):
            ckpt = Path("/workspace/akas") / rel
            if not ckpt.exists() and Path("/data/checkpoint").exists():
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                ckpt.symlink_to("/data/checkpoint")
        env = dict(os.environ)
        env.setdefault(
            "AKAS_CHECKPOINT_SHA256",
            "dff2336439834c4ce4e2c6f0f7b6cd829987f7e9d57ec8074354e0b2cacb2830",
        )
        for candidate_prog in (
            "/data/programs/builtin-winners-v4",
            "/data/programs/builtin-winners-v3",
            "/data/programs/builtin-winners-v2",
            "/data/programs/builtin-winners-v1",
            "/data/programs/builtin-measured-225932",
        ):
            if Path(candidate_prog, "program.json").is_file():
                env.setdefault("AKAS_PRECOMPILED_PROGRAM", candidate_prog)
                break
        env.setdefault("AKAS_SKIP_HTTP_BENCHMARK", "1")
        proc = subprocess.run(command, cwd="/workspace/akas", env=env)
        for log_file in sorted(Path(str(args.out)).rglob("worker.log")):
            print(f"=== {log_file} ===", flush=True)
            print(log_file.read_text(), flush=True)
        volume.commit()
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    with modal.enable_output(), app.run():
        execute_remote.remote(cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=Path("benchmarks/campaigns/standard-v1.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--budget",
        required=True,
        help="cumulative allowance, e.g. 4d; downtime is excluded",
    )
    parser.add_argument(
        "--slice",
        default="30m",
        help="maximum grant before moving to the next configuration",
    )
    parser.add_argument("--history", type=Path, default=Path("benchmarks"))
    parser.add_argument(
        "--list",
        action="store_true",
        help="show the requested matrix without running GPU work",
    )
    parser.add_argument(
        "--no-autotune",
        action="store_true",
        help="qualify the default schedule and first graph recipe without beam search",
    )
    parser.add_argument(
        "--shape-timeout",
        default=None,
        help="maximum duration per measurement repetition, including diagnostics, e.g. 2m",
    )
    parser.add_argument(
        "--modal-gpu",
        default=None,
        help="run the campaign inside a single-container Modal GPU worker, e.g. H200",
    )
    parser.add_argument(
        "--modal-volume",
        default="akas-h200-evidence",
        help="Modal volume mounted at /data when --modal-gpu is used",
    )
    args = parser.parse_args()
    if args.modal_gpu:
        run_modal(args)
        return
    zero_budget = str(args.budget).strip() in ("0", "0s", "0m", "0h")
    if args.no_autotune or zero_budget:
        import os

        os.environ["AKAS_SKIP_AUTOTUNE"] = "1"
        os.environ.setdefault("AKAS_SKIP_HTTP_BENCHMARK", "1")
    if args.shape_timeout is not None:
        import os

        os.environ["AKAS_SHAPE_TIMEOUT"] = str(duration(args.shape_timeout))
    if args.list:
        print(
            json.dumps(entries(json.loads(args.spec.read_text()), Path.cwd()), indent=2)
        )
        return
    effective_budget = duration(args.slice) if zero_budget else duration(args.budget)
    state = run(
        args.spec,
        args.out,
        budget=effective_budget,
        slice_seconds=duration(args.slice),
        history=args.history,
    )
    print(
        json.dumps(
            dict(
                status=state["status"],
                spent_seconds=state["spent_seconds"],
                configurations=len(state["entries"]),
            ),
            indent=2,
        )
    )
    if any(e.get("status") == "failed" for e in state["entries"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
