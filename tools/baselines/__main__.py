"""Run one isolated, bounded baseline probe and retain its complete provenance."""

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from akas.benchmarking.history import Recorder, digest, write
from akas.benchmarking.workload import STANDARD
from akas.cli import checkpoint_digest


def model_definition(directory, name):
    matches = []
    for path in sorted(Path(directory).glob("*.json")):
        data = json.loads(path.read_text())
        if name == data["id"] or name in data.get("aliases", []):
            matches.append((path, data))
    if len(matches) != 1:
        raise ValueError("select one model definition by its ID or alias")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", choices=("vllm", "sglang", "llama"))
    parser.add_argument("model", help="ID or alias from benchmarks/models")
    parser.add_argument("--models", type=Path, default=Path("benchmarks/models"))
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--provider", choices=("cuda", "vulkan"), default="cuda")
    parser.add_argument("--format", choices=("f16", "q4_0", "q4_k_m"), default="q4_k_m")
    parser.add_argument("--history", type=Path, default=Path("benchmarks"))
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="enable graph execution and measure every standard shape",
    )
    parser.add_argument(
        "--workload",
        help="optional workload JSON; defaults to standard-v1 for optimized runs",
    )
    parser.add_argument("--repeats", type=int, default=None)
    args = parser.parse_args()
    if (
        args.seconds <= 0
        or (args.repeats is not None and args.repeats < 1)
        or (args.runtime != "llama" and args.provider != "cuda")
    ):
        parser.error("a positive deadline and compatible provider are required")
    if args.runtime == "llama" and args.mtp:
        parser.error("the llama.cpp probe currently measures AR")
    definition_path, model = model_definition(args.models, args.model)
    mode = "mtp" if args.mtp else "ar"
    if mode not in model["modes"]:
        parser.error("the model definition does not declare this generation mode")
    root = Path(__file__).resolve().parents[2]
    work = root / "build/baselines"
    work.mkdir(parents=True, exist_ok=True)
    weight_format = args.format if args.runtime == "llama" else "bf16"
    weight = model["weights"][weight_format]
    checkpoint = (root / weight["path"]).resolve()
    name = f"{args.runtime}-{model['id']}-{mode}-{args.provider}-{weight_format}"
    if args.optimized:
        name += "-optimized"
    raw, log, configuration = (
        work / f"{name}.{suffix}" for suffix in ("json", "log", "config.json")
    )
    python = (
        work / args.runtime / "bin/python"
        if args.runtime != "llama"
        else Path(sys.executable)
    )
    command = [
        str(python),
        "-u",
        "-m",
        f"tools.baselines.{args.runtime}",
        "--config",
        str(configuration),
        "--out",
        str(raw),
    ]
    detected = next(
        csv.reader(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap,uuid,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
        )
    )
    gpu_name, capability, gpu_uuid, driver, memory_mib = (
        value.strip() for value in detected
    )
    config = dict(
        device_name=gpu_name,
        checkpoint=str(checkpoint),
        mode=mode,
        provider=args.provider,
        output_tokens=STANDARD["output_tokens"],
        shapes=[[1, 32], [4, 32], [1, 2048]],
        runtime_options=model.get("runtime_options", {}).get(args.runtime, {}),
        optimized=args.optimized,
        warmup=1,
        repeats=args.repeats or (3 if args.optimized else 1),
    )
    if args.optimized or args.workload:
        from akas.optimization.context import workload

        requested = workload(args.workload or "standard-v1")
        config.update(
            shapes=[
                [b, p] for b in requested["batches"] for p in requested["prefix_tokens"]
            ],
            output_tokens=requested["output_tokens"],
        )
    hf_config = json.loads(
        (root / model["weights"]["bf16"]["path"] / "config.json").read_text()
    )
    config["vocab"] = hf_config.get("text_config", hf_config)["vocab_size"]
    if args.runtime == "llama":
        config.update(
            binary=str(work / f"llama.cpp/build-{args.provider}/bin/llama-server"),
            runtime_revision=subprocess.check_output(
                ["git", "-C", str(work / "llama.cpp"), "rev-parse", "HEAD"], text=True
            ).strip(),
        )
    metadata = dict(
        runtime=args.runtime,
        model=model["model"],
        mode=mode,
        provider=args.provider,
        target="sm_" + capability.replace(".", "")
        if args.provider == "cuda"
        else "vulkan",
        device=dict(
            name=gpu_name, uuid=gpu_uuid, driver=driver, memory_mib=int(memory_mib)
        ),
        purpose="baseline" if args.optimized else "preflight",
        command=command,
        protocol="baseline-burst-v3" if args.optimized else "baseline-preflight-v1",
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_digest(checkpoint),
        checkpoint_source=weight["source"],
        model_definition_sha256=digest(definition_path),
        environment={
            k: os.environ.get(k)
            for k in (
                "CUDA_HOME",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MAX_JOBS",
                "VK_ICD_FILENAMES",
                "LD_LIBRARY_PATH",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        warmup=1,
        repeats=config["repeats"],
        graphs=args.optimized if args.runtime != "llama" else None,
        weight_format=weight_format,
        kv_dtype="bfloat16" if args.runtime != "llama" else "float16",
        prompts="deterministic uniform model-vocabulary IDs; seed=17+B*4099+P",
        sampling=dict(
            temperature=0, ignore_eos=True, output_tokens=config["output_tokens"]
        ),
    )
    if args.runtime != "llama":
        frozen = subprocess.check_output(
            [
                str(python),
                "-c",
                "import importlib.metadata as m,json,hashlib,sys;d=m.distribution(sys.argv[1]);print(json.dumps(dict(versions={p.metadata['Name']:p.version for p in m.distributions()},record_sha256=hashlib.sha256(d.read_text('RECORD').encode()).hexdigest(),wheel=d.read_text('WHEEL'))))",
                args.runtime,
            ],
            text=True,
        )
        environment = json.loads(frozen)
        metadata["dependencies"] = environment.pop("versions")
        metadata["runtime_distribution"] = environment
    run = Recorder(
        args.history,
        metadata,
        kind="baseline",
        change="Measure optimized baseline execution"
        if args.optimized
        else "Qualify baseline execution; these probes are not tuned performance measurements",
    )
    run.artifact(definition_path, "model.json")
    write(configuration, config)
    run.artifact(configuration, "probe-config.json")
    raw.unlink(missing_ok=True)
    result = dict(schema="akas.benchmark.v1", metadata=metadata, results=[])
    process, status = None, "failed"
    try:
        with log.open("w") as output:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            until = time.monotonic() + args.seconds
            memory_samples = []
            while process.poll() is None:
                if time.monotonic() >= until:
                    raise subprocess.TimeoutExpired(command, args.seconds)
                measured = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                memory_samples.append(int(measured.splitlines()[0]) * 1024**2)
                metadata["sampled_peak_device_bytes"] = max(memory_samples)
                time.sleep(1)
            code = process.returncode
        data = json.loads(raw.read_text()) if raw.exists() else {}
        metadata.update(
            runtime_version=data.get("version"),
            runtime_revision=data.get("revision"),
            configuration=data.get("settings", data.get("command")),
            error=data.get("error"),
            returncode=code,
        )
        cases = {(r["batch"], r["prefix"]): r for r in data.get("cases", [])}
        for b, p in config["shapes"]:
            row = cases.get(
                (b, p),
                dict(
                    batch=b,
                    prefix=p,
                    output=config["output_tokens"],
                    status="failed",
                    error="worker did not complete this shape",
                ),
            )
            result["results"].append(
                dict(
                    case=dict(
                        batch=row["batch"],
                        prompt=row["prefix"],
                        generated=row["output"],
                        depth=0,
                        mixed=False,
                    ),
                    status="ok" if row["status"] == "passed" else "error",
                    samples=row.get("samples", []),
                    error=row.get("error"),
                    correctness=dict(
                        output_count=row.get("output_count", row["status"] == "passed"),
                        deterministic=row.get("deterministic"),
                        full_model_reference=False,
                    ),
                )
            )
        if args.mtp and data.get("status") == "passed":
            previous_path = work / (
                f"{args.runtime}-{model['id']}-ar-{args.provider}-{weight_format}"
                + ("-optimized" if args.optimized else "")
                + ".json"
            )
            previous = json.loads(previous_path.read_text())
            if len(previous["cases"]) != len(data["cases"]) or not all(
                (a["batch"], a["prefix"], a["output_ids"])
                == (b["batch"], b["prefix"], b["output_ids"])
                for a, b in zip(previous["cases"], data["cases"])
            ):
                raise ValueError("MTP output differs from the matching AR probe")
            metadata["greedy_ar_mtp_equal"] = True
        status = "passed" if code == 0 and data.get("status") == "passed" else "failed"
        if status == "failed" and not metadata.get("error"):
            metadata["error"] = (
                f"baseline worker exited with code {code}; inspect its retained log"
            )
    except subprocess.TimeoutExpired:
        status = "budget_exhausted"
        metadata["error"] = f"probe exceeded {args.seconds}s"
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if not result["results"] and raw.exists():
            partial = json.loads(raw.read_text())
            for row in partial.get("cases", []):
                result["results"].append(
                    dict(
                        case=dict(
                            batch=row["batch"],
                            prompt=row["prefix"],
                            generated=row["output"],
                            depth=0,
                            mixed=False,
                        ),
                        status="ok" if row["status"] == "passed" else "error",
                        samples=row.get("samples", []),
                        error=row.get("error"),
                    )
                )
        present = {(r["case"]["batch"], r["case"]["prompt"]) for r in result["results"]}
        for b, p in config["shapes"]:
            if (b, p) not in present:
                result["results"].append(
                    dict(
                        case=dict(
                            batch=b,
                            prompt=p,
                            generated=config["output_tokens"],
                            depth=0,
                            mixed=False,
                        ),
                        status="error",
                        error=metadata.get("error", "worker did not finish"),
                        samples=[],
                    )
                )
        for path in (raw, log):
            if path.exists():
                run.artifact(path, path.name)
        run.artifact(Path(__file__).with_name(args.runtime + ".py"), "probe.py")
        if args.runtime == "llama":
            binary = Path(config["binary"])
            if binary.is_file():
                metadata["binary_sha256"] = digest(binary)
            cache = binary.parents[1] / "CMakeCache.txt"
            if cache.is_file():
                run.artifact(cache, "CMakeCache.txt")
        record = run.update(result, final=True, status=status)
        print(record["id"], record["status"], name, flush=True)
    if record["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
