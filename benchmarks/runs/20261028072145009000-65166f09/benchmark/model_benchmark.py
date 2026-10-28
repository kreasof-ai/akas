"""Matched single-user CUDA context sweeps with checksummed runs and reports.

Use compile, akas, llama, then report. Archive through tools.benchmark_archive;
model files and compiled programs stay under build/, payloads stay off main.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import urllib.request

from akas.benchmarking.history import Recorder, digest, utc, write
from akas.benchmarking.workload import prompt_digest

PREFIXES = [32, 512, 2048, 8192, 16384, 32768, 65536, 130944]
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Profile:
    id: str
    model: str
    source: dict
    checkpoint_sha256: str
    definition: str
    weight_format: str
    file_type: int
    checkpoint: Path
    program: Path
    binary: Path
    vocab: int
    optimized_options: dict
    trials: int = 1
    chunk: int = 256

    @property
    def model_class(self):
        module, name = self.definition.split(":")
        return getattr(import_module(module), name)


def prompts(prefix, vocab=131072):
    from tools.baselines.common import synthetic_prompts

    return synthetic_prompts(vocab, 1, prefix)


def compile_program(args):
    from akas.compilation import compile_checkpoint
    from akas.compilation.tuning import SearchSettings

    compile_checkpoint(
        args.profile.model_class,
        args.checkpoint,
        args.program,
        checkpoint_options={"tokenizer": False},
        geometries=[(1, n) for n in sorted({1, 32, min(512, args.chunk), args.chunk})],
        max_requests=1,
        max_context=131072,
        model_options=dict(
            prefill_chunk=args.chunk,
            **(args.profile.optimized_options if args.optimized else {}),
        ),
        search_settings=SearchSettings(budget=args.trials, samples=3, repeats=3),
    )
    write(
        args.out,
        dict(
            program=str(args.program.resolve()),
            program_sha256=digest(args.program / "program.json"),
        ),
    )


def record_run(args, runtime, measure):

    loaded = args.profile.model_class.checkpoint_type(args.checkpoint, tokenizer=False)
    if (
        loaded.gguf is None
        or loaded.gguf.metadata.get("general.file_type") != args.profile.file_type
    ):
        raise ValueError("checkpoint format differs from the benchmark profile")
    checkpoint_sha256 = digest(args.checkpoint)
    if checkpoint_sha256 != args.profile.checkpoint_sha256:
        raise ValueError("checkpoint differs from the pinned benchmark profile")
    qualified_small_model = False
    if args.qualification_run:
        from tools.campaign_report import record

        qualified = record(args.history, args.qualification_run)
        if (
            not qualified
            or qualified["status"] != "passed"
            or qualified["metadata"].get("purpose") != "model-qualification"
        ):
            raise ValueError("qualification evidence must be a sealed passed model run")
        qualified_small_model = True
    meta = dict(
        runtime=runtime,
        purpose="benchmark",
        model=args.profile.model,
        mode="ar",
        weight_format=args.profile.weight_format,
        provider="cuda",
        target="sm_89",
        protocol="native-token-step-v2"
        if runtime == "akas"
        else "llama-server-eval-v1",
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=checkpoint_sha256,
        qualification_run_id=args.qualification_run,
        independent_small_model_reference=qualified_small_model,
        checkpoint_source=args.profile.source,
        kv_dtype="float16",
        output_tokens=128,
        ignore_eos=True,
        warmup=args.warmup,
        repeats=args.repeats,
        max_context=131072,
        prefill_chunk=args.chunk,
        initialization_requests=1,
        compilation_trials_allowance=args.trials,
        sampling=dict(
            temperature=0,
            repeat_penalty=1,
            presence_penalty=0,
            frequency_penalty=0,
            ignore_eos=True,
        ),
        input="Identical deterministic synthetic token IDs; no prefix reuse; empty cache per request",
        qualification="Independent operation references and small-model equations. Full pretrained-model agreement is reported separately from timing.",
        timing="AKAS synchronized native token phases; llama.cpp server eval phases. Loading excluded. Phase boundaries differ.",
    )
    recorder = Recorder(
        args.history,
        meta,
        kind="baseline" if runtime == "llama" else "configuration",
        change=f"{args.profile.model} {args.profile.weight_format} matched CUDA context sweep",
    )
    report = dict(schema="akas.benchmark.v1", metadata=meta, results=[])
    manifest = recorder.directory / "input-tokens.json"
    write(
        manifest,
        dict(
            schema="akas.synthetic-context-inputs.v1",
            cases=[
                dict(
                    prefix_tokens=p,
                    token_ids=prompts(p, args.profile.vocab)[0],
                    prompt_sha256=prompt_digest(prompts(p, args.profile.vocab)),
                )
                for p in args.prefixes
            ],
        ),
    )
    meta["prompt_manifest_sha256"] = digest(manifest)
    recorder.artifact(manifest, "benchmark/input-tokens.json")
    print("RUN", recorder.id, flush=True)
    try:
        measure(meta, report, recorder)
    except BaseException:
        meta["error"] = traceback.format_exc()
        print(meta["error"], flush=True)
    finally:
        present = {r["case"]["prompt"] for r in report["results"]}
        for p in args.prefixes:
            if p not in present:
                report["results"].append(
                    dict(
                        case=dict(
                            batch=1, prompt=p, generated=128, depth=0, mixed=False
                        ),
                        status="error",
                        samples=[],
                        error=meta.get("error", "shape did not finish"),
                    )
                )
        recorder.artifact(__file__, "benchmark/model_benchmark.py")
        recorder.artifact(
            Path("tools") / (args.profile.id.split("-")[0] + "_benchmark.py"),
            "benchmark/profile.py",
        )
        final = recorder.update(report, final=True)
        write(args.out, dict(run_id=recorder.id, status=final["status"]))
        print("RETAINED", recorder.id, final["status"], flush=True)
    if final["status"] != "passed":
        raise SystemExit(1)


def akas(args):
    def measure(meta, report, recorder):
        import numpy as np
        import tensor
        from akas.benchmarking.cases import Case
        from akas.benchmarking.session import Session
        from akas.benchmarking.runner import run

        program = json.loads((args.program / "program.json").read_text())
        from akas.graph import Graph
        from akas.kernels.registry import graph_outputs

        for phase in program["phases"]:
            manifest = json.loads((args.program / phase["path"]).read_text())
            graph = Graph.from_metadata(manifest["forward"])
            if any(
                "mapping" in (graph_outputs(call, contract=graph.contract) or ())
                for call in graph.calls
            ):
                raise ValueError(
                    "program has an invalid output ABI for request mapping; recompile"
                )
        if program.get("model_options", {}).get("prefill_weight_cache") or program.get(
            "upload_options", {}
        ).get("prefill_cache"):
            raise ValueError(
                "FP16 weight-cache programs are unsupported; recompile with the current packed-weight model"
            )
        with (
            tensor.Device() as device,
            args.profile.model_class(
                args.checkpoint,
                args.program,
                device,
                checkpoint_options={"tokenizer": False},
                graphs=True,
            ) as model,
        ):
            meta["device"] = dict(device.info)
            meta["model_options"] = model.program["model_options"]
            meta["graphs"] = True
            meta["quantized_weight_storage"] = (
                "packed GGML coefficients; exact layout repacking when configured; no persistent dequantized weight copy"
            )
            meta["allocated_weight_bytes"] = sum(
                v.nbytes
                for v in model.batches.weights.values()
                if getattr(v, "memory_kind", "device") != "mapped_host"
            )
            meta["mapped_host_weight_bytes"] = sum(
                v.nbytes
                for v in model.batches.weights.values()
                if getattr(v, "memory_kind", "device") == "mapped_host"
            )
            recorder.program(args.program)
            if not recorder.record["search"]:
                raise ValueError("missing operation qualification")
            for phase in model.program["phases"]:
                model.batches.select(1, phase["chunk"], phase=phase["phase"])
            device.synchronize()
            meta["allocated_state_bytes"] = sum(
                v.nbytes for v in model.batches.states.values()
            )
            meta["phase_loading"] = "all phases preloaded before timing"
            with Session(model, batch=1, chunk=args.chunk) as session:
                run(
                    session,
                    Case(1, 32, 4),
                    prompts=prompts(32, args.profile.vocab),
                    warmup=0,
                    repeats=1,
                )
                for prefix in args.prefixes:
                    print("MEASURE AKAS", prefix, flush=True)
                    row = run(
                        session,
                        Case(1, prefix, 128),
                        prompts=prompts(prefix, args.profile.vocab),
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    # Recapture from an empty request outside timing. Rewinding only
                    # positions cannot restore recurrent convolution/SSM histories.
                    session.reset()
                    output = [
                        int(session.prefill(prompts(prefix, args.profile.vocab))[0])
                    ]
                    for _ in range(127):
                        output.append(int(session.step([output[-1]])[0]))
                    finite = bool(np.isfinite(session.read_feature("logits")).all())
                    if not finite:
                        raise ValueError("nonfinite model logits")
                    row.update(
                        output_ids=[output],
                        correctness=dict(
                            output_count=True,
                            finite_logits=True,
                            full_model_reference=False,
                            independent_small_model_reference=meta[
                                "independent_small_model_reference"
                            ],
                        ),
                    )
                    report["results"].append(row)
                    recorder.update(report)
                    print(
                        "RESULT AKAS",
                        prefix,
                        row["summary"]["prefill_tokens_per_second"]["median"],
                        row["summary"]["decode_tokens_per_second"]["median"],
                        flush=True,
                    )

    record_run(args, "akas", measure)


def request(base, path, data=None):
    req = urllib.request.Request(
        base + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=7200) as response:
        return json.load(response)


def llama(args):
    def measure(meta, report, recorder):
        meta["runtime_version"] = subprocess.check_output(
            [str(args.binary), "--version"], stderr=subprocess.STDOUT, text=True
        ).strip()
        meta["binary_sha256"] = digest(args.binary)
        revision = subprocess.check_output(
            ["git", "-C", str(args.binary.parent.parent.parent), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if revision[:7] not in meta["runtime_version"]:
            raise ValueError("baseline binary/revision mismatch")
        meta["runtime_revision"] = revision
        meta["runtime_files"] = {
            p.name: digest(p) for p in args.binary.parent.glob("*.so*")
        }
        listing = subprocess.check_output(
            [str(args.binary), "--list-devices"], stderr=subprocess.STDOUT, text=True
        )
        if "CUDA0:" not in listing:
            raise ValueError("CUDA0 absent from llama.cpp")
        import tensor

        with tensor.Device() as device:
            meta["device"] = dict(device.info, device_listing=listing)
        if meta["device"]["name"] not in listing:
            raise ValueError("baseline GPU mismatch")
        command = [
            str(args.binary),
            "-m",
            str(args.checkpoint),
            "-ngl",
            "99",
            "--device",
            "CUDA0",
            "-c",
            "131072",
            "-np",
            "1",
            "-b",
            "2048",
            "-ub",
            "512",
            "-t",
            "2",
            "-fa",
            "on",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--cache-ram",
            "0",
            "--ctx-checkpoints",
            "0",
            "--no-cache-idle-slots",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--no-webui",
        ]
        meta["configuration"] = command
        meta["prefill_chunk"] = 512
        log_path = args.out.with_suffix(".server.log")
        base = f"http://127.0.0.1:{args.port}"
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                until = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("llama-server exited")
                    try:
                        if request(base, "/health").get("status") == "ok":
                            break
                    except Exception:
                        if time.monotonic() > until:
                            raise TimeoutError("llama startup")
                        time.sleep(0.25)

                def invoke(prefix, count=128):
                    return request(
                        base,
                        "/completion",
                        dict(
                            prompt=prompts(prefix, args.profile.vocab)[0],
                            n_predict=count,
                            temperature=0,
                            repeat_penalty=1,
                            presence_penalty=0,
                            frequency_penalty=0,
                            seed=17,
                            ignore_eos=True,
                            cache_prompt=False,
                            return_tokens=True,
                        ),
                    )

                invoke(32, 4)
                for prefix in args.prefixes:
                    print("MEASURE LLAMA", prefix, flush=True)
                    samples, outputs = [], []
                    for iteration in range(args.warmup + args.repeats):
                        start = time.perf_counter()
                        response = invoke(prefix)
                        elapsed = time.perf_counter() - start
                        t = response["timings"]
                        if (
                            response["tokens_predicted"] != 128
                            or len(response["tokens"]) != 128
                            or t["prompt_n"] != prefix
                        ):
                            raise ValueError("incomplete output or reused prompt")
                        if iteration >= args.warmup:
                            samples.append(
                                dict(
                                    phase_protocol="llama-server-eval-v1",
                                    prefill_tokens_per_second=prefix
                                    / (t["prompt_ms"] / 1000),
                                    decode_tokens_per_second=127
                                    / (t["predicted_ms"] / 1000),
                                    output_tokens_per_second=128 / elapsed,
                                    wall_seconds=elapsed,
                                    response=response,
                                )
                            )
                            outputs.append(response["tokens"])
                    stable = all(o == outputs[0] for o in outputs)
                    row = dict(
                        case=dict(
                            batch=1, prompt=prefix, generated=128, depth=0, mixed=False
                        ),
                        status="ok" if stable else "error",
                        samples=samples,
                        output_ids=[outputs[0]],
                        prompt_sha256=prompt_digest(
                            prompts(prefix, args.profile.vocab)
                        ),
                        correctness=dict(output_count=True, deterministic=stable),
                    )
                    report["results"].append(row)
                    recorder.update(report)
                    print(
                        "RESULT LLAMA",
                        prefix,
                        samples[-1]["prefill_tokens_per_second"],
                        samples[-1]["decode_tokens_per_second"],
                        flush=True,
                    )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                recorder.artifact(log_path, "baseline/server.log")

    record_run(args, "llama", measure)


def phase_parity(native, baseline):
    """Accept only a complete matched sweep with both phase medians at parity."""
    import statistics

    left = {r["case"]["prompt"]: r for r in native["results"]}
    right = {r["case"]["prompt"]: r for r in baseline["results"]}
    if set(left) != set(right) or not left:
        raise ValueError("parity requires identical nonempty prefix sets")
    rows = []
    for prefix, a in left.items():
        b = right[prefix]
        if a["status"] != "ok" or b["status"] != "ok":
            raise ValueError("parity requires successful measurements")
        row = dict(prefix_tokens=prefix)
        for phase in ("prefill", "decode"):
            key = phase + "_tokens_per_second"
            x = statistics.median(s[key] for s in a["samples"])
            y = statistics.median(s[key] for s in b["samples"])
            row[phase] = dict(akas=x, llama=y, ratio=x / y, parity_or_better=x >= y)
        rows.append(row)
    return dict(
        criterion="AKAS median >= llama.cpp median for both phases at every matched prefix",
        passed=all(
            r[p]["parity_or_better"] for r in rows for p in ("prefill", "decode")
        ),
        prefixes=len(rows),
        rows=rows,
    )


def report(args):
    from tools.campaign_report import (
        SCHEMA,
        measurement,
        metadata_for,
        record,
        render,
        report_directory,
    )

    records = [record(args.history, r) for r in args.runs]
    if len(records) != 2 or any(not r or r["status"] != "passed" for r in records):
        raise ValueError("select one sealed passed AKAS run and one llama run")
    a = next(r for r in records if r["metadata"]["runtime"] == "akas")
    b = next(r for r in records if r["metadata"]["runtime"] == "llama")
    for key in (
        "checkpoint_sha256",
        "prompt_manifest_sha256",
        "output_tokens",
        "kv_dtype",
        "warmup",
        "repeats",
    ):
        if a["metadata"][key] != b["metadata"][key]:
            raise ValueError("mismatched " + key)
    workload = dict(
        batches=[1],
        prefix_tokens=[r["case"]["prompt"] for r in a["results"]],
        output_tokens=128,
    )
    if workload["prefix_tokens"] != [r["case"]["prompt"] for r in b["results"]]:
        raise ValueError("shape mismatch")
    aid = args.profile.id + f"-{args.profile.weight_format}-cuda-float16-ar"
    bid = "baseline-" + args.profile.id + f"-llama-cuda-{args.profile.weight_format}-ar"
    spec = dict(
        schema="akas.matrix-campaign.v1",
        models=[args.profile.id],
        formats=[args.profile.weight_format],
        backends=["cuda"],
        caches=["float16"],
        workload=workload,
        baselines=[
            dict(
                runtime="llama",
                provider="cuda",
                weight_format=args.profile.weight_format,
                mode="ar",
            )
        ],
    )
    state = dict(
        schema="akas.matrix-campaign.v1",
        identity=hashlib.sha256(
            json.dumps(dict(spec=spec, runs=args.runs), sort_keys=True).encode()
        ).hexdigest(),
        specification=spec,
        models={
            args.profile.id: dict(
                id=args.profile.id, model=args.profile.model, modes=["ar"]
            )
        },
        created_at=min(r["started_at"] for r in records),
        updated_at=utc(),
        status="completed",
        source=a["source"],
        entries=[dict(id=aid, status="passed", context=dict(workload=workload))],
        spent_seconds=sum(
            (
                datetime.fromisoformat(r["completed_at"])
                - datetime.fromisoformat(r["started_at"])
            ).total_seconds()
            for r in records
        ),
    )
    profiles = []
    for data, pid, kind in ((a, aid, "akas"), (b, bid, "baseline")):
        m = data["metadata"]
        profiles.append(
            dict(
                id=pid,
                kind=kind,
                runtime=m["runtime"],
                model_id=args.profile.id,
                model=args.profile.model,
                provider="cuda",
                weight_format=args.profile.weight_format,
                cache_request="float16",
                mode="ar",
                status="passed",
                error=None,
                selected_run_id=data["id"],
                qualifying_status="passed",
                kv_dtype="float16",
                spent_seconds=(
                    datetime.fromisoformat(data["completed_at"])
                    - datetime.fromisoformat(data["started_at"])
                ).total_seconds(),
                recipe=dict(
                    chunk=m["prefill_chunk"],
                    prefill_requests=1,
                    graphs=True,
                    trials=m.get("compilation_trials_allowance"),
                ),
                measurements=[
                    measurement(row, kind, True, data["id"], m["protocol"])
                    for row in data["results"]
                ],
            )
        )
    args.out.mkdir(parents=True, exist_ok=True)
    campaign = args.out / "campaign.json"
    write(campaign, state)
    acceptance = phase_parity(a, b)
    comparisons = []
    for left, right in zip(a["results"], b["results"]):
        if left["prompt_sha256"] != right["prompt_sha256"]:
            raise ValueError("prompt mismatch")
        x, y = left["output_ids"][0], right["output_ids"][0]
        comparisons.append(
            dict(
                prefix=left["case"]["prompt"],
                output=128,
                exact=x == y,
                common_prefix_tokens=next(
                    (i for i, (v, w) in enumerate(zip(x, y)) if v != w), 128
                ),
                native_output_ids=x,
                baseline_output_ids=y,
            )
        )
    data = dict(
        schema=SCHEMA,
        generated_at=utc(),
        campaign=state,
        campaign_sha256=digest(campaign),
        workload=workload,
        profiles=profiles,
        attempts=[],
        evidence={r["id"]: metadata_for(r) for r in records},
        schedules={
            aid: dict(run_id=a["id"], recipe=profiles[0]["recipe"], phases=a["search"])
        },
        output_comparisons=comparisons,
        performance_acceptance=acceptance,
    )
    destination = report_directory(state, args.history)
    render(data, destination, args.history)
    write(destination / "PARITY.json", acceptance)
    print(destination / "README.md")
    print("PHASE PARITY", "passed" if acceptance["passed"] else "failed")


def main(profile):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("compile", "akas", "llama", "report"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=profile.checkpoint,
    )
    parser.add_argument("--program", type=Path, default=profile.program)
    parser.add_argument(
        "--binary",
        type=Path,
        default=profile.binary,
    )
    parser.add_argument("--history", type=Path, default=Path("benchmarks"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefixes", type=int, nargs="+", default=PREFIXES)
    parser.add_argument("--chunk", type=int, default=profile.chunk)
    parser.add_argument("--trials", type=int, default=profile.trials)
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="compile with tiled packed weights, compact RoPE, host input embedding and optimized attention",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--port", type=int, default=8826)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--qualification-run")
    args = parser.parse_args()
    args.profile = profile
    if (
        args.chunk < 128
        or args.repeats < 1
        or args.warmup < 0
        or not args.prefixes
        or len(set(args.prefixes)) != len(args.prefixes)
        or min(args.prefixes) < 1
        or max(args.prefixes) > 130944
    ):
        parser.error("invalid chunk, repetitions or context prefixes")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint = args.checkpoint.resolve()
    args.binary = args.binary.resolve()
    globals()["compile_program" if args.action == "compile" else args.action](args)
