"""Compile and measure a matched B1 Vulkan context sweep on Windows or Linux.

AKAS uses WebGPU's Vulkan adapter; llama.cpp uses its direct Vulkan backend.
Native AKAS and llama-server phase rates have different timing boundaries.
Keep the shared token manifest, raw responses and qualified schedules in history.
"""

import argparse
from contextlib import contextmanager
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
CHECKPOINT_SHA256 = "e1a61bf937bc60726e18626e97f7ee9bfd2574d95744c2ed909de98b78006fbe"


@contextmanager
def vulkan_device():
    import tensor

    device = tensor.Device(provider="webgpu", max_buffer_size=1024**3)
    if device.info["adapter"]["backend_type"] != "Vulkan":
        raise ValueError("select a hardware Vulkan adapter (WGPU_BACKEND_TYPE=Vulkan)")
    if device.info["adapter"]["adapter_type"] == "CPU":
        raise ValueError("software Vulkan is not a GPU measurement")
    # AMD's Windows driver reports UINT32_MAX here. In wgpu's device request,
    # that value means UNDEFINED and silently requests the default eight slots.
    # Request a finite subset instead; cache_write needs more than eight slots.
    limit = device._adapter.limits["max-storage-buffers-per-shader-stage"]
    if limit == 0xFFFFFFFF:
        device._adapter._limits["max-storage-buffers-per-shader-stage"] = 32
    with device:
        device.info["adapter_storage_buffers_reported"] = limit
        yield device


def inputs(args):
    data = json.loads(args.prompts.read_text(encoding="utf-8"))
    cases = {row["prefix_tokens"]: row for row in data["cases"]}
    selected = []
    for prefix in args.prefixes:
        row = cases[prefix]
        ids = row["token_ids"]
        if len(ids) != prefix or prompt_digest([ids]) != row["prompt_sha256"]:
            raise ValueError("prompt length or fingerprint differs from manifest")
        selected.append(row)
    return selected


def compile_program(args):
    from akas.compilation import compile_checkpoint
    from akas.compilation.tuning import SearchSettings
    from akas.models.lfm2.model import LFM2

    started = time.monotonic()
    geometries = [(1, n) for n in dict.fromkeys((1, 32, *args.prefill_chunks, args.chunk))]
    with vulkan_device() as device:
        print(json.dumps(device.info), flush=True)
        compile_checkpoint(
            LFM2, args.checkpoint, args.program,
            checkpoint_options={"tokenizer": False},
            geometries=geometries,
            max_requests=1, max_context=args.context,
            model_options={"kv_dtype": args.kv_dtype},
            provider="webgpu", device=device,
            search_settings=SearchSettings(budget=args.trials, samples=3, repeats=3),
        )
        write(args.out, dict(program=str(args.program.resolve()),
            program_sha256=digest(args.program / "program.json"),
            compilation_seconds=time.monotonic() - started, device=device.info,
            geometries=geometries, trials_allowance=args.trials, context=args.context))
    print("PROGRAM", args.program, flush=True)


def metadata(args, runtime):
    checkpoint_hash = digest(args.checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("checkpoint differs from the pinned standard Q4_0 file")
    return dict(
        # Real text must not be selected later as a synthetic matrix baseline.
        runtime=runtime, purpose="benchmark",
        model="LiquidAI/LFM2.5-2.6B", mode="ar", weight_format="q4_0",
        provider="vulkan" if runtime == "llama" else "webgpu",
        target="vulkan" if runtime == "llama" else "webgpu-portable-v1",
        protocol="llama-real-text-context-v1" if runtime == "llama" else "native-token-step-v2",
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_hash,
        checkpoint_source=dict(repo="LiquidAI/LFM2.5-2.6B-GGUF", revision=args.revision,
                               file="LFM2.5-2.6B-Q4_0.gguf"),
        kv_dtype=args.kv_dtype, output_tokens=args.output_tokens, ignore_eos=True,
        warmup=args.warmup, repeats=args.repeats, max_context=args.context,
        prefill_chunk=args.ubatch if runtime == "llama" else args.chunk,
        compilation_trials_allowance=args.trials if runtime == "akas" else None,
        initialization_requests=1,
        sampling=dict(temperature=0, repeat_penalty=1, presence_penalty=0,
                      frequency_penalty=0, ignore_eos=True),
        input="Identical retained real-text token IDs; no prefix reuse; empty cache per request",
        prompt_manifest_sha256=digest(args.prompts),
        qualification="Independent operation references for AKAS; finite timing and exact output counts. "
                      "Retained C4 tokens are a runtime probe, not a corpus quality evaluation. "
                      "Cross-runtime arithmetic and phase timing boundaries can differ.",
    )


def record_run(args, runtime, measure):
    meta = metadata(args, runtime)
    recorder = Recorder(args.history, meta, kind="baseline" if runtime == "llama" else "configuration",
                        parent=args.parent_run, change=args.change)
    report = dict(schema="akas.benchmark.v1", metadata=meta, results=[])
    recorder.artifact(args.prompts, "benchmark/text-workload.json")
    recorder.artifact(__file__, "benchmark/context_sweep.py")
    print("RUN", recorder.id, flush=True)
    passed = False
    try:
        measure(meta, report, recorder)
        passed = all(row["status"] == "ok" for row in report["results"])
    except BaseException:
        meta["error"] = traceback.format_exc()
        print(meta["error"], flush=True)
    finally:
        present = {row["case"]["prompt"] for row in report["results"]}
        for prefix in args.prefixes:
            if prefix not in present:
                report["results"].append(dict(case=dict(batch=1, prompt=prefix,
                    generated=args.output_tokens, depth=0, mixed=False), status="error",
                    samples=[], error=meta.get("error", "shape did not finish")))
        recorder.artifact(args.prompts, "benchmark/text-workload.json")
        recorder.artifact(__file__, "benchmark/context_sweep.py")
        saved = recorder.update(report, final=True, status="passed" if passed else "failed")
        write(args.out, dict(run_id=recorder.id, status=saved["status"]))
        print("RETAINED", recorder.id, saved["status"], flush=True)
    if saved["status"] != "passed":
        raise SystemExit(1)


def akas(args):
    from akas.benchmarking.cases import Case
    from akas.benchmarking.runner import run
    from akas.benchmarking.session import Session
    from akas.models.lfm2.model import LFM2

    cases = inputs(args)

    def measure(meta, report, recorder):
        meta["graphs"] = True
        from tensor.providers import _webgpu_native
        meta["native_submission_helper_sha256"] = digest(_webgpu_native.__file__)
        with vulkan_device() as device:
            with LFM2(args.checkpoint, args.program, device, graphs=True,
                      checkpoint_options={"tokenizer": False}) as model:
                recorder.program(args.program)
                if not recorder.record["search"]:
                    raise ValueError("program has no retained operation qualification evidence")
                # BatchBank loads pipelines and scratch lazily. Load every
                # geometry before the synchronized native timer, including
                # state-only prefill used by intermediate chunks.
                for phase in model.program["phases"]:
                    model.batches.select(1, phase["chunk"], phase=phase["phase"])
                device.synchronize()
                meta["phase_loading"] = "all compiled phases loaded before initialization and timing"
                with Session(model, batch=1, chunk=args.chunk) as session:
                    run(session, Case(1, 32, 4), prompts=[cases[0]["token_ids"][:32]],
                        warmup=0, repeats=1)
                    for item in cases:
                        prefix, ids = item["prefix_tokens"], item["token_ids"]
                        print("MEASURE AKAS", prefix, flush=True)
                        row = run(session, Case(1, prefix, args.output_tokens),
                                  prompts=[ids], warmup=args.warmup, repeats=args.repeats)
                        print("OUTPUT CHECK AKAS", prefix, flush=True)
                        # Readback is outside native phase timing. Retain a separate
                        # greedy trajectory for output-count and cross-runtime inspection.
                        session.reset()
                        output = [int(session.prefill([ids])[0])]
                        for _ in range(args.output_tokens - 1):
                            output.append(int(session.step([output[-1]])[0]))
                        logits = session.read_feature("logits")
                        import numpy as np
                        if not np.isfinite(logits).all() or len(output) != args.output_tokens:
                            raise ValueError("nonfinite logits or incomplete generation")
                        row.update(output_ids=[output], text_case={k:v for k,v in item.items()
                                   if k != "token_ids"}, correctness=dict(correct=True,
                                   finite_logits=True, output_count=True,
                                   full_model_reference=False, scope=meta["qualification"]))
                        report["results"].append(row)
                        meta["device"] = dict(device.info)
                        recorder.update(report)
                        print("RESULT AKAS", prefix,
                              row["summary"]["prefill_tokens_per_second"]["median"],
                              row["summary"]["decode_tokens_per_second"]["median"], flush=True)
    record_run(args, "akas", measure)


def request(url, data=None):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


def llama(args):
    cases = inputs(args)

    def measure(meta, report, recorder):
        listing = subprocess.check_output([str(args.binary), "--list-devices"],
                                         text=True, stderr=subprocess.STDOUT)
        if "Vulkan0:" not in listing or "llvmpipe" in listing.lower():
            raise ValueError("hardware Vulkan0 is absent: " + listing)
        from tensor.providers.webgpu.adapter import probe
        detected = probe(0)
        if detected["adapter"]["device"] not in listing:
            raise ValueError("llama.cpp and AKAS selected different adapters")
        meta["device"] = dict(name=detected["adapter"]["device"],
                              driver=detected["adapter"]["description"],
                              adapter=detected["adapter"], device_listing=listing)
        meta["runtime_version"] = subprocess.check_output([str(args.binary), "--version"],
                                      text=True, stderr=subprocess.STDOUT).strip()
        meta["runtime_revision"] = args.runtime_revision
        if args.runtime_revision[:9] not in meta["runtime_version"]:
            raise ValueError("llama.cpp binary does not match the declared revision")
        meta["binary_sha256"] = digest(args.binary)
        meta["runtime_files"] = {p.name: digest(p) for p in args.binary.parent.glob("*.dll")}
        command = [str(args.binary), "-m", str(args.checkpoint), "-ngl", "99",
                   "--device", "Vulkan0", "-c", str(args.context), "-np", "1",
                   "-b", "2048", "-ub", str(args.ubatch), "-t", "2", "-fa", "on",
                   "--cache-type-k", "f16", "--cache-type-v", "f16", "--cache-ram", "0",
                   "--ctx-checkpoints", "0", "--no-cache-idle-slots", "--host", "127.0.0.1",
                   "--port", str(args.port), "--no-webui"]
        meta["configuration"] = command
        log_path = args.out.with_suffix(".server.log")
        base = f"http://127.0.0.1:{args.port}"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                until = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("llama-server exited; inspect " + str(log_path))
                    try:
                        if request(base + "/health").get("status") == "ok":
                            break
                    except Exception:
                        if time.monotonic() > until:
                            raise TimeoutError("llama-server startup")
                        time.sleep(.25)
                initialized = request(base + "/completion", dict(
                    prompt=cases[0]["token_ids"][:32], n_predict=4, temperature=0,
                    repeat_penalty=1, presence_penalty=0, frequency_penalty=0,
                    ignore_eos=True, cache_prompt=False, return_tokens=True))
                if initialized["tokens_predicted"] != 4:
                    raise ValueError("incomplete initialization request")
                for item in cases:
                    prefix, ids = item["prefix_tokens"], item["token_ids"]
                    print("MEASURE LLAMA", prefix, flush=True)
                    samples, outputs = [], []
                    for iteration in range(args.warmup + args.repeats):
                        start = time.perf_counter()
                        response = request(base + "/completion", dict(prompt=ids,
                            n_predict=args.output_tokens, temperature=0, repeat_penalty=1,
                            presence_penalty=0, frequency_penalty=0, seed=17,
                            ignore_eos=True, cache_prompt=False, return_tokens=True))
                        elapsed = time.perf_counter() - start
                        if (response["tokens_predicted"] != args.output_tokens or
                            len(response["tokens"]) != args.output_tokens or
                            response["timings"]["prompt_n"] != prefix):
                            raise ValueError("incomplete output or reused prompt")
                        if iteration >= args.warmup:
                            t = response["timings"]
                            samples.append(dict(phase_protocol="llama-server-eval-real-text-v1",
                                prefill_tokens_per_second=prefix/(t["prompt_ms"]/1000),
                                decode_tokens_per_second=(args.output_tokens-1)/(t["predicted_ms"]/1000),
                                output_tokens_per_second=args.output_tokens/elapsed,
                                wall_seconds=elapsed, response=response))
                            outputs.append(response["tokens"])
                    stable = all(o == outputs[0] for o in outputs)
                    row = dict(case=dict(batch=1, prompt=prefix, generated=args.output_tokens,
                               depth=0, mixed=False), status="ok" if stable else "error",
                               prompt_sha256=item["prompt_sha256"], samples=samples,
                               output_ids=[outputs[0]], correctness=dict(output_count=True,
                               deterministic=stable, full_model_reference=False))
                    report["results"].append(row)
                    recorder.update(report)
                    print("RESULT LLAMA", prefix, samples[-1]["prefill_tokens_per_second"],
                          samples[-1]["decode_tokens_per_second"], flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                recorder.artifact(log_path, "baseline/server.log")
    record_run(args, "llama", measure)


def report(args):
    from tools.campaign_report import SCHEMA, measurement, metadata_for, record, render, report_directory

    records = [record(args.history, r) for r in args.runs]
    if any(not r or not r.get("completed_at") or r["status"] == "invalidated" for r in records):
        raise ValueError("select sealed runs with unchanged source")
    native = [r for r in records if r["metadata"]["runtime"] == "akas"]
    baseline = [r for r in records if r["metadata"]["runtime"] == "llama"]
    if len(native) != 1 or len(baseline) != 1:
        raise ValueError("select one AKAS and one llama run")
    a, b = native[0], baseline[0]
    for key in ("checkpoint_sha256", "prompt_manifest_sha256", "output_tokens", "kv_dtype"):
        if a["metadata"][key] != b["metadata"][key]:
            raise ValueError("mismatched " + key)
    workload = dict(batches=[1], prefix_tokens=[r["case"]["prompt"] for r in a["results"]],
                    output_tokens=a["metadata"]["output_tokens"])
    if {r["case"]["prompt"] for r in b["results"]} != set(workload["prefix_tokens"]):
        raise ValueError("mismatched shape coverage")
    prompts = {r["case"]["prompt"]: r.get("prompt_sha256") for r in a["results"]}
    if any(r.get("prompt_sha256") != prompts[r["case"]["prompt"]] for r in b["results"]):
        raise ValueError("mismatched prompt fingerprints")
    model_id = "lfm2.5-2.6b"
    aid = f"{model_id}-q4_0-webgpu-float16-ar"
    bid = f"baseline-{model_id}-llama-vulkan-q4_0-ar"
    spec = dict(schema="akas.matrix-campaign.v1", models=[model_id], formats=["q4_0"],
                backends=["webgpu"], caches=["float16"], workload=workload,
                execution="retained-c4-context-sweep", baselines=[dict(runtime="llama",
                provider="vulkan", weight_format="q4_0", mode="ar")])
    state = dict(schema="akas.matrix-campaign.v1", identity=hashlib.sha256(
        json.dumps(dict(spec=spec, runs=args.runs), sort_keys=True).encode()).hexdigest(),
        specification=spec, models={model_id:dict(id=model_id, model=a["metadata"]["model"],
        modes=["ar"])}, created_at=min(r["started_at"] for r in records), updated_at=utc(),
        status="completed" if all(r["status"] == "passed" for r in records) else "failed",
        source=a["source"], entries=[dict(id=aid, status=a["status"],
        context=dict(workload=workload))], spent_seconds=sum(
            (datetime.fromisoformat(r["completed_at"]) - datetime.fromisoformat(r["started_at"])).total_seconds()
            for r in records))
    profiles = []
    for data, pid, kind in ((a, aid, "akas"), (b, bid, "baseline")):
        meta = data["metadata"]
        qualified = data["status"] == "passed"
        profiles.append(dict(id=pid, kind=kind, runtime=meta["runtime"], model_id=model_id,
            model=meta["model"], provider=meta["provider"], weight_format="q4_0",
            cache_request="float16", mode="ar", status=data["status"], error=meta.get("error"),
            selected_run_id=data["id"], qualifying_status="passed" if qualified else "unqualified",
            kv_dtype=meta["kv_dtype"], spent_seconds=(
                datetime.fromisoformat(data["completed_at"]) - datetime.fromisoformat(data["started_at"])
            ).total_seconds(), recipe=dict(chunk=meta["prefill_chunk"],
            prefill_requests=1, graphs=True, trials=meta.get("compilation_trials_allowance")), measurements=[measurement(row, kind,
            qualified, data["id"], meta["protocol"]) for row in data["results"]]))
    args.out.mkdir(parents=True, exist_ok=True)
    campaign = args.out / "campaign.json"
    write(campaign, state)
    payload = dict(schema=SCHEMA, generated_at=utc(), campaign=state,
                   campaign_sha256=digest(campaign), workload=workload, profiles=profiles,
                   attempts=[], evidence={r["id"]:metadata_for(r) for r in records},
                   schedules={aid:dict(run_id=a["id"], recipe=profiles[0]["recipe"],
                                       phases=a["search"])})
    baseline_rows = {row["case"]["prompt"]: row for row in b["results"]}
    agreements = []
    for row in a["results"]:
        other = baseline_rows[row["case"]["prompt"]]
        if row.get("output_ids") and other.get("output_ids"):
            left, right = row["output_ids"][0], other["output_ids"][0]
            if len(left) != workload["output_tokens"] or len(right) != workload["output_tokens"]:
                raise ValueError("incomplete output trajectory")
            agreements.append(dict(prefix=row["case"]["prompt"], output=len(left),
                exact=left == right, common_prefix_tokens=next(
                    (i for i, (x, y) in enumerate(zip(left, right)) if x != y), len(left)),
                native_run_id=a["id"], baseline_run_id=b["id"],
                native_output_ids=left, baseline_output_ids=right))
    payload["output_comparisons"] = agreements
    parent_id = a.get("change", {}).get("parent_run")
    pilot = record(args.history, parent_id) if parent_id else None
    reference_id = pilot.get("change", {}).get("parent_run") if pilot else None
    reference = record(args.history, reference_id) if reference_id else None
    if pilot and reference:
        for item in (pilot, reference):
            if item["status"] != "passed" or not item.get("completed_at"):
                raise ValueError("calibration requires sealed, passed runs")
            for key in ("checkpoint_sha256", "prompt_manifest_sha256", "output_tokens",
                        "kv_dtype", "protocol", "provider"):
                if item["metadata"][key] != a["metadata"][key]:
                    raise ValueError("calibration differs in " + key)
        for key in ("warmup", "repeats"):
            if pilot["metadata"][key] != reference["metadata"][key]:
                raise ValueError("pilot repetition protocols differ")
        before = {row["case"]["prompt"]: row for row in reference["results"]}
        calibration = []
        for row in pilot["results"]:
            old = before[row["case"]["prompt"]]
            if row["case"] != old["case"] or row["prompt_sha256"] != old["prompt_sha256"]:
                raise ValueError("pilot shapes or inputs differ")
            old_rates = measurement(old, "akas", True, reference_id)
            new_rates = measurement(row, "akas", True, parent_id)
            calibration.append(dict(prefix=row["case"]["prompt"],
                before_prefill=old_rates["prefill_tok_s"], after_prefill=new_rates["prefill_tok_s"],
                before_decode=old_rates["decode_tok_s"], after_decode=new_rates["decode_tok_s"],
                prefill_speedup=new_rates["prefill_tok_s"] / old_rates["prefill_tok_s"],
                exact_output=row["output_ids"] == old["output_ids"]))
        payload["calibration"] = dict(reference_run_id=reference_id, selected_run_id=parent_id,
            warmup=pilot["metadata"]["warmup"], repeats=pilot["metadata"]["repeats"],
            before_chunk=reference["metadata"]["prefill_chunk"],
            after_chunk=pilot["metadata"]["prefill_chunk"], cases=calibration)
        payload["evidence"].update({item["id"]: metadata_for(item) for item in (pilot, reference)})
    destination = report_directory(state, args.history)
    render(payload, destination, args.history)
    print(destination / "README.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("compile", "akas", "llama", "report"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--revision", default="e7caca5d835a3901a8e0d63e94009429bafafdfc")
    parser.add_argument("--program", type=Path, default=Path("build/lfm2.5-2.6b-vulkan/program"))
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--runtime-revision", default="3642238cc12050c8ab70afdf4b3121ea365c7579")
    parser.add_argument("--prefixes", type=int, nargs="+", default=PREFIXES)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--kv-dtype", choices=("float16",), default="float16")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--prefill-chunks", type=int, nargs="*", default=[])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--port", type=int, default=8817)
    parser.add_argument("--history", type=Path, default=Path("benchmarks"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-run")
    parser.add_argument("--change", default="Matched Vulkan context sweep with retained C4 token IDs")
    parser.add_argument("--runs", nargs="+")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.action != "report" and not args.checkpoint:
        parser.error("--checkpoint is required")
    if args.action in ("akas", "llama") and not args.prompts:
        parser.error("--prompts is required")
    if args.action == "llama" and not args.binary:
        parser.error("--binary is required")
    if (min(args.prefixes) < 1 or max(args.prefixes) + args.output_tokens > args.context
            or args.repeats < 1 or args.warmup < 0):
        parser.error("invalid shape, context or repetition count")
    globals()["compile_program" if args.action == "compile" else args.action](args)


if __name__ == "__main__":
    main()
