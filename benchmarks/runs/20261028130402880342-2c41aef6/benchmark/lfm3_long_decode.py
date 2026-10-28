"""Real-prefill, exact state-restored LFM3 decode on AMD Vulkan.

Prefill runs once per prefix. Both runtimes restore their complete recurrent
and KV state before each warmup/measurement. Decode wall times include greedy
sampling; llama.cpp's narrower context-eval timers are also retained.
"""

import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np

from akas.benchmarking.history import Recorder, digest, write
from akas.benchmarking.runner import summarize
from akas.benchmarking.workload import prompt_digest
from tools.lfm3_vulkan_benchmark import PROFILE
from tools.model_benchmark import prompts, PREFIXES


def progress(runtime, prefix):
    started, last = time.perf_counter(), [0]

    def report(done, _elapsed=None):
        if done == prefix or done - last[0] >= 4096:
            last[0] = done
            row = dict(
                runtime=runtime,
                prefix=prefix,
                completed_tokens=done,
                elapsed_seconds=time.perf_counter() - started,
            )
            print("PREFILL", json.dumps(row), flush=True)

    return report


def native(args, meta, recorder, result):
    from akas.models.lfm3 import LFM3
    from akas.benchmarking.state import RecurrentSnapshot
    from tools.context_sweep import vulkan_device

    groups = {}
    for prefix in args.prefixes:
        program = (
            args.short_program
            if args.short_program and prefix <= args.short_program_max_prefix
            else args.program
        )
        groups.setdefault(program, []).append(prefix)
    meta["programs"] = []
    with vulkan_device() as device:
        meta["device"] = dict(device.info)
        from tensor.providers import _webgpu_native
        from wgpu.backends.wgpu_native import lib_path, lib_version_info

        meta["native_submission_helper_sha256"] = digest(_webgpu_native.__file__)
        meta["wgpu_native"] = dict(
            version=list(lib_version_info), sha256=digest(Path(lib_path))
        )
        for program, prefixes in groups.items():
            manifest = json.loads((program / "program.json").read_text())
            if manifest["max_context"] != args.context:
                raise ValueError("all programs must have the declared context limit")
            recorder.program(program)
            description = dict(
                path=str(program.resolve()),
                sha256=digest(program / "program.json"),
                prefixes=prefixes,
                options=manifest["model_options"],
            )
            meta["programs"].append(description)
            with LFM3(
                args.checkpoint,
                program,
                device,
                graphs=True,
                checkpoint_options={"tokenizer": False},
            ) as model:
                slot = model.admit()
                first = model.prefill([slot], prompts(32, PROFILE.vocab), chunk=32)
                model.decode_steps([slot], first, 4, read=False)
                device.synchronize()
                model.release(slot)
                for prefix in prefixes:
                    print("CASE native", prefix, flush=True)
                    ids = prompts(prefix, PROFILE.vocab)
                    slot = model.admit()
                    saved = None
                    try:
                        reporter, done = progress("native", prefix), [0]

                        def on_chunk(slots, sequences, frame):
                            done[0] += len(sequences[0])
                            if done[0] == prefix or done[0] % 4096 == 0:
                                device.synchronize()
                                reporter(done[0])

                        device.synchronize()
                        start = time.perf_counter()
                        first = model.prefill(
                            [slot], ids, chunk=args.chunk, on_chunk=on_chunk
                        )
                        device.synchronize()
                        pp = time.perf_counter() - start
                        if model.positions[slot] != prefix:
                            raise ValueError("native prefill position differs")
                        saved = RecurrentSnapshot.capture(model)
                        if args.state_cache_dir and prefix == max(args.prefixes):
                            destination = args.state_cache_dir / str(prefix)
                            destination.mkdir(parents=True, exist_ok=True)
                            files = {}
                            for index, (name, values) in enumerate(
                                sorted(saved.buffers.items())
                            ):
                                path = destination / f"{index}.npy"
                                np.save(path, values, allow_pickle=False)
                                files[name] = dict(
                                    file=path.name,
                                    sha256=digest(path),
                                    shape=list(values.shape),
                                    dtype=str(values.dtype),
                                )
                            write(
                                destination / "state.json",
                                dict(
                                    checkpoint_sha256=PROFILE.checkpoint_sha256,
                                    prompt_sha256=prompt_digest(ids),
                                    context=args.context,
                                    prefix=prefix,
                                    first=first,
                                    positions=saved.positions.tolist(),
                                    producer_program=description,
                                    files=files,
                                ),
                            )
                        samples, reference_logits, final_token = [], None, None
                        for iteration in range(args.warmup + args.repeats):
                            saved.restore(model)
                            start = time.perf_counter()
                            model.decode_steps([slot], first, 127, read=False)
                            device.synchronize()
                            elapsed = time.perf_counter() - start
                            frame = model.batches.select(1, 1)
                            logits = frame.feature("logits").to_numpy()
                            if not np.isfinite(logits).all():
                                raise ValueError("nonfinite native logits")
                            if reference_logits is None:
                                reference_logits = logits.copy()
                            else:
                                np.testing.assert_array_equal(logits, reference_logits)
                            token = int(frame.buffers[frame.prediction].to_numpy()[0])
                            if final_token is not None and token != final_token:
                                raise ValueError("native replay token differs")
                            final_token = token
                            if iteration >= args.warmup:
                                samples.append(
                                    dict(
                                        decode_seconds=elapsed,
                                        decode_tokens_per_second=127 / elapsed,
                                    )
                                )
                        saved.restore(model)
                        output = [int(first[0])]
                        for _ in range(127):
                            output.append(
                                int(model.decode_steps([slot], [output[-1]], 1)[0])
                            )
                        np.testing.assert_array_equal(
                            frame.feature("logits").to_numpy(), reference_logits
                        )
                        if output[-1] != final_token:
                            raise ValueError(
                                "native captured trajectory differs from timed decode"
                            )
                        row = dict(
                            case=dict(batch=1, prompt=prefix, generated=128),
                            status="ok",
                            prompt_sha256=prompt_digest(ids),
                            output_ids=output,
                            prefill_observation=dict(
                                seconds=pp, tokens_per_second=prefix / pp, samples=1
                            ),
                            samples=samples,
                            summary=dict(
                                decode_tokens_per_second=summarize(
                                    [s["decode_tokens_per_second"] for s in samples]
                                )
                            ),
                            program_sha256=description["sha256"],
                            replay=dict(
                                final_logits_bitwise_equal=True,
                                finite=True,
                                snapshot_bytes=sum(
                                    a.nbytes for a in saved.buffers.values()
                                ),
                            ),
                        )
                        result["results"].append(row)
                        recorder.update(result)
                        print(
                            "RESULT native",
                            prefix,
                            row["summary"]["decode_tokens_per_second"]["median"],
                            flush=True,
                        )
                    finally:
                        if saved is not None:
                            del saved
                        model.release(slot)


def llama(args, meta, recorder, result):
    from tools.baselines.llama_state import LlamaState, LLAMA_DLL_SHA256

    directory = args.binary_directory.resolve()
    meta["runtime_revision"] = "8ca4060b7e30e76baaa5670ee1ec1b760b7f8552"
    meta["runtime_files"] = {p.name: digest(p) for p in directory.glob("*.dll")}
    meta["llama_dll_sha256"] = LLAMA_DLL_SHA256
    api = LlamaState(args.checkpoint, directory, context=args.context)
    try:
        meta["device"] = dict(name=api.device, provider="vulkan")
        meta["configuration"] = api.configuration
        if api.device != "AMD Radeon RX 6700 XT" or api.vocab != PROFILE.vocab:
            raise ValueError("C API adapter or vocabulary differs")
        api.clear()
        first, _ = api.prefill(prompts(32, PROFILE.vocab)[0])
        api.generate(4, first=first)
        for prefix in args.prefixes:
            print("CASE llama", prefix, flush=True)
            ids = prompts(prefix, PROFILE.vocab)
            api.clear()
            reporter = progress("llama", prefix)

            def on_chunk(done, elapsed):
                if done == prefix or done % 4096 == 0:
                    api.call("llama_synchronize", api.context)
                    reporter(done)

            first, pp = api.prefill(ids[0], progress=on_chunk)
            saved = api.capture()
            samples, reference, reference_logits = [], None, None
            for iteration in range(args.warmup + args.repeats):
                seed = api.restore(saved)
                if seed != first:
                    raise ValueError("restored llama seed differs")
                start = time.perf_counter()
                output, perf = api.generate(first=seed)
                elapsed = time.perf_counter() - start
                if reference is None:
                    reference = output
                elif output != reference:
                    raise ValueError("llama replay trajectory differs")
                logits = api.last_logits()
                if reference_logits is None:
                    reference_logits = logits.copy()
                else:
                    np.testing.assert_array_equal(logits, reference_logits)
                if (
                    perf["n_eval"] != 127
                    or perf["t_eval_ms"] <= 0
                    or not api.finite_logits()
                ):
                    raise ValueError("invalid llama eval counters or logits")
                if iteration >= args.warmup:
                    samples.append(
                        dict(
                            decode_seconds=elapsed,
                            decode_tokens_per_second=127 / elapsed,
                            context_eval_tokens_per_second=127000 / perf["t_eval_ms"],
                            context_perf=perf,
                        )
                    )
            row = dict(
                case=dict(batch=1, prompt=prefix, generated=128),
                status="ok",
                prompt_sha256=prompt_digest(ids),
                output_ids=reference,
                prefill_observation=dict(
                    seconds=pp, tokens_per_second=prefix / pp, samples=1
                ),
                samples=samples,
                summary=dict(
                    decode_tokens_per_second=summarize(
                        [s["decode_tokens_per_second"] for s in samples]
                    )
                ),
                replay=dict(
                    all_output_tokens_equal=True,
                    final_logits_bitwise_equal=True,
                    finite=True,
                    snapshot_bytes=saved[1],
                ),
            )
            del saved
            result["results"].append(row)
            recorder.update(result)
            print(
                "RESULT llama",
                prefix,
                row["summary"]["decode_tokens_per_second"]["median"],
                flush=True,
            )
    finally:
        api.close()


def measure(args):
    if digest(args.checkpoint) != PROFILE.checkpoint_sha256:
        raise ValueError("model differs from the official QAD Q4_0")
    if args.qualification_run:
        qualification = json.loads(
            (args.history / "runs" / args.qualification_run / "run.json").read_text()
        )
        if (
            qualification["status"] != "passed"
            or qualification["metadata"].get("purpose") != "model-qualification"
        ):
            raise ValueError("qualification must be a sealed passed model reference")
    meta = dict(
        runtime="akas" if args.action == "native" else "llama",
        purpose="decode-benchmark",
        model=PROFILE.model,
        mode="ar",
        weight_format=PROFILE.weight_format,
        kv_dtype="float16",
        target="webgpu-portable-v1" if args.action == "native" else "vulkan",
        independent_small_model_reference=bool(args.qualification_run),
        provider="webgpu" if args.action == "native" else "vulkan",
        protocol="real-prefill-state-restored-wall-decode-v1",
        checkpoint_sha256=PROFILE.checkpoint_sha256,
        max_context=args.context,
        batch=1,
        generated=128,
        measured_decode_steps=127,
        warmup=args.warmup,
        repeats=args.repeats,
        qualification_run_id=args.qualification_run,
        sampling=dict(
            greedy=True, stop_on_eos=False, eos_logit_bias=False, penalties=False
        ),
        timing="Completed wall decode including greedy sampling and control dispatch; prefill, state restore and validation excluded. llama context-eval timers retained separately.",
        prefill="One real deterministic prompt per prefix, empty request; no Gaussian KV. Full recurrent/KV state and first greedy seed restored before each replay.",
    )
    recorder = Recorder(
        args.history,
        meta,
        kind="baseline" if args.action == "llama" else "configuration",
        change="LFM3 real-prefill state-restored decode context sweep",
    )
    result = dict(schema="akas.decode-benchmark.v1", metadata=meta, results=[])
    print("RUN", recorder.id, flush=True)
    try:
        (native if args.action == "native" else llama)(args, meta, recorder, result)
    except BaseException:
        meta["error"] = traceback.format_exc()
        print(meta["error"], flush=True)
    finally:
        recorder.artifact(__file__, "benchmark/lfm3_long_decode.py")
        recorder.artifact("tools/baselines/llama_state.py", "benchmark/llama_state.py")
        recorder.artifact("src/akas/benchmarking/state.py", "benchmark/state.py")
        passed = not meta.get("error") and len(result["results"]) == len(args.prefixes)
        record = recorder.update(
            result, final=True, status="passed" if passed else "failed"
        )
        write(args.out, dict(run_id=recorder.id, status=record["status"]))
        print("RETAINED", recorder.id, record["status"], flush=True)
    if record["status"] != "passed":
        raise SystemExit(1)


def report(args):
    records = [
        json.loads((args.history / "runs" / run / "run.json").read_text())
        for run in args.runs
    ]
    if any(r["status"] != "passed" for r in records):
        raise ValueError("report requires passed sealed runs")
    protocols = {r["metadata"]["protocol"] for r in records}
    contexts = {r["metadata"]["max_context"] for r in records}
    if (
        protocols != {"real-prefill-state-restored-wall-decode-v1"}
        or len(contexts) != 1
    ):
        raise ValueError("context or timing protocol mismatch")
    rows = {}
    for record in records:
        runtime = record["metadata"]["runtime"]
        for row in record["results"]:
            entry = rows.setdefault(row["case"]["prompt"], {})
            if runtime in entry:
                raise ValueError("duplicate runtime/prefix")
            entry[runtime] = row
    output = []
    for prefix, values in sorted(rows.items()):
        if set(values) != {"akas", "llama"}:
            raise ValueError("unmatched prefix")
        a, b = values["akas"], values["llama"]
        if a["prompt_sha256"] != b["prompt_sha256"]:
            raise ValueError("prompt mismatch")
        n, l = (v["summary"]["decode_tokens_per_second"]["median"] for v in (a, b))
        output.append(
            dict(
                prefix=prefix,
                akas=n,
                llama=l,
                ratio=n / l,
                passed=n >= l,
                exact_output=a["output_ids"] == b["output_ids"],
                common_prefix_tokens=next(
                    (
                        i
                        for i, (x, y) in enumerate(
                            zip(a["output_ids"], b["output_ids"])
                        )
                        if x != y
                    ),
                    128,
                ),
            )
        )
    args.out.mkdir(parents=True, exist_ok=True)
    passed = all(r["passed"] for r in output)
    write(
        args.out / "decode.json",
        dict(
            passed=passed,
            protocol=next(iter(protocols)),
            context=next(iter(contexts)),
            runs=args.runs,
            rows=output,
        ),
    )
    text = "# LFM3 real-prefill decode comparison\n\n"
    text += f"Context limit {next(iter(contexts)):,}; one request; 128 outputs (127 measured decode steps). Both engines restore complete real-prefill recurrent/KV state and the initial greedy seed before each sample. Prefill runs once per prefix.\n\n"
    text += "| Prefix | AKAS tok/s | llama.cpp tok/s | Change | Exact tokens |\n|---:|---:|---:|---:|:---:|\n"
    for r in output:
        text += f"| {r['prefix']:,} | {r['akas']:.2f} | {r['llama']:.2f} | {(r['ratio'] - 1) * 100:+.2f}% | {r['exact_output']} |\n"
    text += (
        "\nDecode median parity: **" + ("passed" if passed else "failed") + "**.\n\n"
    )
    text += "Times cover completed wall decode including greedy sampling and host dispatch, excluding state restore and output validation. llama.cpp C API context-eval timers are also retained in raw samples; these narrower timers do not gate the wall-time comparison. Prefill observations are single requests, not repeated prefill benchmarks. No claim is made for untested prefixes or end-to-end request latency.\n\n"
    for record in records:
        text += f"- {record['metadata']['runtime']}: `{record['id']}`; {record['metadata']['warmup']} warmups and {record['metadata']['repeats']} samples.\n"
    (args.out / "README.md").write_text(text, encoding="utf-8")
    print(args.out / "README.md", "PARITY", passed)


def compile_long(args):
    from akas.models.lfm3 import LFM3
    from akas.compilation import compile_checkpoint
    from akas.compilation.tuning import SearchSettings
    from tensor.compiler.search import ScheduleProfile
    from tools.context_sweep import vulkan_device

    recipe = Path("benchmarks/recipes/lfm3-rx6700xt-vulkan")
    profile = json.loads((recipe / "schedules.json").read_text())
    for entry in profile["entries"]:
        p = entry["parameters"]
        if p.get("capacity") == 8192:
            p["capacity"] = args.context
        if p.get("cache_capacity") == 8192:
            p["cache_capacity"] = args.context
        if p.get("blocks") == 8:
            p["blocks"] = (args.context + 1023) // 1024
        if (
            entry["operation"] in ("window_grouped", "window_merge")
            and p.get("r") == 1
            and p.get("window") == 0
        ):
            p["splits"] = args.splits
    with vulkan_device() as device:
        compile_checkpoint(
            LFM3,
            args.checkpoint,
            args.program,
            checkpoint_options={"tokenizer": False},
            geometries=[(1, 1), (1, 32), (1, args.chunk)],
            max_requests=1,
            max_context=args.context,
            model_options={
                "prefill_chunk": args.chunk,
                **PROFILE.optimized_options,
                "decode_splits": args.splits,
            },
            provider="webgpu",
            device=device,
            policy=json.loads((recipe / "policy.json").read_text()),
            schedules=ScheduleProfile(profile),
            search_settings=SearchSettings(budget=1, samples=1, repeats=1),
        )
    write(
        args.out,
        dict(
            program=str(args.program.resolve()),
            sha256=digest(args.program / "program.json"),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("native", "llama", "report", "compile"))
    parser.add_argument("--checkpoint", type=Path, default=PROFILE.checkpoint)
    parser.add_argument("--binary-directory", type=Path, default=PROFILE.binary.parent)
    parser.add_argument(
        "--program", type=Path, default=Path("build/lfm3-vulkan/long-128")
    )
    parser.add_argument("--short-program", type=Path)
    parser.add_argument("--short-program-max-prefix", type=int, default=8192)
    parser.add_argument(
        "--state-cache-dir",
        type=Path,
        help="retain the largest real-prefill state as checksummed arrays for decode tuning",
    )
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--prefixes", type=int, nargs="+", default=PREFIXES)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--splits", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--qualification-run")
    parser.add_argument("--history", type=Path, default=Path("benchmarks"))
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.repeats < 1
        or len(set(args.prefixes)) != len(args.prefixes)
        or min(args.prefixes) < 1
        or max(args.prefixes) + 128 > args.context
    ):
        parser.error("invalid repetitions or prefix/context geometry")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "report":
        report(args)
    elif args.action == "compile":
        compile_long(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
