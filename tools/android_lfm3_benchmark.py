"""Quick on-device LFM3 WebGPU sweep with complete greedy output capture.

Run with an Android Python consumer and an offline-compiled AKAS program.
Phase wall times include Python control, sampling and per-token readback;
model upload and pipeline creation are excluded. One short warmup is shared
by the sweep. This is a performance smoke test, not a model quality eval.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np

from akas.benchmarking.session import Session
from akas.models.lfm3 import LFM3


def write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
    temporary.replace(path)


def prompt(prefix, vocab):
    rng = np.random.default_rng(17 + 4099 + prefix)
    return rng.integers(0, vocab, prefix, dtype="int32").astype(int).tolist()


def measure(session, tokens, generated, *, progress=None, logits_path=None):
    session.reset()
    session.device.synchronize()
    start = time.perf_counter()
    output = session.prefill([tokens])
    session.device.synchronize()
    pp = time.perf_counter() - start
    output = [int(output[0])]
    if progress:
        progress("prefill", pp)
    if logits_path is not None:
        np.save(logits_path, session.read_feature("logits").reshape(-1))
    start = time.perf_counter()
    for _ in range(generated - 1):
        output.append(int(session.step([output[-1]])[0]))
    session.device.synchronize()
    tg = time.perf_counter() - start
    logits = session.read_feature("logits")
    if not np.isfinite(logits).all():
        raise ValueError("model produced nonfinite logits")
    return dict(
        prefill_seconds=pp,
        decode_seconds=tg,
        prefill_tokens_per_second=len(tokens) / pp,
        decode_tokens_per_second=(generated - 1) / tg,
        total_seconds=pp + tg,
        output_ids=output,
        finite_logits=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[32, 512, 2048])
    parser.add_argument("--generated", type=int, default=32)
    parser.add_argument("--warmup-prefix", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    program = json.loads((args.program / "program.json").read_text())
    if args.generated < 2 or not args.prefixes or min(args.prefixes) < 1 or args.warmup_prefix < 1:
        parser.error("positive prefixes and at least two output tokens required")
    if max(*args.prefixes, args.warmup_prefix) + args.generated > program["max_context"]:
        parser.error("request exceeds compiled context")
    import tensor
    import cffi
    import wgpu
    from wgpu.backends.wgpu_native import lib_version_info

    report = dict(
        schema="akas.android-lfm3.v1",
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        metadata=dict(
            runtime="akas",
            platform=platform.platform(),
            python=platform.python_version(),
            numpy=np.__version__,
            cffi=cffi.__version__,
            wgpu=wgpu.__version__,
            wgpu_native=list(lib_version_info),
            checkpoint_sha256=hashlib.file_digest(args.checkpoint.open("rb"), "sha256").hexdigest(),
            program_sha256=hashlib.sha256((args.program / "program.json").read_bytes()).hexdigest(),
            prefixes=args.prefixes,
            generated_tokens=args.generated,
            decode_steps=args.generated - 1,
            prefill_chunk=program["model_options"]["prefill_chunk"],
            max_context=program["max_context"],
            kv_dtype="float16",
            warmup=f"one {args.warmup_prefix}-token prefix and 3 generated tokens before the sweep",
            repeats=1,
            sampling="greedy, ignore EOS to execute the requested output length",
            input="deterministic synthetic token IDs; seed = 17 + 4099 + prefix",
            timing="synchronized on-device wall time; includes Python controls, greedy sampling and per-token readback; excludes loading and pipeline creation",
            independently_qualified=False,
        ),
        cases=[],
    )
    write(args.out, report)
    try:
        start = time.perf_counter()
        device = tensor.Device(provider="webgpu", max_buffer_size=268435456)
        # Match context_sweep.vulkan_device: some AMD drivers report UINT32_MAX,
        # which wgpu treats as UNDEFINED and otherwise replaces with eight.
        reported = device._adapter.limits["max-storage-buffers-per-shader-stage"]
        if reported == 0xFFFFFFFF:
            device._adapter._limits["max-storage-buffers-per-shader-stage"] = 32
        with device:
            if device.info["adapter"].get("backend_type") != "Vulkan":
                raise RuntimeError("benchmark requires a Vulkan adapter")
            print("DEVICE", json.dumps(device.info), flush=True)
            with LFM3(args.checkpoint, args.program, device, graphs=True) as model:
                for row in program["phases"]:
                    print("LOAD PHASE", row["phase"], row["chunk"], flush=True)
                    model.batches.select(row["rows"], row["chunk"], phase=row["phase"])
                device.synchronize()
                report["metadata"].update(
                    device=dict(device.info),
                    load_seconds=time.perf_counter() - start,
                )
                with Session(model, batch=1, chunk=program["model_options"]["prefill_chunk"]) as session:
                    report["metadata"]["warmup_result"] = measure(session, prompt(args.warmup_prefix, model.config.vocab), 3)
                    write(args.out, report)
                    print("WARMUP COMPLETE", flush=True)
                    for prefix in args.prefixes:
                        print("MEASURE", prefix, flush=True)
                        row = dict(prefix_tokens=prefix, generated_tokens=args.generated, status="running")
                        report["cases"].append(row)
                        write(args.out, report)
                        values = measure(
                            session, prompt(prefix, model.config.vocab), args.generated,
                            progress=lambda phase, seconds: print("PREFILL COMPLETE", prefix, seconds, flush=True),
                            logits_path=args.out.with_name(f"{args.out.stem}-prefix{prefix}-logits.npy"),
                        )
                        row.update(values, status="passed", allocated_bytes=session.allocated_bytes)
                        row["output_text"] = model.checkpoint.tokenizer.decode(row["output_ids"])
                        write(args.out, report)
                        print("RESULT", prefix, row["prefill_tokens_per_second"], row["decode_tokens_per_second"], flush=True)
                    if args.demo:
                        text = "What is the capital of France? Answer briefly."
                        tokens = model.checkpoint.tokenizer.chat(text)
                        report["demo"] = dict(prompt=text, **measure(session, tokens, args.generated))
                        report["demo"]["text"] = model.checkpoint.tokenizer.decode(report["demo"]["output_ids"])
                        print("DEMO", report["demo"]["text"], flush=True)
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(args.out, report)


if __name__ == "__main__":
    main()
