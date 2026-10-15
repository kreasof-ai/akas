"""Worker protocol shared by isolated third-party runtime environments."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback


_output = None


def save(report):
    """Retain completed shapes even if a later shape or process fails."""
    if _output is not None:
        temporary = _output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temporary.replace(_output)


def measurements(config, invoke, report):
    """Retain timings and numerical disagreements for every healthy shape."""
    import statistics
    import time

    save(report)
    for batch, prefix in config["shapes"]:
        samples, tokens, output = [], None, None
        deterministic, recoverable = True, False
        try:
            for _ in range(config.get("warmup", 1)):
                invoke(batch, prefix)
            for _ in range(config.get("repeats", 1)):
                start = time.perf_counter()
                output, details = invoke(batch, prefix)
                elapsed = time.perf_counter() - start
                if len(output) != batch or any(
                    len(row) != config["output_tokens"] for row in output
                ):
                    raise ValueError(
                        f"expected {batch} requests with {config['output_tokens']} outputs; got {len(output)} requests with output lengths {[len(row) for row in output]}"
                    )
                sample = dict(
                    wall_seconds=elapsed,
                    output_tokens_per_second=batch * config["output_tokens"] / elapsed,
                    **details,
                )
                if tokens is not None and output != tokens:
                    deterministic = False
                    sample["output_ids"] = output
                if tokens is None:
                    tokens = output
                samples.append(sample)
            row = dict(
                batch=batch,
                prefix=prefix,
                output=config["output_tokens"],
                wall_seconds=statistics.median(s["wall_seconds"] for s in samples),
                output_ids=tokens,
                samples=samples,
                output_count=True,
                deterministic=deterministic,
                status="passed" if deterministic else "failed",
            )
            if not deterministic:
                row["error"] = "greedy outputs changed between repetitions"
        except Exception as error:
            row = dict(
                batch=batch,
                prefix=prefix,
                output=config["output_tokens"],
                status="failed",
                error=f"{type(error).__name__}: {error}",
                samples=samples,
                output_ids=output,
                output_count=False,
            )
            recoverable = isinstance(error, ValueError)
            report["cases"].append(row)
            save(report)
            print("CASE", batch, prefix, "failed", row["error"], flush=True)
            if not recoverable:
                raise
            continue
        report["cases"].append(row)
        save(report)
        print("CASE", batch, prefix, row["status"], row["wall_seconds"], flush=True)


def phase_metrics(batch, prefix, output, first, decode):
    """Burst rates from request timings, including scheduling; not kernel timings."""
    import math

    if (
        not first
        or not decode
        or any(not math.isfinite(v) or v <= 0 for v in (*first, *decode))
    ):
        raise ValueError("runtime did not expose valid phase timings")
    return dict(
        phase_protocol="request-phase-v1",
        prefill_tokens_per_second=batch * prefix / max(first),
        decode_tokens_per_second=batch * (output - 1) / max(decode),
        ttft_seconds=max(first),
        tpot_seconds=max(decode) / (output - 1),
    )


def main(run):
    global _output
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    _output = args.out
    config = json.loads(args.config.read_text())
    report = dict(started_at=datetime.now(timezone.utc).isoformat())
    try:
        report.update(run(config))
        failed = [r for r in report.get("cases", []) if r["status"] != "passed"]
        report["status"] = "failed" if failed else "passed"
        if failed:
            report["error"] = "; ".join(
                f"B{r['batch']}/P{r['prefix']}: {r.get('error', 'qualification failed')}"
                for r in failed
            )
    except Exception as error:
        if args.out.exists():
            report.update(json.loads(args.out.read_text()))
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        print(report["traceback"], flush=True)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if report["status"] != "passed":
        raise SystemExit(1)


def prompt_tokens(tokenizer, count):
    seed = tokenizer.encode(
        "The capital of France is Paris. Explain the history of this city briefly. ",
        add_special_tokens=False,
    )
    return (seed * ((count + len(seed) - 1) // len(seed)))[:count]


def synthetic_prompts(vocab, batch, prefix):
    import numpy as np

    rng = np.random.default_rng(17 + batch * 4099 + prefix)
    return rng.integers(0, vocab, (batch, prefix), dtype="int32").astype(int).tolist()
