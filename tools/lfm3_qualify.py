"""Seal independent LFM3 CUDA reference qualification as benchmark evidence."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from akas.benchmarking.history import Recorder, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("build/lfm3/qualification.json")
    )
    parser.add_argument("--history", type=Path, default=Path("benchmarks"))
    parser.add_argument("--optimized-only", action="store_true")
    parser.add_argument("--base-qualification")
    args = parser.parse_args()
    if args.optimized_only:
        from tools.campaign_report import record

        base = (
            record(args.history, args.base_qualification)
            if args.base_qualification
            else None
        )
        if (
            not base
            or base["status"] != "passed"
            or base["metadata"].get("purpose") != "model-qualification"
            or not {"bf16", "fp8", "gguf"}.issubset(
                base["metadata"].get("weight_formats", [])
            )
        ):
            parser.error(
                "--optimized-only requires a passed BF16/FP8/GGUF --base-qualification"
            )
    meta = dict(
        runtime="akas",
        purpose="model-qualification",
        model="LiquidAI/LFM3-1.7B",
        provider="cuda",
        target="sm_89",
        mode="ar",
        weight_formats=["gguf"] if args.optimized_only else ["bf16", "fp8", "gguf"],
        base_qualification_run_id=args.base_qualification,
        reference="Independent NumPy full-sequence equations; reordered/ragged requests, sliding wrap, convolution slot reuse and Q8 packing",
    )
    recorder = Recorder(
        args.history,
        meta,
        kind="qualification",
        change="LFM3 independent CUDA model qualification",
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_lfm3.py",
        "tests/test_activation_q8.py",
        "-q",
    ]
    if args.optimized_only:
        command += ["-k", "gguf-q8-fused or cuda_q8_exact"]
    log = recorder.directory / "pytest.log"
    env = {
        **os.environ,
        "AKAS_LFM3_CUDA": "1",
        "AKAS_FORWARD_CUDA": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "AKAS_LFM3_PROGRAM_CACHE": str(
            Path("build/lfm3/qualification-programs").resolve()
        ),
    }
    with log.open("w") as stream:
        result = subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    meta.update(command=command, exit_code=result.returncode)
    recorder.artifact(log, "qualification/pytest.log")
    for path in (
        "tests/test_lfm3.py",
        "tests/support/lfm3.py",
        "tests/test_activation_q8.py",
    ):
        recorder.artifact(path, "qualification/" + Path(path).name)
    record = recorder.update(
        dict(metadata=meta, results=[]),
        final=True,
        status="passed" if result.returncode == 0 else "failed",
    )
    write(args.out, dict(run_id=recorder.id, status=record["status"]))
    print(recorder.id, record["status"])
    if record["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
