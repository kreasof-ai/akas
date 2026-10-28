"""Seal independent LFM3 GPU reference qualification as benchmark evidence."""

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
    parser.add_argument("--provider", choices=("cuda", "webgpu"), default="cuda")
    parser.add_argument("--cooperative-decode", action="store_true")
    parser.add_argument(
        "--cooperative-score-tile", type=int, choices=(32, 64, 128, 256), default=64
    )
    parser.add_argument("--cooperative-score-unroll", action="store_true")
    parser.add_argument(
        "--cooperative-value-tile", type=int, choices=(16, 32, 64), default=64
    )
    args = parser.parse_args()
    if args.cooperative_decode and args.provider != "webgpu":
        parser.error("cooperative decode requires WebGPU")
    if args.provider == "webgpu" and args.optimized_only:
        parser.error("WebGPU qualification includes both floating and mixed prefill")
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
        provider=args.provider,
        cooperative_decode_attention=args.cooperative_decode,
        cooperative_score_tile=args.cooperative_score_tile,
        cooperative_score_unroll=args.cooperative_score_unroll,
        cooperative_value_tile=args.cooperative_value_tile,
        target="webgpu-portable-v1" if args.provider == "webgpu" else "sm_89",
        mode="ar",
        weight_formats=["gguf"]
        if args.optimized_only or args.provider == "webgpu"
        else ["bf16", "fp8", "gguf"],
        base_qualification_run_id=args.base_qualification,
        reference="Independent NumPy full-sequence equations; reordered/ragged requests, sliding wrap, convolution slot reuse; floating and mixed prefill on WebGPU, Q8 packing on CUDA",
    )
    recorder = Recorder(
        args.history,
        meta,
        kind="qualification",
        change=f"LFM3 independent {args.provider} model qualification",
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
    if args.provider == "webgpu":
        command = [sys.executable, "-m", "pytest", "tests/test_lfm3_webgpu.py", "-q"]
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
    if args.provider == "webgpu":
        env.update(
            AKAS_LFM3_CUDA="",
            AKAS_FORWARD_CUDA="",
            AKAS_LFM3_WEBGPU="1",
            AKAS_LFM3_WEBGPU_COOPERATIVE="1" if args.cooperative_decode else "",
            AKAS_LFM3_WEBGPU_SCORE_TILE=str(args.cooperative_score_tile),
            AKAS_LFM3_WEBGPU_SCORE_UNROLL="1" if args.cooperative_score_unroll else "",
            AKAS_LFM3_WEBGPU_VALUE_TILE=str(args.cooperative_value_tile),
            PYTHONUTF8="1",
            AKAS_LFM3_WEBGPU_CACHE=str(
                Path("build/lfm3-vulkan/qualification-programs").resolve()
            ),
        )
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
        "tests/test_lfm3_webgpu.py",
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
