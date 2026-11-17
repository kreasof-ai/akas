"""Fixed-input, closed-loop streaming TTS benchmark for both ZONOS3 HTTP APIs."""


from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import threading
import time
import wave
from pathlib import Path

import httpx
import numpy as np

TEXT = (
    "Your order has been confirmed and will arrive tomorrow between nine and noon. "
    "You can change the delivery address, contact the courier, or track your package online at any time."
)
MODEL_REVISION = "d2d142a86b1e6aeb4252679673c86ecd86177bf8"


def gpu_snapshot() -> dict[str, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used, total = (int(value.strip()) for value in output.splitlines()[0].split(","))
    return {"memory_used_mib": used, "memory_total_mib": total}


class GPUMonitor:
    def __init__(self) -> None:
        self.initial = gpu_snapshot()
        self.peak = dict(self.initial)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.1):
            self.peak["memory_used_mib"] = max(
                self.peak["memory_used_mib"], gpu_snapshot()["memory_used_mib"]
            )

    def __enter__(self) -> "GPUMonitor":
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._stop.set()
        self._thread.join()
        self.peak["memory_used_mib"] = max(
            self.peak["memory_used_mib"], gpu_snapshot()["memory_used_mib"]
        )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


async def request_audio(
    client: httpx.AsyncClient,
    runtime: str,
    text: str,
    seed: int,
    save_audio: Path | None,
) -> dict:
    if runtime == "omni":
        payload = {
            "input": text,
            "stream": True,
            "response_format": "pcm",
            "seed": seed,
        }
    elif runtime in ("native", "akas"):
        payload = {
            "model": "zonos3",
            "input": text,
            "response_format": "pcm",
            "language": "en_us",
            "seed": seed,
            "max_tokens": 384,
            "stream_decode": "prefix",
            "decoder": "keep2_plain",
        }
    else:
        raise ValueError(runtime)
    started = time.perf_counter()
    first_audio = None
    chunks = []
    headers = {}
    try:
        async with client.stream("POST", "/v1/audio/speech", json=payload) as response:
            headers = dict(response.headers)
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                raise RuntimeError(f"HTTP {response.status_code}: {body[:2000]}")
            async for chunk in response.aiter_bytes():
                if chunk and first_audio is None:
                    first_audio = time.perf_counter()
                chunks.append(chunk)
        completed = time.perf_counter()
        audio = b"".join(chunks)
        sample_rate = int(
            headers.get("x-sample-rate", headers.get("x-audio-sample-rate", "0"))
        )
        channels = int(headers.get("x-channels", headers.get("x-audio-channels", "0")))
        sample_width = 2 if runtime == "omni" else 4
        if sample_rate != 44100 or channels != 1:
            raise RuntimeError(f"unexpected audio headers: {headers}")
        if len(audio) % (sample_width * channels) or not audio:
            raise RuntimeError(f"invalid PCM byte count: {len(audio)}")
        dtype = "<f4" if sample_width == 4 else "<i2"
        samples = np.frombuffer(audio, dtype=dtype)
        if not np.isfinite(samples).all():
            raise RuntimeError("audio contains nonfinite samples")
        peak = float(np.max(np.abs(samples)))
        if peak <= 1e-5:
            raise RuntimeError("audio is silent")
        duration_s = len(audio) / (sample_rate * channels * sample_width)
        if duration_s < 1.0 or first_audio is None:
            raise RuntimeError(f"invalid generated audio: {duration_s=}")
        if save_audio is not None:
            save_audio.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(save_audio), "wb") as wav:
                wav.setnchannels(channels)
                wav.setsampwidth(sample_width)
                wav.setframerate(sample_rate)
                wav.writeframes(audio)
        return {
            "ok": True,
            "seed": seed,
            "started_s": started,
            "completed_s": completed,
            "latency_s": completed - started,
            "ttfa_s": first_audio - started,
            "audio_duration_s": duration_s,
            "rtf": (completed - started) / duration_s,
            "audio_bytes": len(audio),
            "sample_rate": sample_rate,
            "sample_width_bytes": sample_width,
            "audio_peak": peak,
            "chunks": len(chunks),
            "audio_sha256": hashlib.sha256(audio).hexdigest(),
            "headers": headers,
        }
    except Exception as error:
        return {
            "ok": False,
            "seed": seed,
            "started_s": started,
            "completed_s": time.perf_counter(),
            "latency_s": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }


async def one_batch(
    runtime: str,
    port: int,
    concurrency: int,
    count: int,
    save_dir: Path | None,
) -> dict:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(connect=30.0, read=1800.0, write=60.0, pool=1800.0)
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}", limits=limits, timeout=timeout
    ) as client:
        tasks = [
            request_audio(
                client,
                runtime,
                TEXT,
                1000 + index,
                save_dir / f"request-{index:04d}.wav" if save_dir is not None else None,
            )
            for index in range(count)
        ]
        batch_started = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_s = time.perf_counter() - batch_started
    good = [result for result in results if result["ok"]]
    failed = [result for result in results if not result["ok"]]
    return {
        "status": "ok" if not failed else "error",
        "case": {
            "id": "delivery",
            "text": TEXT,
            "concurrency": concurrency,
            "requests": count,
        },
        "runtime": runtime,
        "concurrency": concurrency,
        "requests": count,
        "completed": len(good),
        "failed": len(failed),
        "wall_s": wall_s,
        "throughput_qps": len(good) / wall_s,
        "audio_seconds_per_second": sum(r["audio_duration_s"] for r in good) / wall_s,
        "latency_s": summary([r["latency_s"] for r in good]),
        "ttfa_s": summary([r["ttfa_s"] for r in good]),
        "rtf": summary([r["rtf"] for r in good]),
        "audio_duration_s": summary([r["audio_duration_s"] for r in good]),
        "results": results,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=["omni", "native", "akas"], required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrencies", default="1,4,16,64,256")
    parser.add_argument("--requests-per-concurrency", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--save-smoke", action="store_true")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--bundle", type=Path, help="Measured AKAS streaming bundle")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--num-pages", type=int)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--safe-slot-lifecycle", action="store_true")
    args = parser.parse_args()
    akas = args.runtime == "akas"
    if akas and args.history and not args.bundle:
        parser.error("--bundle is required to record an AKAS run")
    args.output.mkdir(parents=True, exist_ok=True)
    workload_path = Path("src/akas/benchmarking/workloads/speech-concurrent-v1.json")
    workload = json.loads(workload_path.read_text()) if workload_path.is_file() else None
    manifest = {
        "protocol": "client-speech-pcm-stream-v1",
        "speech_workload": {
            "id": "speech-concurrent-v1",
            "definition": workload,
            "sha256": hashlib.sha256(workload_path.read_bytes()).hexdigest() if workload else None,
        },
        "text": TEXT,
        "words": len(TEXT.split()),
        "utf8_bytes": len(TEXT.encode()),
        "model": "Zyphra/ZONOS3-Mini",
        "model_revision": MODEL_REVISION,
        "runtime": "zonos3-sgl" if args.runtime == "native" else args.runtime,
        "provider": "cuda",
        "device": {"name": "NVIDIA L40S", "memory_total_mib": 46068},
        "model_definition_sha256": hashlib.sha256(
            Path("benchmarks/models/zonos3-mini.json").read_bytes()
        ).hexdigest(),
        "weight_format": "bf16",
        "purpose": "akas-streaming-capacity" if akas else "native-capacity-diagnostic",
        "configuration": {
            "max_running_requests": args.max_requests,
            "num_pages": args.num_pages,
            "max_frames": args.max_frames,
            "request_bucketing": False,
            "weight_mmap": True,
            "codec": "keep2_plain",
            "compact_codec": {
                "path": os.environ.get("ZONOS3_COMPACT_CODEC"),
                "bytes": Path(os.environ["ZONOS3_COMPACT_CODEC"]).stat().st_size
                if os.environ.get("ZONOS3_COMPACT_CODEC")
                else None,
            }
            if os.environ.get("ZONOS3_COMPACT_CODEC")
            else None,
            "safe_slot_lifecycle": args.safe_slot_lifecycle,
        },
        "stream": True,
        "sample_rate_hz": 44100,
        "default_sampling": True,
        "default_voice_or_unconditioned": True,
        "decoder": "keep2_plain" if args.runtime != "omni" else None,
        "max_tokens": 384,
        "stream_decode": "prefix" if args.runtime != "omni" else None,
        "server_pid": args.server_pid,
        "gpu_memory_before": gpu_snapshot(),
        "concurrencies": [int(c) for c in args.concurrencies.split(",")],
        "requests_per_concurrency": args.requests_per_concurrency,
        "warmup": args.warmup,
        "started_unix_s": time.time(),
    }
    if akas:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.port}") as server:
            props = (await server.get("/props")).raise_for_status().json()
        manifest["server"] = props
        manifest["program_sha256"] = props["program_sha256"]
        if args.bundle:
            program = hashlib.sha256((args.bundle / "streaming.json").read_bytes()).hexdigest()
            if program != props["program_sha256"]:
                raise RuntimeError("server is not using the supplied streaming bundle")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    recorder = None
    if args.history:
        import sys

        sys.path.insert(0, "src")
        from akas.benchmarking.history import Recorder

        recorder = Recorder(
            args.history,
            manifest,
            kind="configuration" if akas else "baseline",
            change="Measure AKAS BF16 streaming capacity" if akas else "Measure native ZONOS3 streaming capacity",
            dependency_sources=None if akas else {"zonos3": "build/zonos3-concurrent-v1/ZONOS3-Mini"},
        )
        if akas:
            recorder.artifact(__file__, "bench_client.py")
            recorder.artifact(args.bundle / "streaming.json", "program/streaming.json")
            index = json.loads((args.bundle / "streaming.json").read_bytes())
            phases = index["phases"]
            for record in [
                *phases["prefill"].values(),
                *phases["decode"].values(),
                *(r for rows in phases["codec"].values() for r in rows.values()),
                phases["reset"],
            ]:
                name = f"{record['directory']}/forward.json"
                recorder.artifact(args.bundle / name, f"program/{name}")
    results = []
    for concurrency in manifest["concurrencies"]:
        count = args.requests_per_concurrency or concurrency
        if args.warmup:
            print(f"warmup c={concurrency} n={args.warmup}", flush=True)
            warmup = await one_batch(args.runtime, args.port, concurrency, args.warmup, None)
            (args.output / f"c{concurrency}-warmup.json").write_text(json.dumps(warmup, indent=2) + "\n")
            if warmup["failed"]:
                raise RuntimeError(f"warmup failed: {warmup['failed']} errors")
        print(f"measure c={concurrency} n={count}", flush=True)
        with GPUMonitor() as monitor:
            result = await one_batch(
                args.runtime,
                args.port,
                concurrency,
                count,
                args.output / f"c{concurrency}" / "audio" if args.save_smoke else None,
            )
        result["gpu_memory_initial"] = monitor.initial
        result["gpu_memory_peak"] = monitor.peak
        if args.server_pid:
            pids = subprocess.check_output(
                ["ps", "-eo", "pid=,ppid="], text=True
            ).splitlines()
            children = {
                int(line.split()[0])
                for line in pids
                if int(line.split()[1]) == args.server_pid
            }
            result["server_processes_rss_mib"] = {}
            for pid in [args.server_pid, *sorted(children)]:
                try:
                    pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
                except (FileNotFoundError, ProcessLookupError):
                    continue
                result["server_processes_rss_mib"][str(pid)] = (
                    pages * os.sysconf("SC_PAGE_SIZE") // (1024 * 1024)
                )
        (args.output / f"c{concurrency}" / "result.json").parent.mkdir(parents=True, exist_ok=True)
        (args.output / f"c{concurrency}" / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
        if recorder:
            recorder.update({"schema": "akas.benchmark.v1", "metadata": manifest, "results": results})
        if result["failed"]:
            if recorder:
                if args.server_log:
                    recorder.artifact(args.server_log, "server.log")
                recorder.update(
                    {"schema": "akas.benchmark.v1", "metadata": manifest, "results": results},
                    final=True,
                    status="failed",
                )
            raise RuntimeError(f"measured batch failed: {result['failed']} errors")
    if recorder:
        if args.server_log:
            recorder.artifact(args.server_log, "server.log")
        recorder.update(
            {"schema": "akas.benchmark.v1", "metadata": manifest, "results": results},
            final=True,
            status="passed" if all(row["status"] == "ok" for row in results) else "failed",
        )
        print(f"recorded {recorder.id}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
