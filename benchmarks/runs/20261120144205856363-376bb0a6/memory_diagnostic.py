"""Use the AKAS producer and benchmark worker to diagnose projection precision."""

import csv
import ctypes as c
import json
import os
from pathlib import Path
import sys
from threading import Event, Thread
from time import monotonic

import numpy as np
import psutil

from akas.artifacts.write import checksum
from akas.benchmarking.enhancement import errors
from akas.benchmarking.history import Recorder
from akas.compilation.compile import compile_forwards
from akas.compilation.tuning import SearchSettings
from akas.models.voice_isolator import VoiceIsolator
from akas.models.voice_isolator.audio import float_wav, wav_samples
from akas.models.voice_isolator.weights import VoiceIsolatorCheckpoint
from akas.tensor.device import vulkan_device
from tools.baselines.enhancement import worker


class Value(c.Structure):
    _fields_ = [("status", c.c_uint32), ("number", c.c_int64)]


class Item(c.Structure):
    _fields_ = [("name", c.c_wchar_p), ("value", Value)]


class GpuMemory:
    """WDDM per-process memory counters, over every adapter instance."""

    def __init__(self):
        self.api = c.WinDLL("pdh")
        self.api.PdhOpenQueryW.argtypes = [
            c.c_wchar_p, c.c_size_t, c.POINTER(c.c_void_p)
        ]
        self.api.PdhAddEnglishCounterW.argtypes = [
            c.c_void_p, c.c_wchar_p, c.c_size_t, c.POINTER(c.c_void_p)
        ]
        self.api.PdhCollectQueryData.argtypes = [c.c_void_p]
        self.api.PdhCloseQuery.argtypes = [c.c_void_p]
        self.api.PdhGetFormattedCounterArrayW.argtypes = [
            c.c_void_p, c.c_uint32, c.POINTER(c.c_uint32), c.POINTER(c.c_uint32),
            c.c_void_p,
        ]
        self.api.PdhGetFormattedCounterArrayW.restype = c.c_uint32
        self.query = c.c_void_p()
        self.check(self.api.PdhOpenQueryW(None, 0, c.byref(self.query)))
        self.counters = {}
        for name in ("Dedicated Usage", "Shared Usage", "Total Committed"):
            handle = c.c_void_p()
            self.check(self.api.PdhAddEnglishCounterW(
                self.query, rf"\GPU Process Memory(*)\{name}", 0, c.byref(handle)
            ))
            self.counters[name] = handle

    @staticmethod
    def check(status):
        if status:
            raise OSError(f"WDDM counter status {status & 0xffffffff:#010x}")

    def sample(self, pid):
        self.check(self.api.PdhCollectQueryData(self.query))
        totals = {}
        prefix = f"pid_{pid}_"
        for name, handle in self.counters.items():
            size, count = c.c_uint32(), c.c_uint32()
            status = self.api.PdhGetFormattedCounterArrayW(
                handle, 0x400, c.byref(size), c.byref(count), None
            )
            if status != 0x800007D2:
                continue
            buf = c.create_string_buffer(size.value)
            status = self.api.PdhGetFormattedCounterArrayW(
                handle, 0x400, c.byref(size), c.byref(count), buf
            )
            if status:
                continue
            items = c.cast(buf, c.POINTER(Item))
            values = [
                items[i].value.number for i in range(count.value)
                if items[i].name.startswith(prefix) and items[i].value.status in (0, 1)
            ]
            if values:
                totals[name] = sum(values)
        return totals

    def close(self):
        self.api.PdhCloseQuery(self.query)


FIELDS = (
    "seconds", "stage", "worker_pid", "host_working_set_bytes", "host_private_bytes",
    "dedicated_gpu_bytes", "shared_gpu_bytes", "committed_gpu_bytes",
)


def sample_memory(path, stop, state, peaks):
    source = GpuMemory()
    parent = psutil.Process()
    maximum = lambda old, new: max(old if old is not None else 0, new)
    start = monotonic()
    try:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            while not stop.is_set():
                candidates = parent.children(recursive=True)
                for process in candidates:
                    try:
                        if "tools.baselines.enhancement" not in process.cmdline():
                            continue
                        stage = state["stage"]
                        host = process.memory_info()
                        gpu = source.sample(process.pid)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                    values = dict(
                        host_working_set_bytes=host.rss,
                        host_private_bytes=host.private,
                        dedicated_gpu_bytes=gpu.get("Dedicated Usage"),
                        shared_gpu_bytes=gpu.get("Shared Usage"),
                        committed_gpu_bytes=gpu.get("Total Committed"),
                    )
                    writer.writerow(dict(
                        seconds=round(monotonic() - start, 6),
                        stage=stage, worker_pid=process.pid, **values,
                    ))
                    stream.flush()
                    record = peaks.setdefault(stage, dict(samples=0, gpu_samples=0))
                    record["samples"] += 1
                    record["gpu_samples"] += "Dedicated Usage" in gpu
                    for key, value in values.items():
                        if value is not None:
                            record[key] = maximum(record.get(key), value)
                stop.wait(0.20)
    finally:
        source.close()


ROOT = Path("build/voice-isolator").resolve()
DATA = Path("build/datasets/voice-isolator-golden").resolve()
CHECKPOINT = Path("build/checkpoints/voice-isolator-v1/denoiser.safetensors").resolve()
CANDIDATES = (
    ("mamba", ("mamba_coefficients",)),
    ("time-qkv", ("time_qkv",)),
    ("decoder", ("decoder_output",)),
    ("combined", ("mamba_coefficients", "time_qkv", "decoder_output")),
)


def qualify(label, bundle, r, *, compact_scratch=True, alternate=False):
    prefix = f"profiles/{label}"
    out = r.directory / prefix
    out.mkdir(parents=True)
    r.artifact(bundle / "speech.json", f"{prefix}/speech.json")
    index = json.loads((bundle / "speech.json").read_text())
    for phase in index["phases"]["enhance"].values():
        r.artifact(bundle / phase["path"], f"{prefix}/{phase['path']}")
    configuration = dict(
        factory="akas.models.voice_isolator:service",
        options=dict(
            checkpoint=str(CHECKPOINT), bundle=str(bundle),
            compact_scratch=compact_scratch,
        ),
    )
    command = [
        sys.executable, "-u", "-m", "tools.baselines.enhancement", "akas",
        json.dumps(configuration),
    ]
    trace, log = out / "memory.csv", out / "worker.log"
    stop = Event()
    state, peaks, monitor_errors = dict(stage="load"), {}, []

    def monitor():
        try:
            sample_memory(trace, stop, state, peaks)
        except Exception as e:
            monitor_errors.append(str(e))

    sampler = Thread(target=monitor, daemon=True)
    sampler.start()
    result = dict(label=label, program_sha256=checksum(bundle / "speech.json"), checks=[])
    plan = [("golden-3s", "", 1), ("golden-11s", "_11s", 2)]
    if alternate:
        plan.append(("golden-3s-after-long", "", 1))
    try:
        with worker(command, log, monotonic()+180, env=dict(os.environ), cwd=Path.cwd()) as (
            invoke, props,
        ):
            result["model"] = props
            for n, (case, suffix, chunks) in enumerate(plan, 1):
                state["stage"] = case
                wav, fixture = DATA / f"input{suffix}.wav", DATA / f"expected{suffix}.npy"
                arrays, _ = invoke(wav, 0, n)
                expected = np.load(fixture)
                with np.load(arrays) as values:
                    voice, background = values["voice"], values["background"]
                    check = errors(voice, expected)
                    check.update(
                        case=case, chunks=int(values["chunks"]),
                        limit_exceedances=int(np.count_nonzero(np.abs(voice-expected)>1e-4)),
                        reconstructs_input=bool(np.allclose(
                            voice+background, wav_samples(wav.read_bytes())[0],
                            rtol=0, atol=1e-6,
                        )),
                    )
                    check["passed"] = (
                        check["finite"] and check["max_abs_error"] <= 1e-4
                        and check["chunks"] == chunks and check["reconstructs_input"]
                    )
                    result["checks"].append(check)
                    print(label, case, "max error:", check["max_abs_error"], "pass:", check["passed"], flush=True)
                    if check["finite"]:
                        for name, data in (("voice", voice), ("background", background)):
                            dest = out / f"{case}-{name}.wav"
                            dest.write_bytes(float_wav(data))
                            r.artifact(dest, f"{prefix}/{dest.name}")
    finally:
        stop.set()
        sampler.join(timeout=4)
        if trace.is_file():
            r.artifact(trace, f"{prefix}/memory.csv")
        if log.is_file():
            r.artifact(log, f"{prefix}/worker.log")
    result.update(memory_peak_by_stage=peaks, memory_errors=monitor_errors)
    result["qualified"] = len(result["checks"]) == len(plan) and all(
        check["passed"] for check in result["checks"]
    )
    print(label, "memory:", peaks, monitor_errors, flush=True)
    return result


def main():
    if sys.argv[1:] == ["--test-counters"]:
        source = GpuMemory()
        try:
            import subprocess
            pid = int(subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-Process -Name dwm | Select-Object -First 1 -ExpandProperty Id)"],
                text=True
            ).strip())
            print(pid, source.sample(pid))
        finally:
            source.close()
        return
    bundle = ROOT / "w16a16-mixed"
    metadata = dict(
        purpose="Voice Isolator FP32 projection exceptions and WDDM memory use",
        provider="webgpu", tolerance=1e-4, memory_sample_interval_seconds=0.20,
        profiles=[],
    )
    r = Recorder(
        ROOT / "history", metadata, dependency_sources={"tensor": Path("../tensor")}
    )
    status = "failed"
    print(r.directory, flush=True)
    try:
        r.artifact(__file__, "diagnostic.py")
        for case, suffix in (("golden-3s", ""), ("golden-11s", "_11s")):
            r.artifact(DATA / f"input{suffix}.wav", f"inputs/{case}.wav")
            r.artifact(DATA / f"expected{suffix}.npy", f"reference/{case}.npy")
        for label in ("f32-fused", "w16a16"):
            metadata["profiles"].append(qualify(label, ROOT / label, r))
        winner = None
        for label, groups in CANDIDATES:
            checkpoint = VoiceIsolatorCheckpoint(
                CHECKPOINT, weight_dtype="float16", fp32_projections=groups
            )
            try:
                with vulkan_device() as device:
                    compile_forwards(
                        checkpoint, bundle, definition=VoiceIsolator, device=device,
                        search_settings=SearchSettings(budget=3,width=3,warmup=1,samples=1,repeats=1),
                    )
                outcome = qualify(label, bundle, r)
            except (OSError, ValueError, RuntimeError) as e:
                print(label, str(e), flush=True)
                metadata["profiles"].append(dict(label=label, fp32_projections=groups, error=str(e)))
                continue
            metadata["profiles"].append(outcome)
            if outcome["qualified"]:
                size = outcome["model"]["buffer_bytes"]["weights_and_constants"]
                if winner is None or size < winner[0]:
                    winner = size, label
            r.update(dict(schema="akas.benchmark.v1", metadata=metadata, results=[]))
        if winner:
            label = winner[1]
            payload = (r.directory / f"profiles/{label}/speech.json").read_bytes()
            (bundle / "speech.json").write_bytes(payload)
            metadata["qualified_winner"] = label
            print(f"Qualified mixed policy: {label}", flush=True)
        else:
            print("No mixed policy passed both golden checks; no performance run.", flush=True)
        status = "passed"
    except BaseException as e:
        metadata["error"] = str(e)
        raise
    finally:
        record = r.update(
            dict(schema="akas.benchmark.v1", metadata=metadata, results=[]),
            final=True, status=status,
        )
        print("Summary:", r.directory / "report.json", record["status"], flush=True)
    if record["status"] != "passed":
        return 1
    return 0 if winner else 2


if __name__ == "__main__":
    raise SystemExit(main())
