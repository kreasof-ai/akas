"""llama.cpp localhost-server preflight using the common worker protocol."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from tools.baselines.common import main, measurements, synthetic_prompts


def run(config):
    capacity = max(b for b, _ in config["shapes"])
    context = max(p for _, p in config["shapes"]) + config["output_tokens"]
    command = [
        config["binary"],
        "-m",
        config["checkpoint"],
        "-ngl",
        "99",
        "--device",
        "CUDA0" if config["provider"] == "cuda" else "Vulkan0",
        "-c",
        str(context * capacity),
        "-np",
        str(capacity),
        "-b",
        "2048",
        "-ub",
        "512",
        "-t",
        "2",
        "-fa",
        "on",
        # The workload starts from empty prefixes. Retained prompt copies and
        # recurrent-state checkpoints otherwise consume host RAM across bursts.
        "--cache-ram",
        "0",
        "--ctx-checkpoints",
        "0",
        "--no-cache-idle-slots",
        "--host",
        "127.0.0.1",
        "--port",
        "8817",
        "--no-webui",
    ]
    report = dict(
        runtime="llama.cpp",
        command=command,
        revision=config["runtime_revision"],
        cases=[],
    )
    listing = subprocess.check_output(
        [config["binary"], "--list-devices"],
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    selected = "CUDA0:" if config["provider"] == "cuda" else "Vulkan0:"
    if not any(
        selected in line and config["device_name"] in line
        for line in listing.splitlines()
    ):
        raise RuntimeError(
            "requested NVIDIA GPU is absent from the baseline device list: " + listing
        )
    report["device_listing"] = listing
    process = None

    def request(path, data=None):
        req = urllib.request.Request(
            "http://127.0.0.1:8817" + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response)

    try:
        process = subprocess.Popen(
            command, stdout=None, stderr=None, start_new_session=False
        )
        startup_seconds = config.get("deadline_seconds", 120)
        until = time.monotonic() + startup_seconds
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited: {process.returncode}")
            try:
                if request("/health").get("status") == "ok":
                    break
            except Exception:
                if time.monotonic() > until:
                    raise TimeoutError(f"server startup exceeded {startup_seconds}s")
                time.sleep(0.25)
        from concurrent.futures import ThreadPoolExecutor

        def invoke(batch, prefix):
            prompts = synthetic_prompts(config["vocab"], batch, prefix)
            payload = dict(
                prompt=None,
                n_predict=config["output_tokens"],
                temperature=0,
                seed=17,
                ignore_eos=True,
                cache_prompt=False,
                return_tokens=True,
            )
            with ThreadPoolExecutor(max_workers=batch) as executor:
                outputs = list(
                    executor.map(
                        lambda ids: request("/completion", {**payload, "prompt": ids}),
                        prompts,
                    )
                )
            if any(o["tokens_predicted"] != config["output_tokens"] for o in outputs):
                raise ValueError("llama.cpp returned the wrong output count")
            timings = [o["timings"] for o in outputs]
            # llama.cpp exposes server eval phases, not client TTFT. Keep that
            # protocol distinct from request timestamp metrics in other runtimes.
            return [o["tokens"] for o in outputs], dict(
                responses=outputs,
                phase_protocol="llama-server-eval-v1",
                prefill_tokens_per_second=batch
                * prefix
                / max(t["prompt_ms"] / 1000 for t in timings),
                decode_tokens_per_second=batch
                * (config["output_tokens"] - 1)
                / max(t["predicted_ms"] / 1000 for t in timings),
            )

        measurements(config, invoke, report)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return report


if __name__ == "__main__":
    main(run)
