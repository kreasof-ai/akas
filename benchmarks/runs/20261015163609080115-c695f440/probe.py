"""llama.cpp localhost-server preflight using the common worker protocol."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from tools.baselines.common import main


def run(config):
    command = [
        config["binary"],
        "-m",
        config["checkpoint"],
        "-ngl",
        "99",
        "--device",
        "CUDA0" if config["provider"] == "cuda" else "Vulkan0",
        "-c",
        "2304",
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
        with urllib.request.urlopen(req, timeout=90) as response:
            return json.load(response)

    try:
        process = subprocess.Popen(
            command, stdout=None, stderr=None, start_new_session=False
        )
        until = time.monotonic() + 120
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited: {process.returncode}")
            try:
                if request("/health").get("status") == "ok":
                    break
            except Exception:
                if time.monotonic() > until:
                    raise TimeoutError("server startup exceeded 120s")
                time.sleep(0.25)
        seed = request(
            "/tokenize",
            dict(
                content="The capital of France is Paris. Explain the history of this city briefly. ",
                add_special=False,
            ),
        )["tokens"]
        for batch, prefix in config["shapes"]:
            if batch != 1:
                continue  # Server concurrency is not a controlled native batch.
            ids = (seed * ((prefix + len(seed) - 1) // len(seed)))[:prefix]
            payload = dict(
                prompt=ids,
                n_predict=config["output_tokens"],
                temperature=0,
                seed=17,
                ignore_eos=True,
                cache_prompt=False,
                return_tokens=True,
            )
            request("/completion", payload)
            start = time.perf_counter()
            output = request("/completion", payload)
            elapsed = time.perf_counter() - start
            assert output["tokens_predicted"] == config["output_tokens"], output
            report["cases"].append(
                dict(
                    batch=1,
                    prefix=prefix,
                    output=config["output_tokens"],
                    wall_seconds=elapsed,
                    response=output,
                    status="passed",
                )
            )
            print("CASE", prefix, elapsed, flush=True)
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
