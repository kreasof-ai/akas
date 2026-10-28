"""Run a real text request on LFM3 QAD Q4_0 with both local Vulkan runtimes."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from akas.benchmarking.history import digest, write
from akas.benchmarking.session import Session
from akas.models.lfm3 import LFM3, LFM3Checkpoint
from tools.context_sweep import vulkan_device
from tools.lfm3_benchmark import PROFILE
from tools.model_benchmark import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROFILE.checkpoint)
    parser.add_argument(
        "--program", type=Path, default=Path("build/lfm3-vulkan/optimized")
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("build/baselines/llama-b11946-vulkan/llama-server.exe"),
    )
    parser.add_argument(
        "--prompt", default="What is the capital of France? Answer briefly."
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--port", type=int, default=8827)
    parser.add_argument("--out", type=Path, default=Path("build/lfm3-vulkan/demo.json"))
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if digest(args.checkpoint) != PROFILE.checkpoint_sha256:
        raise ValueError("checkpoint differs from the official QAD Q4_0 file")
    checkpoint = LFM3Checkpoint(args.checkpoint)
    tok = checkpoint.tokenizer
    ids = tok.chat(args.prompt)
    program = json.loads((args.program / "program.json").read_text(encoding="utf-8"))
    chunk = program["model_options"]["prefill_chunk"]
    context = program["max_context"]
    if len(ids) + args.max_tokens > context:
        parser.error("request exceeds compiled context")
    result = dict(
        prompt=args.prompt,
        prompt_ids=ids,
        checkpoint_sha256=PROFILE.checkpoint_sha256,
        program_sha256=digest(args.program / "program.json"),
        sampling=dict(temperature=0, repeat_penalty=1),
        note="Text demonstration; host readback and HTTP timings are not the native phase benchmark.",
    )
    with vulkan_device() as device:
        result["device"] = dict(device.info)
        with LFM3(args.checkpoint, args.program, device, graphs=True) as model:
            with Session(model, batch=1, chunk=chunk) as session:
                session.reset()
                start = time.perf_counter()
                output = [int(session.prefill([ids])[0])]
                while len(output) < args.max_tokens and output[-1] != tok.eos:
                    output.append(int(session.step([output[-1]])[0]))
                result["akas"] = dict(
                    output_ids=output,
                    text=tok.decode(output),
                    wall_seconds=time.perf_counter() - start,
                )
                print("AKAS:", result["akas"]["text"], flush=True)
    write(args.out, result)
    command = [
        str(args.binary.resolve()),
        "-m",
        str(args.checkpoint.resolve()),
        "-ngl",
        "99",
        "--device",
        "Vulkan0",
        "-c",
        str(context),
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
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--cache-ram",
        "0",
        "--ctx-checkpoints",
        "0",
        "--no-cache-idle-slots",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--no-webui",
    ]
    result["llama_binary_sha256"] = digest(args.binary)
    result["llama_command"] = command
    base = f"http://127.0.0.1:{args.port}"
    with args.out.with_suffix(".server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            until = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError("llama-server exited; inspect the server log")
                try:
                    if request(base, "/health").get("status") == "ok":
                        break
                except Exception:
                    if time.monotonic() > until:
                        raise TimeoutError("llama startup")
                    time.sleep(0.25)
            response = request(
                base,
                "/completion",
                dict(
                    prompt=ids,
                    n_predict=args.max_tokens,
                    temperature=0,
                    repeat_penalty=1,
                    presence_penalty=0,
                    frequency_penalty=0,
                    seed=17,
                    cache_prompt=False,
                    return_tokens=True,
                ),
            )
            result["llama"] = response
            left, right = result["akas"]["output_ids"], response["tokens"]
            result["common_prefix_tokens"] = next(
                (i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)),
            )
            print("llama.cpp:", response["content"], flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    write(args.out, result)
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
