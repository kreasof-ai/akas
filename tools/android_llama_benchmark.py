"""Deploy a pinned CPU-only llama.cpp runner and retain Android measurements."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

REVISION = "8ca4060b7e30e76baaa5670ee1ec1b760b7f8552"
CHECKPOINT_SHA256 = "43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adb", type=Path, required=True)
    p.add_argument("--serial", required=True)
    p.add_argument("--binary", type=Path, default=Path("build/android/llama-build/akas-llama-bench"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-threads", type=int, default=4)
    p.add_argument("--decode-mask", default="f0", help="CPU affinity mask in hex; amber big cores are 4-7")
    p.add_argument("--batch-mask", default="ff", help="CPU affinity mask in hex")
    p.add_argument("--generated", type=int, default=32)
    p.add_argument("--prefixes", type=int, nargs="+", default=[32, 512, 2048, 8192])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    for mask in (a.decode_mask, a.batch_mask):
        if not 0 < int(mask, 16) < 2**64:
            p.error("CPU masks must be nonzero 64-bit hex values")
    if (min(a.threads, a.batch_threads) < 1 or a.generated < 2 or
            not a.prefixes or min(a.prefixes) < 1 or max(a.prefixes) + a.generated > 8320):
        p.error("invalid workload or thread count")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    inputs = a.out.parent / "llama-inputs"
    inputs.mkdir(exist_ok=True)
    fingerprints = {}
    for prefix in sorted({32, *a.prefixes}):
        rng = np.random.default_rng(17 + 4099 + prefix)
        tokens = rng.integers(0, 128000, prefix, dtype="int32")
        path = inputs / f"tokens-{prefix}.bin"
        path.write_bytes(tokens.astype("<i4").tobytes())
        fingerprints[str(prefix)] = hashlib.sha256(path.read_bytes()).hexdigest()
    adb = [str(a.adb), "-s", a.serial]
    def call(*args):
        return subprocess.check_output([*adb, *args], text=True, stderr=subprocess.STDOUT)
    observed = call("shell", "sha256sum /data/local/tmp/akas/lfm3-qad-q4_0.gguf").split()[0]
    if observed != CHECKPOINT_SHA256:
        raise ValueError("phone checkpoint differs from the official pinned QAD Q4_0")
    call("push", str(a.binary), "/data/local/tmp/akas/akas-llama-bench")
    call("push", str(inputs), "/data/local/tmp/akas/")
    call("shell", "chmod 755 /data/local/tmp/akas/akas-llama-bench")
    command = (
        f"cd /data/local/tmp/akas && AKAS_LLAMA_DECODE_CPUS={int(a.decode_mask, 16):x} "
        f"AKAS_LLAMA_BATCH_CPUS={int(a.batch_mask, 16):x} "
        "./akas-llama-bench lfm3-qad-q4_0.gguf llama-inputs "
        f"{a.threads} {a.batch_threads} {a.generated} " + " ".join(map(str, a.prefixes))
    )
    report = dict(
        schema="akas.android-llama.v1", status="running", cases=[],
        metadata=dict(runtime="llama.cpp", provider="cpu", revision=REVISION,
                      binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
                      checkpoint_sha256=observed, prompt_file_sha256=fingerprints,
                      threads=a.threads, batch_threads=a.batch_threads, max_context=8320,
                      decode_cpu_mask=a.decode_mask, batch_cpu_mask=a.batch_mask,
                      microbatch=512, kv_dtype="float16", output_tokens=a.generated,
                      sampling="greedy, ignore EOS", repeats=1,
                      warmup="one 32-token prefix and 3 generated tokens",
                      timing="synchronized native C API wall time, including greedy sampling; loading excluded",
                      command=command),
    )
    def save():
        a.out.write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        with a.out.with_suffix(".log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([*adb, "shell", command], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith("RESULT "):
                    row = json.loads(line.removeprefix("RESULT "))
                    report["cases"].append(row)
                    save()
                    print(line.rstrip(), flush=True)
                elif line.startswith(("WARMUP", "MEASURE", "ERROR", "CPU FEATURES")):
                    print(line.rstrip(), flush=True)
            code = process.wait()
            if code:
                raise RuntimeError(f"phone benchmark exited with status {code}; see {a.out.with_suffix('.log')}")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
