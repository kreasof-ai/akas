"""Measure synchronized per-call latency, including synchronization overhead.

Use this for rough bottleneck discovery; sums do not predict whole-plan time.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import tensor
from akas.execution.plan import submit
from akas.models.lfm3 import LFM3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--limit", type=int, default=18, help="prefill calls; decode profiles the complete phase")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = dict(schema="akas.android-kernel-profile.v1",
                  timing="per-call host wall time with synchronization; not GPU timestamps", cases=[])
    with tensor.Device(provider="webgpu", max_buffer_size=268435456) as device:
        with LFM3(args.checkpoint, args.program, device, checkpoint_options={"tokenizer": False}) as model:
            for chunk in args.chunks:
                frame = model.batches.select(1, chunk)
                slot = model.admit()
                rng = np.random.default_rng(17 + 4099 + chunk)
                tokens = rng.integers(0, model.config.vocab, chunk, dtype="int32").astype(int).tolist()
                frame.prepare([slot], [tokens], model.positions)
                for i, (entry, call) in enumerate(zip(frame.forward.plan, frame.forward.definition.calls)):
                    if chunk > 1 and i >= args.limit:
                        break
                    device.synchronize()
                    start = time.perf_counter()
                    submit(device, [entry])
                    device.synchronize()
                    seconds = time.perf_counter() - start
                    row = dict(chunk=chunk, index=i, kind=call.kind, parameters=call.parameters, seconds=seconds)
                    report["cases"].append(row)
                    if call.kind in ("linear", "ffn", "window_grouped", "argmax"):
                        print("PROFILE", chunk, i, call.kind, call.parameters.get("k"), call.parameters.get("o"), seconds, flush=True)
                    args.out.write_text(json.dumps(report, indent=2) + "\n")
                model.release(slot)
    print("PROFILE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
