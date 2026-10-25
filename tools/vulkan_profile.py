"""Attribute a compiled WebGPU phase using synchronized prepared call groups.

These isolated hot-buffer timings diagnose bottlenecks; they are not additive
end-to-end inference measurements. No generated model state is used as a quality
reference. Keep each original kernel and buffer binding intact.
"""

import argparse
from collections import Counter
from pathlib import Path

from akas.benchmarking.history import digest, write
from akas.execution.plan import ExecutionPlan
from akas.models.lfm2.model import LFM2
from tools.context_sweep import vulkan_device


def profile(args):
    from tensor.compiler.tuning import measure
    import numpy as np
    from akas.memory.transfer import write as transfer

    result = dict(program_sha256=digest(args.program / "program.json"),
                  chunk=args.chunk, context_position=args.position,
                  scope="isolated hot-buffer prepared kernels; diagnostic only", kernels=[])
    with vulkan_device() as device:
        with LFM2(args.checkpoint, args.program, device, graphs=True,
                  checkpoint_options={"tokenizer": False}) as model:
            slot = model.admit()
            frame = model.batches.select(1, args.chunk)
            positions = [args.position]
            frame.prepare([slot], [[1] * args.chunk], positions)
            transfer(device, frame.buffers["positions"], np.array(positions, dtype="int32"))
            frame.forward.run()
            device.synchronize()
            counts = Counter(call.key for call in frame.forward.definition.calls)
            seen = set()
            for graph_call, entry in zip(frame.forward.definition.calls, frame.forward.plan):
                if graph_call.key in seen:
                    continue
                seen.add(graph_call.key)
                plan = ExecutionPlan(device, [entry] * 16)
                prepared = plan.capture(resources=tuple(frame.buffers.values()) + (entry[0],))
                try:
                    timing = measure(device, prepared.launch, warmup=2, samples=5, repeats=1)
                finally:
                    prepared.close()
                row = dict(kind=graph_call.kind, parameters=graph_call.parameters,
                           count=counts[graph_call.key], timing=timing)
                row["dispatches_per_plan"] = 16
                row["weighted_seconds"] = counts[graph_call.key] * timing["median_seconds"] / 16
                result["kernels"].append(row)
                write(args.out, result)
            result["device"] = device.info
    result["kernels"].sort(key=lambda r:r["weighted_seconds"], reverse=True)
    write(args.out, result)
    for row in result["kernels"]:
        print(row["kind"], row["parameters"], row["count"],
              round(row["weighted_seconds"] * 1000, 3), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--position", type=int, default=2048)
    parser.add_argument("--out", type=Path, required=True)
    profile(parser.parse_args())
