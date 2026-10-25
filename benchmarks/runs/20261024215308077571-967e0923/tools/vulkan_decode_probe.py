"""Synthetic long-cache decode latency probe; never a text/quality benchmark.

Weights and qualified kernels are real. A Gaussian KV cache is assigned a
logical prefix length without prefill, so the result only screens schedules.
Always confirm selected programs on the retained real-text context sweep.
"""

import argparse
from pathlib import Path

import numpy as np

from akas.benchmarking.history import digest, write
from akas.memory.transfer import write as transfer
from akas.models.lfm2.model import LFM2
from tools.context_sweep import vulkan_device


def main():
    from tensor.compiler.tuning import measure

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--positions", type=int, nargs="+", default=[32, 2048, 65536, 130944])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = dict(scope="synthetic Gaussian KV decode; no text or quality claim", seed=47,
                  program_sha256=digest(args.program / "program.json"), results=[])
    with vulkan_device() as device:
        with LFM2(args.checkpoint, args.program, device, graphs=True,
                  checkpoint_options={"tokenizer": False}) as model:
            slot = model.admit()
            frame = model.batches.select(1, 1)
            rng = np.random.default_rng(47)
            upload = device.prepare_plan(())
            try:
                for name, buffer in sorted(frame.buffers.items()):
                    if name.endswith((".kc", ".vc")):
                        transfer(device, buffer, (rng.standard_normal(buffer.shape,
                            dtype="float32") * .1).astype("float16"))
                        upload.launch()
                        device.synchronize()
            finally:
                upload.close()
            for position in args.positions:
                def reset():
                    model.batches.reset(slot)
                    model.positions[slot] = position
                    frame.prepare([slot], [[1]], model.positions)
                    transfer(device, frame.buffers["positions"], np.array([position], "int32"))

                reset()
                timing = measure(device, lambda: model.decode_steps([slot], [1], 128, read=False),
                                 reset=reset, warmup=1, samples=3, repeats=1)
                row = dict(position=position, steps=128, timing=timing,
                           decode_tok_s=128 / timing["median_seconds"])
                result["results"].append(row)
                write(args.out, result)
                print(position, row["decode_tok_s"], flush=True)


if __name__ == "__main__":
    main()
