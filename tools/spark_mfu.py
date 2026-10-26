"""Estimate useful forward FLOPs and prefill MFU from a Spark campaign report.

Counts dense projections, one final-row vocabulary projection, and useful causal
QK/PV pairs. Excludes embedding lookups, elementwise work, padding, dequantization,
and cache/control traffic. Counts every layer even when intermediate prefill
chunks prune the final layer's output. This is an algorithmic estimate, not a
hardware instruction counter. The default peak is L4 dense FP16 (121 TFLOP/s),
without the 2:4 sparsity multiplier.
"""

import argparse
import json
import math
from pathlib import Path

from akas.benchmarking.history import write

PEAK_SOURCE = "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf"


def prefill_flops(config, prefix):
    if type(prefix) is not int or prefix < 1:
        raise ValueError("prefix must be a positive integer")
    c = config
    projections = len(c.layers) * (
        c.width * (c.heads + 2 * c.kv_heads) * c.head_dim
        + c.width * c.heads
        + c.width * c.heads * c.head_dim
        + 3 * c.width * c.ff
    )
    full_pairs = prefix * (prefix + 1) // 2
    width = min(prefix, c.window)
    window_pairs = width * (width + 1) // 2 + (prefix - width) * width
    pairs = sum(
        window_pairs if kind == "sliding_attention" else full_pairs for kind in c.layers
    )
    return (
        2 * prefix * projections
        + 2 * c.width * c.vocab
        + 4 * c.heads * c.head_dim * pairs
    )


def estimate(report, config, peak_tflops=121):
    if not math.isfinite(peak_tflops) or peak_tflops <= 0:
        raise ValueError("dense peak TFLOP/s must be finite and positive")
    rows = []
    for profile in report["profiles"]:
        for row in profile["measurements"]:
            if row["batch"] != 1:
                raise ValueError("this estimate requires a single request")
            n, rate = row["prefix"], row["prefill_tok_s"]
            work = prefill_flops(config, n)
            tflops = work * rate / n / 1e12
            rows.append(
                dict(
                    runtime=profile["runtime"],
                    prefix=n,
                    prefill_tok_s=rate,
                    useful_flops=work,
                    tflops=tflops,
                    mfu_percent=100 * tflops / peak_tflops,
                )
            )
    return dict(
        schema="akas.spark-prefill-mfu.v1",
        dense_peak_tflops=peak_tflops,
        peak_source=PEAK_SOURCE,
        method=__doc__,
        rows=rows,
    )


def main():
    from akas.models.spark import SparkCheckpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--peak-tflops", type=float, default=121)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    c = SparkCheckpoint(args.checkpoint, tokenizer=False).config
    result = estimate(json.loads(args.report.read_text()), c, args.peak_tflops)
    if args.out:
        write(args.out, result)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
