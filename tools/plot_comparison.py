"""Render the README LFM2.5-24B-A2B-FP8 multi-batch benchmark figure.

Run: python -m tools.plot_comparison REPORT_JSON OUTPUT_SVG
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ID = "20261101T074055Z-ad51ac551826"


def _load_report(report):
    if report is not None:
        path = Path(report)
        if path.is_dir():
            path = path / "report.json"
        return json.loads(path.read_text(encoding="utf-8"))
    from akas.benchmarking import archive

    raw = archive.read_file(ROOT / "benchmarks", "reports", DEFAULT_REPORT_ID, "report.json")
    return json.loads(raw)


def plot(report=None, output=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    data = _load_report(report)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "path",
            "svg.hashsalt": "akas-lfm25-fp8-showcase",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    series = (
        ("akas", "AKAS · FP16 KV", "#2563eb", "o", "-"),
        ("vllm", "vLLM · BF16 KV", "#ea580c", "s", "--"),
        ("sglang", "SGLang · BF16 KV", "#0d9488", "^", ":"),
    )
    phases = (
        ("prefill_tok_s", "prefill_tokens_per_second", "Prefill"),
        ("decode_tok_s", "decode_tokens_per_second", "Decode"),
    )
    for axis, (metric, sample_key, title) in zip(axes, phases):
        for runtime, label, color, marker, style in series:
            profile = next(p for p in data["profiles"] if p["runtime"] == runtime)
            rows = sorted(profile["measurements"], key=lambda row: row["batch"])
            batches = [row["batch"] for row in rows]
            values = [row[metric] for row in rows]
            low = [
                row[metric] - min(s[sample_key] for s in row["samples"]) for row in rows
            ]
            high = [
                max(s[sample_key] for s in row["samples"]) - row[metric] for row in rows
            ]
            axis.errorbar(
                batches,
                values,
                yerr=[low, high],
                label=label,
                color=color,
                marker=marker,
                linestyle=style,
                linewidth=2.2,
                markersize=5,
                capsize=3,
            )
        axis.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Concurrent requests (log scale)", labelpad=10)
        axis.set_ylabel("Tokens / second")
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#cbd5e1")
        axis.tick_params(colors="#475569")
    fig.suptitle(
        "LFM2.5-24B-A2B-FP8 on NVIDIA L40S",
        x=0.07,
        y=0.97,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color="#0f172a",
    )
    fig.text(
        0.07,
        0.895,
        "Block-scaled FP8 weights  ·  CUDA  ·  2,048 prompt × 128 output tokens  ·  November 1, 2026",
        fontsize=11,
        color="#475569",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.15),
        ncol=3,
        frameon=False,
        columnspacing=2.5,
    )
    fig.text(
        0.07,
        0.045,
        "16-bit KV · Median and observed range of 3 repetitions after 1 warmup · client-stream-burst-v1\n"
        "Identical checkpoint and synthetic input IDs. Prefill rate is total prompt tokens divided by maximum TTFT; "
        "decode counts the remaining 127 tokens.",
        fontsize=9,
        color="#475569",
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.31, wspace=0.22)
    output = Path(output) if output is not None else ROOT / "build/throughput.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".svg", ".png"):
        target = output.with_suffix(extension)
        fig.savefig(
            target,
            dpi=160,
            facecolor="white",
            metadata={"Date": None} if extension == ".svg" else None,
        )
    plt.close(fig)
    return output.with_suffix(".svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=None)
    parser.add_argument("output", nargs="?", type=Path, default=None)
    parser.add_argument("--report", dest="report_opt", type=Path, default=None)
    parser.add_argument("--out", dest="output_opt", type=Path, default=None)
    args = parser.parse_args()
    out = plot(args.report_opt or args.report, args.output_opt or args.output)
    print(out)


if __name__ == "__main__":
    main()
