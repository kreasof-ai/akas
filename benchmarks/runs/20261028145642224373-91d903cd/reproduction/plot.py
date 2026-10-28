"""Render the Radeon LFM3 wall-decode showcase with the README plot styling."""

import argparse
import json
from pathlib import Path


def plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    data = json.loads(Path(report).read_text(encoding="utf-8"))
    # Outline the same DejaVu Sans fallback used by the other showcase plots.
    # GitHub then renders the intended font without relying on installed fonts.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "path",
            "svg.hashsalt": "akas-lfm3-vulkan-showcase",
        }
    )
    fig, axis = plt.subplots(figsize=(13, 6.2))
    axes = [axis]
    series = (
        ("akas", "AKAS · FP16 KV", "#2563eb", "o"),
        ("baseline", "llama.cpp · FP16 KV", "#ea580c", "s"),
    )
    phases = (("decode_tok_s", "decode_tokens_per_second", "Decode"),)
    for axis, (metric, sample_key, title) in zip(axes, phases):
        for kind, label, color, marker in series:
            profile = next(p for p in data["profiles"] if p["kind"] == kind)
            rows = sorted(profile["measurements"], key=lambda row: row["prefix"])
            values = [row[metric] for row in rows]
            low = [
                row[metric] - min(s[sample_key] for s in row["samples"]) for row in rows
            ]
            high = [
                max(s[sample_key] for s in row["samples"]) - row[metric] for row in rows
            ]
            axis.errorbar(
                [row["prefix"] for row in rows],
                values,
                yerr=[low, high],
                label=label,
                color=color,
                marker=marker,
                linewidth=2.2,
                markersize=5,
                capsize=3,
            )
        axis.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
        axis.set_xscale("log", base=2)
        axis.set_xticks(
            [32, 512, 2048, 8192, 32768, 130944],
            ["32", "512", "2k", "8k", "32k", "130,944"],
        )
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Prefix tokens (log scale)", labelpad=10)
        axis.set_ylabel("Tokens / second")
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#cbd5e1")
        axis.tick_params(colors="#475569")
    fig.suptitle(
        "LFM3-1.7B on AMD Radeon RX 6700 XT",
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
        "QAD Q4_0 weights  ·  Vulkan  ·  1 request × 128 output tokens  ·  October 28, 2026",
        fontsize=11,
        color="#475569",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.15),
        ncol=2,
        frameon=False,
        columnspacing=2.5,
    )
    fig.text(
        0.07,
        0.045,
        "FP16 KV · 131,072 context limit · Median and observed range of 5 repetitions after 2 warmups\n"
        "Completed wall decode including greedy sampling/control. Prefill, state restore and validation excluded.\n"
        "Identical checkpoint and synthetic input IDs; complete state is produced by actual model prefill.",
        fontsize=9,
        color="#475569",
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.31, wspace=0.22)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".svg", ".png"):
        fig.savefig(
            output.with_suffix(extension),
            dpi=160,
            facecolor="white",
            metadata={"Date": None} if extension == ".svg" else None,
        )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.report, args.output)
