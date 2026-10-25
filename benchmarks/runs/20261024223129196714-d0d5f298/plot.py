"""Render retained Vulkan report phase rates, including an earlier AKAS sweep."""

import argparse
import json
from pathlib import Path


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    data = json.loads(report.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
                         "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for axis, metric, label in zip(axes, ["prefill_tok_s", "decode_tok_s"], ["Prefill", "Decode"]):
        previous = data.get("retuning", {}).get("cases")
        if previous:
            rows = sorted(previous, key=lambda r:r["prefix"])
            key = "before_prefill" if metric == "prefill_tok_s" else "before_decode"
            axis.plot([r["prefix"] for r in rows], [r[key] for r in rows],
                      color="#94a3b8", linestyle="--", marker="^", label="AKAS before", linewidth=1.7)
        for profile, color, marker in zip(data["profiles"], ["#2563eb", "#ea580c"], ["o", "s"]):
            rows = sorted(profile["measurements"], key=lambda r:r["prefix"])
            name = "AKAS tuned · WebGPU/Vulkan" if profile["kind"] == "akas" else "llama.cpp · Vulkan"
            axis.plot([r["prefix"] for r in rows], [r[metric] for r in rows],
                      marker=marker, color=color, label=name, linewidth=2.3, markersize=5)
        axis.set_title(label, loc="left", fontsize=14, fontweight="bold")
        axis.set_xscale("log", base=2)
        axis.set_xticks([32, 512, 2048, 8192, 32768, 130944],
                       ["32", "512", "2k", "8k", "32k", "130,944"])
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Prefix tokens (log scale)", labelpad=10)
        axis.set_ylabel("Tokens / second")
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.grid(axis="y", color="#e2e8f0")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("#cbd5e1")
    fig.suptitle("LFM2.5-2.6B Q4_0 on AMD Radeon RX 6700 XT",
                 x=.07, y=.97, ha="left", fontsize=20, fontweight="bold", color="#0f172a")
    fig.text(.07, .88, "Vulkan  ·  FP16 KV  ·  1 request × 128 outputs  ·  Identical C4 input IDs and checkpoint",
             color="#475569")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.06, .115), ncol=3, frameon=False)
    fig.text(.07, .045, "One retained sample per prefix; AKAS phases preloaded; llama.cpp warmed per prefix.\n"
             "Phase boundaries differ across runtimes. These diagnostic rates do not establish end-to-end speedups.",
             fontsize=9, color="#475569", linespacing=1.5)
    fig.subplots_adjust(left=.07, right=.98, top=.79, bottom=.28, wspace=.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.report, args.output)
