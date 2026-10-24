"""Render the README benchmark figure from a retained comparison report.

Run: uv run python tools/plot_comparison.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "benchmarks/reports/20261023T155400Z-cdcde64a14b5"


def read_phases(path):
    """Read measured phase rows, excluding the unmeasured burst tables."""
    phases = {}
    mode = phase = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            mode = line.rsplit(" / ", 1)[-1]
        elif line.startswith("### "):
            phase = line.split()[1].lower()
        elif line.startswith("| 1 |") and phase in ("prefill", "decode"):
            cells = [cell.strip().replace(",", "") for cell in line.split("|")[1:-1]]
            phases.setdefault((mode, phase), []).append(
                (int(cells[1]), *(float(cell) for cell in cells[2:]))
            )
    return phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    phases = read_phases(args.report / "COMPARISON.md")
    # Outline SVG text so the chosen font renders consistently on GitHub.
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 11,
        "svg.fonttype": "path",
    })
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
    fig.patch.set_facecolor("#ffffff")
    series = (
        ("AKAS · FP16 KV", "#2563eb", "o", "-"),
        ("AKAS · FP8 KV", "#0d9488", "s", "--"),
        ("llama.cpp · FP16 KV", "#d97706", "^", ":"),
    )
    for row, (mode, title) in enumerate((("ar", "Autoregressive"), ("dspark", "Dspark speculative"))):
        for col, phase in enumerate(("prefill", "decode")):
            ax = axes[row, col]
            values = phases[mode, phase]
            for index, (label, color, marker, style) in enumerate(series, start=1):
                ax.plot([v[0] for v in values], [v[index] for v in values],
                        label=label, color=color, marker=marker, linestyle=style,
                        linewidth=2.2, markersize=5)
            ax.set_title(f"{title} / {phase}", loc="left", fontsize=13, fontweight="bold", pad=12)
            ax.set_xscale("log", base=2)
            ax.set_xticks([32, 512, 2048, 16384, 130944], ["32", "512", "2,048", "16,384", "130,944"])
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Tokens / second")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#cbd5e1")
            ax.tick_params(colors="#475569")
            if row == 1:
                ax.set_xlabel("Prefix tokens (log scale)", labelpad=9)
    fig.suptitle("LFM2.5-24B-A2B on NVIDIA L4", x=0.065, y=0.975,
                 ha="left", fontsize=23, fontweight="bold", color="#0f172a")
    fig.text(0.065, 0.927,
             "Q4_0 weights  ·  CUDA  ·  1 request × 128 output tokens  ·  October 23, 2026",
             fontsize=12, color="#475569")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.058, 0.906),
               ncol=3, frameon=False, columnspacing=2.5)
    fig.subplots_adjust(left=0.065, right=0.98, top=0.825, bottom=0.20, hspace=0.30, wspace=0.20)
    fig.text(0.065, 0.035,
             "Reported phase rates · 1 retained sample per shape · Panel y-scales differ\n"
             "AKAS: native-token-step-v2; llama.cpp: llama-server-eval-real-text-v1.\n"
             "Timing boundaries differ; these are not end-to-end speedup measurements.",
             fontsize=10, color="#475569", linespacing=1.5)
    for extension in ("svg", "png"):
        output = args.report / f"throughput.{extension}"
        fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
