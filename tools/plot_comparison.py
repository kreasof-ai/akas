"""Render README benchmark figures from retained campaign or speech reports.

Run: python -m tools.plot_comparison REPORT_JSON OUTPUT_SVG
"""

import argparse
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from statistics import median, quantiles


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ID = "20261101T074055Z-ad51ac551826"


def _load_report(report):
    if report is not None:
        path = Path(report)
        if path.is_dir():
            path = path / "report.json"
        return json.loads(path.read_text(encoding="utf-8"))
    from akas.benchmarking import archive

    raw = archive.read_file(
        ROOT / "benchmarks", "reports", DEFAULT_REPORT_ID, "report.json"
    )
    return json.loads(raw)


def _style_axis(axis):
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#cbd5e1")
    axis.tick_params(colors="#475569")


def _save(figure, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".svg", ".png"):
        figure.savefig(
            output.with_suffix(extension),
            dpi=160,
            facecolor="white",
            metadata={"Date": None} if extension == ".svg" else None,
        )
    return output.with_suffix(".svg")


def _plot_speech(data, plt, output):
    from matplotlib.ticker import StrMethodFormatter

    profiles = {profile["id"]: profile for profile in data["profiles"]}
    series = [
        ("akas", "AKAS · WebGPU/Vulkan", "#2563eb", "o", "-"),
        ("llama", "llama.cpp · Vulkan", "#ea580c", "s", "--"),
    ]
    has_cpu = "llama-cpu" in profiles
    if has_cpu:
        series.append(("llama-cpu", "llama.cpp · CPU", "#0d9488", "^", ":"))
    successful = {
        runtime: {
            row["case"]["id"]: row
            for row in profiles[runtime]["measurements"]
            if row["status"] == "ok"
        }
        for runtime, *_ in series
    }
    shared = set.intersection(*(set(rows) for rows in successful.values()))
    cases = successful["akas"]
    workload = data["workload"]
    transcription = workload.get("protocol") == "warm-asr-request-v1"
    tier_field = "duration_band" if transcription else "tier"
    tiers = ("short", "medium", "long")
    positions = range(len(tiers))
    grouped = [
        sorted(
            case_id for case_id in shared if cases[case_id]["case"][tier_field] == tier
        )
        for tier in tiers
    ]
    if any(not group for group in grouped):
        raise ValueError(
            "speech plot requires matched measurements in every length tier"
        )
    panels = [
        ("wall_seconds", "Request latency", "Wall milliseconds", "{x:,.0f}")
        if transcription
        else ("wall_seconds", "Request latency", "Wall seconds", "{x:,.1f}"),
        (
            "real_time_factor",
            "Real-time factor",
            "Wall seconds / input audio second"
            if transcription
            else "Wall seconds / audio second",
            "{x:.3f}" if transcription else "{x:.2f}",
        ),
    ]
    if has_cpu:
        panels.append(
            (
                "speedup",
                "Wall speedup vs CPU",
                "CPU wall time / GPU wall time",
                "{x:g}×",
            )
        )
    fig, axes = plt.subplots(
        1, len(panels), figsize=(16 if has_cpu else 13, 6.2), sharex=True
    )

    def point(runtime, key, case_id):
        row = successful[runtime][case_id]["summary"]
        if key == "speedup":
            return (
                successful["llama-cpu"][case_id]["summary"]["wall_seconds"]
                / row["wall_seconds"]
            )
        if transcription and key == "wall_seconds":
            return row[key] * 1000
        return row[key]

    for axis, (key, title, unit, tick_format) in zip(axes, panels, strict=True):
        for runtime, label, color, marker, style in series:
            if runtime == "llama-cpu" and key == "speedup":
                axis.axhline(1, color=color, linestyle=style, linewidth=1.5)
                continue
            values = [
                [point(runtime, key, case_id) for case_id in group] for group in grouped
            ]
            centers = [median(group) for group in values]
            quartiles = [quantiles(group, n=4, method="inclusive") for group in values]
            axis.errorbar(
                positions,
                centers,
                yerr=[
                    [center - q[0] for center, q in zip(centers, quartiles)],
                    [q[2] - center for center, q in zip(centers, quartiles)],
                ],
                label=label,
                color=color,
                marker=marker,
                linestyle=style,
                linewidth=2.2,
                markersize=5,
                capsize=3,
            )
        axis.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
        labels = [tier.title() for tier in tiers]
        if transcription:
            durations = [
                [cases[case_id]["summary"]["audio_seconds"] for case_id in group]
                for group in grouped
            ]
            labels = [
                f"{label}\n{min(values):.1f}–{max(values):.1f} s"
                for label, values in zip(labels, durations, strict=True)
            ]
        axis.set_xticks(positions, labels)
        axis.set_xlim(-0.2, len(tiers) - 0.8)
        axis.set_xlabel(
            "Input duration tier" if transcription else "Prompt length tier",
            labelpad=10,
        )
        axis.set_ylabel(unit)
        axis.yaxis.set_major_formatter(StrMethodFormatter(tick_format))
        axis.grid(axis="x", color="#e2e8f0", linewidth=0.8)
        _style_axis(axis)

    metadata = profiles["akas"]["metadata"]
    model = metadata["model"].rsplit("/", 1)[-1]
    device = metadata["device"]["name"]
    title = f"{model} on {device}"
    hardware = "Vulkan"
    if has_cpu:
        title = f"{model} speech synthesis"
        cpu_name = profiles["llama-cpu"]["metadata"]["device"]["name"]
        hardware = f"{device} (Vulkan)  ·  {cpu_name} (CPU)"
    languages = len(
        {
            item.get(
                "language" if transcription else "lang", workload["request"].get("lang")
            )
            for item in workload["utterances"]
        }
    )
    started = datetime.fromisoformat(data["campaign"]["created_at"])
    language_label = "language" if languages == 1 else "languages"
    date = f"{started:%B} {started.day}, {started.year}"
    fig.suptitle(
        title,
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
        f"{metadata['weight_format'].upper()} weights  ·  {hardware}  ·  {languages} {language_label}  ·  {date}",
        fontsize=11,
        color="#475569",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.15),
        ncol=len(series),
        frameon=False,
        columnspacing=2.5,
    )
    direction = (
        "Lower is better for latency and RTF. Higher is better for speedup."
        if has_cpu
        else "Lower is better on both panels."
    )
    units = "clips" if transcription else "prompts"
    unit = "clip" if transcription else "prompt"
    completion = "complete transcript" if transcription else "complete PCM16"
    fig.text(
        0.07,
        0.045,
        f"{direction} Points are tier medians with interquartile ranges across {len(shared)} matched {units}.\n"
        f"Each {unit} contributes the median of {metadata['repeats']} measured requests after {metadata['warmup']} warmup. "
        f"Timings end at receipt of {completion}.",
        fontsize=9,
        color="#475569",
        linespacing=1.5,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.78,
        bottom=0.35 if transcription else 0.31,
        wspace=0.22,
    )
    try:
        svg = _save(fig, output)
        if transcription:
            svg.write_bytes(
                b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines())
                + b"\n"
            )
        return svg
    finally:
        plt.close(fig)


def _native_stream_points(data, baseline):
    """Match completed single-burst results from the same speech workload."""
    meta = data["metadata"]
    if (
        data.get("schema") != "akas.benchmark.v1"
        or meta.get("protocol") != "client-speech-pcm-stream-v1"
        or meta.get("runtime") != "akas"
        or meta.get("weight_format") != "bf16"
        or baseline.get("schema") != "akas.native-stream-baseline-report.v1"
        or meta["speech_workload"]["definition"] != baseline["workload"]
    ):
        raise ValueError("incompatible native speech streaming reports")
    prompt = baseline["workload"]["utterances"][0]["text"]

    def by_concurrency(rows):
        matched = {}
        for row in rows:
            count = row["case"]["concurrency"]
            if (
                row["case"]["text"] != prompt
                or count < 1
                or count in matched
                or count != row["requests"]
                or count != row["completed"]
                or row["failed"]
                or any(
                    not isfinite(row[key]) or row[key] <= 0
                    for key in ("audio_seconds_per_second", "wall_s")
                )
            ):
                raise ValueError(
                    "native speech plot requires one complete burst per concurrency"
                )
            matched[count] = row
        return matched

    akas = by_concurrency(data["results"])
    native = by_concurrency(baseline["measurements"])
    if not akas or akas.keys() != native.keys():
        raise ValueError("native speech reports must cover the same concurrencies")
    return [(count, akas[count], native[count]) for count in sorted(akas)]


def _plot_native_stream(data, baseline, plt, output):
    from matplotlib.ticker import StrMethodFormatter

    points = _native_stream_points(data, baseline)
    positions = range(len(points))
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharex=True)
    panels = (
        ("Burst completion time", "Wall seconds", "{x:g}"),
        ("Audio throughput", "Audio seconds / wall second", "{x:g}"),
        ("Throughput speedup vs Zyphra", "AKAS / Zyphra native", "{x:g}×"),
    )
    for axis, (title, unit, ticks) in zip(axes, panels, strict=True):
        axis.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
        axis.set_xticks(positions, [str(count) for count, *_ in points])
        axis.set_xlim(-0.2, len(points) - 0.8)
        axis.set_xlabel("Concurrent requests", labelpad=10)
        axis.set_ylabel(unit)
        axis.yaxis.set_major_formatter(StrMethodFormatter(ticks))
        axis.grid(axis="x", color="#e2e8f0", linewidth=0.8)
        _style_axis(axis)
    axes[0].set_ylim(
        0,
        1.15 * max(row["wall_s"] for _, a, n in points for row in (a, n)),
    )
    axes[1].set_ylim(
        0,
        1.15
        * max(row["audio_seconds_per_second"] for _, a, n in points for row in (a, n)),
    )
    for axis, metric in zip(
        axes[:2], ("wall_s", "audio_seconds_per_second"), strict=True
    ):
        for col, label, color, marker, style in (
            (1, "AKAS · CUDA", "#2563eb", "o", "-"),
            (2, "Zyphra native · CUDA", "#ea580c", "s", "--"),
        ):
            axis.plot(
                positions,
                [point[col][metric] for point in points],
                label=label,
                color=color,
                marker=marker,
                linestyle=style,
                linewidth=2.2,
                markersize=5,
            )
    rates = [
        a["audio_seconds_per_second"] / n["audio_seconds_per_second"]
        for _, a, n in points
    ]
    axes[2].axhline(1, color="#ea580c", linestyle="--", linewidth=1.5)
    axes[2].plot(
        positions, rates, color="#2563eb", marker="o", linewidth=2.2, markersize=5
    )
    axes[2].set_ylim(0, 1.2 * max(1, *rates))
    for x, ratio in zip(positions, rates, strict=True):
        offset, align = (-6, "right") if x == len(points) - 1 else (0, "center")
        axes[2].annotate(
            f"{ratio:.2f}×",
            (x, ratio),
            xytext=(offset, 8),
            textcoords="offset points",
            ha=align,
            color="#2563eb",
            fontsize=9,
        )
    meta = data["metadata"]
    started = datetime.fromtimestamp(meta["started_unix_s"], tz=timezone.utc)
    model = meta["model"].rsplit("/", 1)[-1]
    device = meta["device"]["name"]
    fig.suptitle(
        f"{model} streaming speech",
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
        f"BF16 LM · FP32 keep2_plain codec · {device} (CUDA) · {started:%B} {started.day}, {started.year}",
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
        "Lower is better for wall time. Higher is better for throughput and speedup. "
        "One HTTP PCM burst per runtime and concurrency.\n"
        "Speedup uses audio throughput at matched concurrency. "
        "Zyphra's C256 run admitted requests in waves.",
        fontsize=9,
        color="#475569",
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.31, wspace=0.25)
    try:
        return _save(fig, output)
    finally:
        plt.close(fig)


def plot(report=None, output=None, *, native_baseline=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    data = _load_report(report)
    speech = data.get("schema") == "akas.speech-report.v1"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "path",
            "svg.hashsalt": "akas-speech-stream-showcase"
            if native_baseline is not None
            else "akas-speech-showcase"
            if speech
            else "akas-lfm25-fp8-showcase",
        }
    )
    output = output or ROOT / "build/throughput.svg"
    if native_baseline is not None:
        return _plot_native_stream(data, _load_report(native_baseline), plt, output)
    if speech:
        return _plot_speech(data, plt, output)
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
        axis.set_xlabel("Concurrent requests (log scale)", labelpad=10)
        axis.set_ylabel("Tokens / second")
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        _style_axis(axis)
    is_h200 = "NVIDIA H200" in json.dumps(data.get("evidence", {}))
    gpu_label = "NVIDIA H200" if is_h200 else "NVIDIA L40S"
    date_label = "November 3, 2026" if is_h200 else "November 1, 2026"
    fig.suptitle(
        f"LFM2.5-24B-A2B-FP8 on {gpu_label}",
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
        f"Block-scaled FP8 weights  ·  CUDA  ·  2,048 prompt × 128 output tokens  ·  {date_label}",
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
    try:
        return _save(fig, output)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=None)
    parser.add_argument("output", nargs="?", type=Path, default=None)
    parser.add_argument("--report", dest="report_opt", type=Path, default=None)
    parser.add_argument("--out", dest="output_opt", type=Path, default=None)
    parser.add_argument(
        "--native-baseline", type=Path, help="paired Zyphra speech report"
    )
    args = parser.parse_args()
    out = plot(
        args.report_opt or args.report,
        args.output_opt or args.output,
        native_baseline=args.native_baseline,
    )
    print(out)


if __name__ == "__main__":
    main()
