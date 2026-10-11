"""Validate and summarize every measured native AR/MTP configuration."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean

from benchmarks.qwen35.dense_ablation import configurations
from benchmarks.qwen35.dense_scaling import CONCURRENCIES, REVISION


def read_json(path):
    return json.loads(path.read_text())


def records(path):
    text = (
        path.read_text()
        if path.exists()
        else gzip.decompress(Path(str(path) + ".gz").read_bytes()).decode()
    )
    return [json.loads(line) for line in text.splitlines()]


def summarize(root, baselines):
    matrix = configurations()
    invariance = read_json(root / "batch-invariance.json")
    if (
        invariance["status"] != "passed"
        or {p["capacity"] for p in invariance["results"]} != set(CONCURRENCIES)
        or any(
            not p["bitwise_states"] or not p["predictions"]
            for p in invariance["results"]
        )
    ):
        raise ValueError("batch-invariance check incomplete or failed")
    profiles = {}
    for name, config in matrix.items():
        quality = read_json(root / f"{name}-qualification.json")
        if (
            quality["status"] != "passed"
            or quality["configuration"] != config
            or {p["concurrency"] for p in quality["points"]} != set(CONCURRENCIES)
            or any(
                not p["greedy_identical"] or not p["bitwise_terminal_states"]
                for p in quality["points"]
            )
        ):
            raise ValueError(f"incomplete qualification: {name}")
        diagnostic = read_json(root / f"{name}-gpu-profile.json")
        if len(diagnostic["points"]) != 9:
            raise ValueError(f"incomplete profile: {name}")
        points = []
        for count in CONCURRENCIES:
            path = root / "results" / name / f"c{count}" / "report.json"
            report = read_json(path)
            if report["status"] != "completed" or len(report["servers"]) != 1:
                raise ValueError(f"incomplete replay: {path}")
            server = report["servers"][0]
            settings = server["configuration"]
            if (
                settings["model_revision"] != REVISION
                or settings["weight_format"] != "BF16"
                or settings["kv_dtype"] != "bfloat16"
                or settings["state_dtype"] != "float32"
                or settings["prefix_cache"]
                or settings["cpu_offload"] != "none"
                or settings["speculative"] != bool(config["window"])
            ):
                raise ValueError(f"unexpected configuration: {path}")
            repeats = server["points"]
            if len(repeats) != 2 or {r["repeat"] for r in repeats} != {0, 1}:
                raise ValueError(f"invalid repeats: {path}")
            for sample in repeats:
                summary = sample["summary"]
                expected = max(16, 4 * count)
                if (
                    sample["concurrency"] != count
                    or sample["disposition"] != "measured"
                    or summary["failed"]
                    or summary["completed"] != expected
                    or summary["output_tokens"] != expected * 256
                ):
                    raise ValueError(f"invalid replay counts: {path}")
                raw = records(path.parent / settings["name"] / sample["requests_file"])
                if len(raw) != expected or any(
                    r["status"] != "completed"
                    or r["output_tokens"] != 256
                    or sum(e["delta_tokens"] for e in r["token_events"]) != 256
                    for r in raw
                ):
                    raise ValueError(f"stream counts disagree: {path}")
                rate = sum(r["output_tokens"] for r in raw) / summary["elapsed_seconds"]
                if abs(rate - summary["output_tokens_per_second"]) > 1e-6:
                    raise ValueError(f"throughput denominator disagrees: {path}")
            counters = read_json(root / f"{name}-c{count}-counters.json")
            before, after = counters["before"], counters["after"]
            delta = {
                key: after["speculation_stats"][key] - value
                for key, value in (before["speculation_stats"] or {}).items()
            }
            summaries = [s["summary"] for s in repeats]
            point = dict(
                concurrency=count,
                workload_sha256=report["workload_sha256"],
                repeat_tok_s=[s["output_tokens_per_second"] for s in summaries],
                tok_s=mean(s["output_tokens_per_second"] for s in summaries),
                mean_ttft_ms=1000 * mean(s["ttft_seconds"]["mean"] for s in summaries),
                p95_ttft_ms=1000 * mean(s["ttft_seconds"]["p95"] for s in summaries),
                speculation=delta,
                acceptance=(
                    delta["accepted"] / delta["proposed"]
                    if delta.get("proposed")
                    else None
                ),
                emitted_per_request_round=(
                    delta["emitted"] / delta["request_rounds"]
                    if delta.get("request_rounds")
                    else None
                ),
                gpu=next(p for p in diagnostic["points"] if p["concurrency"] == count),
            )
            points.append(point)
        profiles[name] = dict(
            configuration=config, points=points, qualification=quality
        )
    rows = []
    for count in CONCURRENCIES:
        token_hashes = {
            next(
                p for p in entry["qualification"]["points"] if p["concurrency"] == count
            )["token_sha256"]
            for entry in profiles.values()
        }
        if len(token_hashes) != 1:
            raise ValueError(
                f"C{count}: greedy qualification outputs differ across profiles"
            )
        native = {
            name: next(p for p in entry["points"] if p["concurrency"] == count)
            for name, entry in profiles.items()
        }
        baseline = {}
        hashes = {p["workload_sha256"] for p in native.values()}
        for name in ("vllm-ar-stream4", "sglang-ar-stream4-mem50"):
            report = read_json(
                baselines / "results" / name / f"c{count}" / "report.json"
            )
            hashes.add(report["workload_sha256"])
            baseline[name] = mean(
                p["summary"]["output_tokens_per_second"]
                for p in report["servers"][0]["points"]
            )
        if len(hashes) != 1:
            raise ValueError(f"C{count}: different workload hashes")
        winner = max(native, key=lambda name: native[name]["tok_s"])
        rows.append(
            dict(
                concurrency=count,
                winner=winner,
                tok_s=native[winner]["tok_s"],
                best_baseline_tok_s=max(baseline.values()),
                baselines=baseline,
                speedup_vs_best_baseline=native[winner]["tok_s"]
                / max(baseline.values()),
                workload_sha256=hashes.pop(),
            )
        )
    return dict(
        model_revision=REVISION,
        profiles=profiles,
        best_by_concurrency=rows,
        measured_points=len(matrix) * 9 * 2,
        measured_output_tokens=len(matrix)
        * 2
        * sum(max(16, 4 * c) * 256 for c in CONCURRENCIES),
        independent_model_quality_qualified=False,
    )


def render(summary, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(
        ar="#444444",
        **{
            "mtp-w2-host": "#a6cee3",
            "mtp-w2-graph": "#1f78b4",
            "mtp-w2-deferred": "#005080",
            "mtp-w4-host": "#b2df8a",
            "mtp-w4-graph": "#33a02c",
            "mtp-w4-deferred": "#006620",
        },
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for column, fused in enumerate((True, False)):
        suffix = "fused" if fused else "unfused"
        for name, entry in summary["profiles"].items():
            if entry["configuration"]["fused_head"] != fused:
                continue
            label = name.removesuffix("-" + suffix)
            points = entry["points"]
            x = [p["concurrency"] for p in points]
            axes[0, column].plot(
                x,
                [p["tok_s"] for p in points],
                marker="o",
                markersize=3,
                color=colors[label],
                label=label,
            )
            ar = summary["profiles"][f"ar-{suffix}"]["points"]
            axes[1, column].plot(
                x,
                [p["tok_s"] / a["tok_s"] for p, a in zip(points, ar)],
                marker="o",
                markersize=3,
                color=colors[label],
                label=label,
            )
        axes[0, column].set_title(
            "Fused vocabulary head" if fused else "Unfused vocabulary head"
        )
        axes[0, column].legend(frameon=False, fontsize=8)
        axes[0, column].set_ylim(bottom=0)
        axes[1, column].axhline(1, color="#999999", linestyle="--", linewidth=1)
    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xticks(CONCURRENCIES, [str(c) for c in CONCURRENCIES])
        axis.set_xlabel("Concurrency")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[0]:
        axis.set_ylabel("Output tok/s, whole HTTP replay")
    for axis in axes[1]:
        axis.set_ylabel("Throughput / matched AR")
    fig.suptitle("Qwen3.5-0.8B · L40S · matched native AR/MTP ablation")
    fig.supxlabel(
        "66–81 input / 256 output · BF16 weights/KV · FP32 state · mean of two full replays",
        fontsize=9,
    )
    fig.savefig(destination)
    plt.close(fig)


def markdown(summary, path):
    text = [
        "# Native Qwen3.5-0.8B AR/MTP ablation",
        "",
        "One NVIDIA L40S; pinned official model revision `" + REVISION + "`. "
        "Every cell is the arithmetic mean of two complete HTTP replays using identical "
        "66–81-token prompts and 256 forced greedy outputs. These are short-context tests, "
        "not the separate 32K/16K workload.",
        "",
        "All 14 supported configurations use the same freshly compiled projection, "
        "attention and recurrent kernels. We vary fused versus unfused vocabulary head, "
        "AR versus official MTP, verifier window 2 (one draft) versus 4 (three drafts), "
        "host-controlled versus complete CUDA graph, and immediate versus deferred "
        "recurrent commit. Deferred commit requires the complete graph. Weights and KV "
        "are BF16; recurrent state is FP32. Prefix reuse and offload are disabled.",
        "",
        "Every configuration passed matched greedy output and bitwise terminal "
        "recurrent/KV-state checks at C1–C256 with unequal 18–24-token output limits. "
        "AR also checks fused/unfused head equivalence. These internal checks do not "
        "resolve the outstanding independent reference-model qualification.",
        "",
        "![Matched scaling](qwen35-08b-native-ablation.svg)",
        "",
    ]
    for suffix in ("fused", "unfused"):
        names = [f"ar-{suffix}"] + [
            f"mtp-w{w}-{m}-{suffix}"
            for w in (2, 4)
            for m in ("host", "graph", "deferred")
        ]
        labels = [n.removesuffix("-" + suffix) for n in names]
        text += [
            f"## {suffix.capitalize()} vocabulary head: output tok/s",
            "",
            "| C | " + " | ".join(labels) + " |",
            "|---:" + "|---:" * len(names) + "|",
        ]
        for index, count in enumerate(CONCURRENCIES):
            text.append(
                f"| {count} | "
                + " | ".join(
                    f"{summary['profiles'][n]['points'][index]['tok_s']:,.0f}"
                    for n in names
                )
                + " |"
            )
        text.append("")
    text += [
        "## Fastest measured configuration at each concurrency",
        "",
        "Choosing a winner per concurrency is a description of this sweep, not a "
        "validated automatic serving policy. Configuration choices still require held-out validation.",
        "",
        "| C | Winner | Output tok/s | Mean TTFT ms | Versus best retained baseline AR |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in summary["best_by_concurrency"]:
        point = next(
            p
            for p in summary["profiles"][row["winner"]]["points"]
            if p["concurrency"] == row["concurrency"]
        )
        text.append(
            f"| {row['concurrency']} | {row['winner']} | {row['tok_s']:,.0f} | "
            f"{point['mean_ttft_ms']:.1f} | {row['speedup_vs_best_baseline']:.3f}× |"
        )
    text += [
        "",
        "The baseline denominator is the faster retained AR measurement of "
        "vLLM 0.17.1 and SGLang 0.5.9, with the same workload hashes and streaming "
        "interval 4. Baselines were not rerun during this native ablation; their MTP "
        "profiles were slower, as documented in [the baseline comparison](qwen35-08b-local-scaling.md).",
        "",
        "## CUDA diagnostics",
        "",
        "Diagnostics run after the serving replay, in a separate process. For W2 "
        "there is no separate draft stage: repair also produces the next proposal. The table "
        "below uses the middle of five fixed-input diagnostic graph samples with "
        "CUDA event nodes around every kernel. It sums measured kernel times, which "
        "include instrumentation effects and exclude serving, host/copy gaps, prefill "
        "and fill/drain. In host mode the diagnostic combines the same GPU calls in "
        "a graph; real host-controlled decode-round wall times are retained separately "
        "in JSON. Diagnostic timings must not be interpreted as serving token rates.",
        "",
        "| Configuration | C | Kernel sum ms | Recurrent scan/update ms | Draft ms | Verify/target ms | Commit ms | Repair ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, entry in summary["profiles"].items():
        for count in (1, 256):
            p = next(p for p in entry["points"] if p["concurrency"] == count)
            gpu = p["gpu"]
            recurrent = sum(
                ms
                for kind, ms in gpu["groups"].items()
                if kind in ("gdn_scan", "gdn_deferred", "gdn_materialize")
            )
            stages = gpu["stages"]
            text.append(
                f"| {name} | {count} | {gpu['total_ms']:.3f} | {recurrent:.3f} | "
                f"{stages.get('draft',0):.3f} | {stages.get('verify',stages.get('target',0)):.3f} | "
                f"{stages.get('commit',0):.3f} | {stages.get('repair',0):.3f} |"
            )
    text += [
        "",
        "## Draft acceptance and TTFT",
        "",
        "Acceptance counters cover the two measured replays plus their four serial "
        "warmup requests at each C. The untimed concurrency-wide prewarm is excluded. "
        "Emission counts are per request-round, not per scheduler iteration.",
        "",
        "| Configuration | C | Accepted draft fraction | Emitted tokens/request-round | Mean TTFT ms | Mean repeat p95 TTFT ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, entry in summary["profiles"].items():
        for count in (1, 8, 32, 128, 256):
            p = next(p for p in entry["points"] if p["concurrency"] == count)
            accept = f"{p['acceptance']:.1%}" if p["acceptance"] is not None else "—"
            emit = (
                f"{p['emitted_per_request_round']:.3f}"
                if p["emitted_per_request_round"]
                else "—"
            )
            text.append(
                f"| {name} | {count} | {accept} | {emit} | {p['mean_ttft_ms']:.1f} | {p['p95_ttft_ms']:.1f} |"
            )
    text += [
        "",
        "## Evidence and reproduction",
        "",
        f"{summary['measured_points']} measured replay points, "
        f"{summary['measured_output_tokens']:,} output tokens, zero measured request failures. "
        "Every point has an untimed wave of C requests generating 64 outputs before "
        "the sweep, then two serial warmup requests before each measured repeat. "
        "Requests per repeat are `max(16, 4*C)`, with closed-loop refill and forced "
        "length. TTFT is client-observed; all throughput includes prefill and fill/drain.",
        "",
        "Evidence is in `data/qwen35-08b-native-ablation/`: all raw request streams, "
        "workloads, telemetry, reports, launch arguments, source fingerprints, "
        "per-point scheduler/MTP counters, qualification checks and kernel profiles. "
        "Large JSONL/JSON streams and logs are losslessly gzipped. The source identity "
        "matches the serving bundle; kernel artifacts themselves remain local.",
        "",
        "```bash",
        "PYTHONPATH=.:src:src build/tensor-dev/bin/python -m akas compile --definition akas.models.qwen35.model:DenseEngine --profile batched-bank \\",
        "  --checkpoint /tmp/tensor-qwen08/models/qwen08 --out /tmp/tensor-qwen08/native-ablation-bundle \\",
        "  --config '{\"verify_chunks\":[2,4],\"mtp\":true}'",
        "PYTHONPATH=.:src:src build/tensor-dev/bin/python -m benchmarks.qwen35.dense_ablation sweep \\",
        "  --checkpoint /tmp/tensor-qwen08/models/qwen08 --bundle /tmp/tensor-qwen08/native-ablation-bundle \\",
        "  --references docs/research/data/qwen35-08b-local-scaling/sglang-quality.json \\",
        "  --workloads /tmp/tensor-qwen08/workloads/train --out /tmp/tensor-qwen08/native-ablation",
        "```",
        "",
        "Restore the pinned snapshot and decompress the previously retained train "
        "workloads before running. Use a fresh output directory. Full results and "
        "replay-level samples are in `summary.json`; all nine concurrency diagnostics "
        "are retained even though the compact report shows only selected rows.",
        "",
    ]
    path.write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("docs/research/data/qwen35-08b-native-ablation"),
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("docs/research/data/qwen35-08b-local-scaling"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/research/qwen35-08b-native-ablation.md"),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("docs/research/qwen35-08b-native-ablation.svg"),
    )
    args = parser.parse_args()
    summary = summarize(args.data, args.baselines)
    (args.data / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render(summary, args.plot)
    markdown(summary, args.report)
    print(json.dumps(summary["best_by_concurrency"], indent=2))


if __name__ == "__main__":
    main()
