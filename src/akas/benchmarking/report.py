"""Machine-readable reports, compact console tables, and matched-case comparisons."""

import json
from pathlib import Path

SCHEMA = "akas.benchmark.v1"


def write_report(path, metadata, results):
    report = dict(schema=SCHEMA, metadata=metadata, results=results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return report


def table(results):
    lines = ["batch  prompt  depth     pp tok/s     tg tok/s   tg/request   TTFT ms"]
    for result in results:
        c = result["case"]
        if result["status"] != "ok":
            lines.append(
                f"{c['batch']:5} {c['prompt']:7} {c['depth']:6}  {result['status']}: {result.get('error', '')}"
            )
            continue
        s = result["summary"]
        mean = lambda k: s[k]["mean"]
        lines.append(
            f"{c['batch']:5} {c['prompt']:7} {c['depth']:6} {mean('prefill_tokens_per_second'):12.1f} {mean('decode_tokens_per_second'):12.1f} {mean('decode_tokens_per_second_per_request'):12.1f} {mean('ttft_seconds') * 1000:9.1f}"
        )
    return "\n".join(lines)


def compare(current, baseline):
    if current.get("schema") != SCHEMA or baseline.get("schema") != SCHEMA:
        raise ValueError("unsupported benchmark report schema")
    # A changed model, device, precision, or measurement protocol is not a regression.
    keys = (
        "checkpoint_sha256",
        "definition",
        "provider",
        "device",
        "kv_dtype",
        "protocol",
        "graphs",
        "target",
        "seed",
        "mode",
        "weight_dtype",
        "activation_dtype",
        "recurrent_dtype",
    )
    if any(current["metadata"].get(k) != baseline["metadata"].get(k) for k in keys):
        raise ValueError(
            "baseline differs in model/device/precision/measurement protocol"
        )
    key = lambda row: json.dumps(row["case"], sort_keys=True)
    previous = {key(row): row for row in baseline["results"] if row["status"] == "ok"}
    comparisons = []
    for row in current["results"]:
        old = previous.get(key(row))
        if old is None or row["status"] != "ok":
            continue
        if row.get("seed") != old.get("seed") or any(
            name in row and name in old and row[name] != old[name]
            for name in ("prompt_sha256", "cached_prefix_sha256")
        ):
            raise ValueError("baseline uses different prompt inputs")
        ratios = {
            name: row["summary"][name]["mean"] / old["summary"][name]["mean"]
            for name in ("prefill_tokens_per_second", "decode_tokens_per_second")
        }
        comparisons.append(dict(case=row["case"], ratios=ratios))
    return comparisons


def scaling_regressions(results, maximum):
    """Find aggregate throughput drops between batches of the same workload."""
    if not 0 <= maximum < 1:
        raise ValueError("scaling regression limit must be in [0, 1)")
    groups = {}
    for row in results:
        if row["status"] == "ok":
            workload = {k: v for k, v in row["case"].items() if k != "batch"}
            groups.setdefault(json.dumps(workload, sort_keys=True), []).append(row)
    failures = []
    for rows in groups.values():
        rows.sort(key=lambda row: row["case"]["batch"])
        for before, after in zip(rows, rows[1:]):
            for metric in ("prefill_tokens_per_second", "decode_tokens_per_second"):
                ratio = (
                    after["summary"][metric]["mean"] / before["summary"][metric]["mean"]
                )
                if ratio < 1 - maximum:
                    failures.append(
                        dict(
                            case=after["case"],
                            previous_batch=before["case"]["batch"],
                            metric=metric,
                            ratio=ratio,
                        )
                    )
    return failures
