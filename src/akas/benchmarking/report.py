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
        ratios = {
            name: row["summary"][name]["mean"] / old["summary"][name]["mean"]
            for name in ("prefill_tokens_per_second", "decode_tokens_per_second")
        }
        comparisons.append(dict(case=row["case"], ratios=ratios))
    return comparisons
