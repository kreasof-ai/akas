"""Export the matched Vulkan wall-decode sweep as a canonical archive report."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from akas.benchmarking.history import digest, utc, write
from tools.campaign_report import (
    SCHEMA,
    measurement,
    metadata_for,
    record,
    render,
    report_directory,
)


def export(history, runs, assets_run):
    records = [record(history, identity) for identity in runs]
    if not records or any(r["status"] != "passed" for r in records):
        raise ValueError("showcase requires sealed passed runs")
    protocol = "real-prefill-state-restored-wall-decode-v1"
    identities = {
        (
            r["metadata"]["protocol"],
            r["metadata"]["checkpoint_sha256"],
            r["metadata"]["max_context"],
            r["metadata"]["warmup"],
            r["metadata"]["repeats"],
        )
        for r in records
    }
    if len(identities) != 1 or next(iter(identities))[0] != protocol:
        raise ValueError("showcase timing/model/context/repetition mismatch")
    rows = {"akas": {}, "llama": {}}
    for run in records:
        for row in run["results"]:
            prefix = row["case"]["prompt"]
            runtime = run["metadata"]["runtime"]
            if prefix in rows[runtime] or row["case"]["generated"] != 128:
                raise ValueError("duplicate prefix or output length mismatch")
            rows[runtime][prefix] = (run, row)
    if not rows["akas"] or rows["akas"].keys() != rows["llama"].keys():
        raise ValueError("showcase requires matched prefix sets")
    prefixes = sorted(rows["akas"])
    workload = dict(
        id="lfm3-rx6700xt-wall-decode-v1",
        batches=[1],
        prefix_tokens=prefixes,
        output_tokens=128,
    )
    model = "lfm3-1.7b"
    aid = model + "-q4_0-webgpu-float16-ar"
    bid = "baseline-" + model + "-llama-vulkan-q4_0-ar"
    specification = dict(
        schema="akas.matrix-campaign.v1",
        models=[model],
        formats=["q4_0"],
        backends=["webgpu"],
        caches=["float16"],
        workload=workload,
        baselines=[
            dict(runtime="llama", provider="vulkan", weight_format="q4_0", mode="ar")
        ],
    )
    representative = rows["akas"][max(prefixes)][0]
    elapsed = lambda r: (
        datetime.fromisoformat(r["completed_at"])
        - datetime.fromisoformat(r["started_at"])
    ).total_seconds()
    campaign = dict(
        schema="akas.matrix-campaign.v1",
        identity=hashlib.sha256(
            json.dumps(
                dict(spec=specification, runs=runs, assets_run=assets_run),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        specification=specification,
        models={model: dict(id=model, model="LiquidAI/LFM3-1.7B", modes=["ar"])},
        created_at=min(r["started_at"] for r in records),
        updated_at=max(r["completed_at"] for r in records),
        status="completed",
        source=representative["source"],
        entries=[dict(id=aid, status="passed", context=dict(workload=workload))],
        spent_seconds=sum(elapsed(r) for r in records),
    )
    profiles = []
    for runtime, pid, kind in (("akas", aid, "akas"), ("llama", bid, "baseline")):
        selected = rows[runtime][max(prefixes)][0]
        measurements = []
        for prefix in prefixes:
            run, row = rows[runtime][prefix]
            item = measurement(row, kind, True, run["id"], protocol)
            item["phase_protocol"] = protocol
            item["prefill_observation"] = row["prefill_observation"]
            item["imported_prefill_state"] = row.get("imported_prefill_state")
            measurements.append(item)
        profiles.append(
            dict(
                id=pid,
                kind=kind,
                runtime=runtime,
                model_id=model,
                model="LiquidAI/LFM3-1.7B",
                provider="webgpu" if runtime == "akas" else "vulkan",
                weight_format="q4_0",
                cache_request="float16",
                kv_dtype="float16",
                mode="ar",
                status="passed",
                error=None,
                selected_run_id=selected["id"],
                qualifying_status="passed",
                spent_seconds=sum(
                    elapsed(r) for r in records if r["metadata"]["runtime"] == runtime
                ),
                recipe=dict(chunk=512, prefill_requests=1, graphs=True, trials=1),
                measurements=measurements,
            )
        )
    comparisons, acceptance_rows = [], []
    for prefix in prefixes:
        _, a = rows["akas"][prefix]
        _, b = rows["llama"][prefix]
        if a["prompt_sha256"] != b["prompt_sha256"]:
            raise ValueError("showcase prompt hash mismatch")
        x, y = a["output_ids"], b["output_ids"]
        comparisons.append(
            dict(
                prefix=prefix,
                output=128,
                exact=x == y,
                common_prefix_tokens=next(
                    (i for i, (v, w) in enumerate(zip(x, y)) if v != w), 128
                ),
                native_output_ids=x,
                baseline_output_ids=y,
            )
        )
        n, l = (row["summary"]["decode_tokens_per_second"]["median"] for row in (a, b))
        acceptance_rows.append(
            dict(
                prefix_tokens=prefix,
                decode=dict(akas=n, llama=l, ratio=n / l, parity_or_better=n >= l),
            )
        )
    evidence = {r["id"]: metadata_for(r) for r in records}
    evidence[assets_run] = metadata_for(record(history, assets_run))
    destination = report_directory(campaign, history)
    write(destination / "campaign.json", campaign)
    data = dict(
        schema=SCHEMA,
        generated_at=utc(),
        campaign=campaign,
        campaign_sha256=digest(destination / "campaign.json"),
        workload=workload,
        profiles=profiles,
        attempts=[],
        evidence=evidence,
        schedules={
            aid: dict(
                run_id=representative["id"],
                recipe=profiles[0]["recipe"],
                phases=representative["search"],
                additional_runs=[
                    r["id"]
                    for r in records
                    if r["metadata"]["runtime"] == "akas" and r != representative
                ],
            )
        },
        output_comparisons=comparisons,
        prefill_observations=[
            dict(
                prefix=prefix,
                akas=rows["akas"][prefix][1]["prefill_observation"],
                baseline=rows["llama"][prefix][1]["prefill_observation"],
            )
            for prefix in prefixes
        ],
        performance_acceptance=dict(
            criterion="AKAS median >= llama.cpp median for completed wall decode at every matched prefix",
            phases=["decode"],
            passed=all(r["decode"]["parity_or_better"] for r in acceptance_rows),
            prefixes=len(prefixes),
            rows=acceptance_rows,
        ),
        showcase_assets_run=assets_run,
        measurement_notes="Decode rates are medians of five samples after two warmups. One request generates 128 outputs; 127 decode steps are timed. Both timers cover completed wall decode including greedy sampling and control dispatch; model load, real prefill, snapshot restore and validation are excluded. AKAS uses GPU token feedback; llama.cpp b11946 Vulkan C API uses its stock greedy sampler. Both restore complete real-prefill recurrent/KV state and the initial prediction. Prompts use fixed synthetic token IDs, never fabricated KV. Prefill observations are single samples in the portable data, not repeated benchmarks, and do not gate decode parity. A dash means no measured median. The maximum-prefix AKAS state is imported from sealed real-prefill run 20261028130402880342-2c41aef6 with checked hashes and unchanged prefill arithmetic/storage. Short prefixes through 8,192 select the 32-partition program; longer ones select the shared-KV 512-partition program. Both reserve a 131,072-token context. Generated sequences differ at some prefixes; performance parity does not establish identical text or end-to-end request latency.",
    )
    render(data, destination, history)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("benchmarks"))
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--assets-run", required=True)
    args = parser.parse_args()
    print(export(args.history, args.runs, args.assets_run))
