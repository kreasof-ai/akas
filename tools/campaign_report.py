"""Export portable campaign tables from retained measurements and baseline history."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from akas.benchmarking.history import digest, write
from akas.benchmarking.workload import prompt_digest
from akas.optimization.context import workload

SCHEMA = "akas.campaign-report.v1"
REPORT_FILES = frozenset(
    (
        "README.md",
        "COMPARISON.md",
        "AKAS.md",
        "BASELINES.md",
        "CANDIDATES.md",
        "measurements.csv",
        "candidate-measurements.csv",
        "candidates.csv",
        "SCHEDULES.json",
        "campaign.json",
        "report.json",
    )
)
METRICS = (
    "prefill_tok_s",
    "decode_tok_s",
    "native_output_tok_s",
    "engine_output_tok_s",
    "burst_output_tok_s",
    "ttft_ms",
    "tpot_ms",
)


def read(path):
    return json.loads(Path(path).read_text())


def report_directory(state, history):
    stamp = datetime.fromisoformat(state["created_at"].replace("Z", "+00:00"))
    return (
        Path(history).resolve()
        / "reports"
        / (stamp.strftime("%Y%m%dT%H%M%SZ") + "-" + state["identity"][:12])
    )


def median(samples, key, scale=1):
    values = [
        s[key] * scale
        for s in samples
        if isinstance(s.get(key), (int, float)) and math.isfinite(s[key])
    ]
    return statistics.median(values) if values else None


def measurement(row, kind, qualified, run_id, protocol=None):
    samples = row.get("samples", [])
    native = row.get("summary", {})
    serving = row.get("serving", {}).get("samples", [])
    engine = row.get("engine", {}).get("samples", [])
    case = row["case"]
    result = dict(
        batch=case["batch"],
        prefix=case["prompt"],
        output=case["generated"],
        status=row["status"],
        qualification="passed"
        if qualified and row["status"] == "ok"
        else "unqualified",
        run_id=run_id,
        error=row.get("error"),
        repetitions=len(samples),
        phase_protocol=protocol or "unrecorded"
        if kind == "akas"
        else next(
            (s.get("phase_protocol") for s in samples if s.get("phase_protocol")), None
        ),
        prefill_tok_s=median(samples, "prefill_tokens_per_second"),
        decode_tok_s=median(samples, "decode_tokens_per_second"),
        native_output_tok_s=median(samples, "output_tokens_per_second")
        if kind == "akas"
        else None,
        engine_output_tok_s=median(engine, "output_tokens_per_second"),
        burst_output_tok_s=median(
            serving if kind == "akas" else samples, "output_tokens_per_second"
        ),
        ttft_ms=median(serving, "ttft_ms")
        if kind == "akas"
        else median(samples, "ttft_seconds", 1000),
        tpot_ms=median(serving, "tpot_ms")
        if kind == "akas"
        else median(samples, "tpot_seconds", 1000),
    )
    for metric, key in (
        ("prefill_tok_s", "prefill_tokens_per_second"),
        ("decode_tok_s", "decode_tokens_per_second"),
    ):
        if result[metric] is None:
            result[metric] = native.get(key, {}).get("median")
    # Keep the actual scalar samples; token IDs and request-level detail remain in run.json.
    result["samples"] = [
        {k: v for k, v in s.items() if isinstance(v, (int, float, str, bool))}
        for s in samples
    ]
    result["engine_samples"] = engine
    result["serving_samples"] = serving
    return result


def record(history, run_id):
    path = Path(history) / "runs" / run_id / "run.json"
    if not path.exists():
        return None
    data = read(path)
    for item in data.get("artifacts", []):
        if (
            item["path"] == "report.json"
            and digest(path.parent / item["path"]) != item["sha256"]
        ):
            raise ValueError("benchmark report checksum mismatch: " + run_id)
    return data


def shapes_for(state):
    for entry in state["entries"]:
        if "context" in entry:
            return entry["context"]["workload"]
    return workload(state["specification"]["workload"])


def identity_map(state):
    result = {}
    spec = state["specification"]
    for model_id, model in state["models"].items():
        for weight, backend, cache, mode in itertools.product(
            spec["formats"], spec["backends"], spec["caches"], model["modes"]
        ):
            key = "-".join((model_id, weight, backend, cache, mode))
            result[key] = dict(
                model_id=model_id,
                model=model["model"],
                weight_format=weight,
                provider=backend,
                cache_request=cache,
                mode=mode,
            )
    return result


def blank_rows(requested, status, reason):
    return [
        dict(
            batch=b,
            prefix=p,
            output=requested["output_tokens"],
            status=status,
            qualification="unsupported" if status == "unsupported" else "not_measured",
            error=reason,
            run_id=None,
            repetitions=0,
            phase_protocol=None,
            **dict.fromkeys(METRICS),
        )
        for b, p in itertools.product(requested["batches"], requested["prefix_tokens"])
    ]


def baseline_combinations(state, model):
    formats = state["specification"]["formats"]
    return [
        (runtime, "cuda", "bf16", mode)
        for runtime in ("vllm", "sglang")
        for mode in model["modes"]
        if "bf16" in formats
    ] + [
        ("llama", provider, fmt, "ar")
        for provider in ("cuda", "vulkan")
        for fmt in formats
        if fmt != "bf16"
    ]


def metadata_for(data):
    meta = data["metadata"]
    return dict(
        run_id=data["id"],
        started_at=data["started_at"],
        completed_at=data.get("completed_at"),
        status=data["status"],
        source={k: v for k, v in data["source"].items() if k != "files"},
        metadata=meta,
        dependencies=data.get("dependencies", {}),
        environment=data.get("environment", {}),
        change=data.get("change"),
        workload=data.get("workload"),
        artifacts=data.get("artifacts", []),
        checkpoint_sha256=meta.get("checkpoint_sha256"),
    )


def job_record(history, job):
    run_id = job.get("run_id")
    if not run_id and job.get("history_record"):
        run_id = Path(job["history_record"]).parent.name
    pointer = Path(job.get("directory", ".")) / "history.json"
    if not run_id and pointer.exists():
        run_id = read(pointer).get("id")
    return record(history, run_id) if run_id else None


def precision_comparisons(state, history, evidence):
    """Publish paired cache probes from checksummed, sealed diagnostic records."""
    comparisons = []
    expected_hashes = {
        item.get("checkpoint_sha256")
        for item in evidence.values()
        if item.get("checkpoint_sha256")
    }
    for run_id in state.get("precision_comparison_runs", []):
        data = record(history, run_id)
        if (
            not data
            or data["status"] != "passed"
            or data["metadata"].get("protocol") != "kv-output-comparison-v1"
            or not data.get("completed_at")
        ):
            raise ValueError(
                "precision comparison requires a sealed, passed probe: " + run_id
            )
        if data["metadata"].get("checkpoint_sha256") not in expected_hashes:
            raise ValueError(
                "precision comparison checkpoint differs from the campaign"
            )
        parent_id = data.get("change", {}).get("parent_run")
        parent = record(history, parent_id) if parent_id else None
        if (
            not parent
            or parent["status"] != "passed"
            or parent["metadata"].get("kv_dtype") != "float16"
            or parent["metadata"].get("protocol") != "kv-output-comparison-v1"
            or parent["metadata"].get("checkpoint_sha256")
            != data["metadata"].get("checkpoint_sha256")
        ):
            raise ValueError("precision comparison requires its matching F16 reference")

        def outputs(item):
            artifact = next(
                (a for a in item["artifacts"] if a["path"] == "precision/outputs.json"),
                None,
            )
            if artifact is None:
                raise ValueError("precision comparison outputs are missing")
            path = Path(history) / "runs" / item["id"] / artifact["path"]
            if digest(path) != artifact["sha256"]:
                raise ValueError("precision comparison output checksum mismatch")
            payload = read(path)
            if payload["run_id"] != item["id"]:
                raise ValueError("precision comparison output identity mismatch")
            return payload["cases"]

        references = {row["name"]: row for row in outputs(parent)}
        cases = []
        for row in outputs(data):
            reference = references.pop(row["name"], None)
            if (
                reference is None
                or reference["prompt_sha256"] != row["prompt_sha256"]
                or reference["output_ids"] != row["greedy_comparison"]["f16_output_ids"]
            ):
                raise ValueError("precision comparison prompt or continuation differs")
            old, new = reference["output_ids"], row["output_ids"]
            prefix = next(
                (i for i, (a, b) in enumerate(zip(old, new)) if a != b),
                min(len(old), len(new)),
            )
            steps = row["teacher_forced"]["steps"]
            cases.append(
                dict(
                    name=row["name"],
                    prefix_tokens=row["prefix"],
                    prompt_sha256=row["prompt_sha256"],
                    reference_output_ids=old,
                    candidate_output_ids=new,
                    reference_output_text=reference["output_text"],
                    candidate_output_text=row["output_text"],
                    exact=old == new,
                    common_prefix_tokens=prefix,
                    reference_tokens=len(old),
                    candidate_tokens=len(new),
                    teacher_forced_steps=len(steps),
                    top1_agreement=statistics.mean(
                        int(step["top1_equal"]) for step in steps
                    ),
                    mean_kl_f16_to_candidate=statistics.mean(
                        step["kl_f16_to_fp8"] for step in steps
                    ),
                    mean_reference_token_nll_delta=statistics.mean(
                        step["reference_token_nll_delta"] for step in steps
                    ),
                )
            )
        if references or not cases:
            raise ValueError("precision comparison has incomplete prompt coverage")
        evidence[data["id"]], evidence[parent["id"]] = (
            metadata_for(data),
            metadata_for(parent),
        )
        comparisons.append(
            dict(
                run_id=run_id,
                reference_run_id=parent_id,
                reference_kv_dtype=parent["metadata"]["kv_dtype"],
                candidate_kv_dtype=data["metadata"]["kv_dtype"],
                qualification=data["metadata"]["qualification"],
                cases=cases,
            )
        )
    return comparisons


def text_comparisons(state, history, evidence):
    """Keep each real example separate from synthetic shape measurements."""
    groups = []
    expected = {item.get("checkpoint_sha256") for item in evidence.values()
                if item.get("checkpoint_sha256")}
    for run_id in state.get("text_workload_runs", []):
        data = record(history, run_id)
        if not data or data["status"] != "passed" or not data.get("completed_at"):
            raise ValueError("text comparison requires a sealed, passed run: " + run_id)
        meta = data["metadata"]
        if meta.get("checkpoint_sha256") not in expected:
            raise ValueError("text comparison checkpoint differs from campaign")
        artifact = next((a for a in data["artifacts"]
                         if a["path"] == "benchmark/text-workload.json"), None)
        path = Path(history) / "runs" / run_id / "benchmark/text-workload.json"
        if (not artifact or digest(path) != artifact["sha256"] or
            artifact["sha256"] != meta.get("text_workload_sha256")):
            raise ValueError("text workload checksum mismatch")
        workload = read(path)
        if workload.get("tokenizer_checkpoint_sha256") != meta["checkpoint_sha256"]:
            raise ValueError("text workload tokenizer identity differs")
        cases = {item["id"]: item for item in workload["cases"]}
        selected = []
        seen = set()
        for row in data["results"]:
            item = row.get("text_case", {})
            case = cases.get(item.get("id"))
            ar = row.get("ar", {})
            if (not case or case["id"] in seen or
                case["split"] != meta.get("dataset_split") or
                prompt_digest([case["token_ids"]]) != case["prompt_sha256"] or
                any(candidate.get("prompt_sha256") != case["prompt_sha256"]
                    for candidate in (row, ar)) or
                row["case"] != ar.get("case") or
                row["case"]["prompt"] != case["prefix_tokens"] or
                row["case"]["generated"] != workload["output_tokens"]):
                raise ValueError("text comparison inputs or split differ")
            seen.add(case["id"])
            stats = row["native_speculation"]
            rounds, proposed = stats["request_rounds"], stats["proposed"]
            ar_rate = ar["summary"]["decode_tokens_per_second"]["median"]
            rate = row["summary"]["decode_tokens_per_second"]["median"]
            ar_prefill = ar["summary"]["prefill_tokens_per_second"]["median"]
            prefill = row["summary"]["prefill_tokens_per_second"]["median"]
            if (rounds < 1 or not 0 <= stats["accepted"] <= proposed or
                stats["emitted"] != row["decode_tokens"] or
                any(not math.isfinite(value) or value <= 0
                    for value in (rate, ar_rate, prefill, ar_prefill))):
                raise ValueError("invalid text comparison accounting")
            actual, reference = row["output_ids"], row["ar_output_ids"]
            if (len(actual) != 1 or len(reference) != 1 or
                any(len(ids) != workload["output_tokens"] for ids in (*actual, *reference))):
                raise ValueError("incomplete real-text outputs")
            agreement = dict(exact=actual == reference, common_prefix_tokens=next(
                (i for i, (a, b) in enumerate(zip(actual[0], reference[0])) if a != b),
                workload["output_tokens"]))
            if agreement != row["output_agreement"]:
                raise ValueError("text output agreement differs from retained tokens")
            selected.append(dict(
                id=case["id"], category=case["category"], split=case["split"],
                source=case["source"], example_id=case["example_id"],
                prefix_tokens=case["prefix_tokens"], prompt_sha256=case["prompt_sha256"],
                preprocessing=case["preprocessing"], ar_decode_tps=ar_rate,
                ar_prefill_tps=ar_prefill, speculative_prefill_tps=prefill,
                speculative_decode_tps=rate, decode_speedup=rate / ar_rate,
                accepted=stats["accepted"], proposed=proposed, emitted=stats["emitted"],
                request_rounds=rounds, tokens_per_round=stats["emitted"] / rounds,
                acceptance=stats["accepted"] / proposed if proposed else 0,
                output_agreement=agreement,
                output_ids=row["output_ids"], ar_output_ids=row["ar_output_ids"],
                output_text=row["output_text"], ar_output_text=row["ar_output_text"],
                warmup=row["warmup"], repeats=row["repeats"]))
        if seen != {key for key, value in cases.items()
                    if value["split"] == meta.get("dataset_split")}:
            raise ValueError("incomplete text workload split")
        groups.append(dict(run_id=run_id, confidence_threshold=meta["confidence_threshold"],
                           variant=meta.get("variant"),
                           split=meta["dataset_split"], kv_dtype=meta["kv_dtype"],
                           draft_kv_dtype=meta["draft_kv_dtype"], sources=workload["sources"],
                           ar_reference_run_id=meta.get("ar_reference_run_id", run_id),
                           qualification=meta["qualification"], cases=selected))
        evidence[run_id] = metadata_for(data)
    return groups


def collect(campaign, history):
    campaign = Path(campaign).resolve()
    history = Path(history).resolve()
    state = read(campaign)
    requested = shapes_for(state)
    identities = identity_map(state)
    profiles, attempts, schedules, evidence = [], [], {}, {}
    for entry in state["entries"]:
        identity = identities[entry["id"]]
        child_path = campaign.parent / entry["id"] / "campaign.json"
        child = read(child_path) if child_path.exists() else {}
        probes = child.get("probes", [])
        errors = [
            e.get("error")
            for p in probes
            for e in p.get("errors", [])
            if e.get("error")
        ]
        errors += [p["error"] for p in probes if p.get("error")]
        errors += [j["error"] for j in child.get("jobs", []) if j.get("error")]
        reason = (
            entry.get("error")
            or "; ".join(dict.fromkeys(errors))
            or (
                "No qualified deployment retained"
                if not entry.get("deployment")
                else None
            )
        )
        profile = dict(
            id=entry["id"],
            kind="akas",
            runtime="akas",
            **identity,
            status=entry.get("status", "pending"),
            error=reason,
            spent_seconds=entry.get("spent_seconds", 0),
            selected_run_id=None,
            recipe=None,
            kv_dtype=next(
                (
                    p.get("model_options", {}).get("kv_dtype")
                    for p in probes
                    if p.get("model_options", {}).get("kv_dtype")
                ),
                None,
            ),
            measurements=blank_rows(requested, entry.get("status", "pending"), reason),
        )
        saved = {}
        for job in child.get("jobs", []):
            data = job_record(history, job)
            rows = []
            if data:
                run_id = data["id"]
                evidence[run_id] = metadata_for(data)
                rows = [
                    measurement(
                        r,
                        "akas",
                        data["status"] == "passed",
                        run_id,
                        data["metadata"].get("protocol"),
                    )
                    for r in data["results"]
                ]
                saved[run_id] = (data, job, rows)
            progress = Path(job.get("directory", ".")) / "progress.json"
            attempts.append(
                dict(
                    configuration=entry["id"],
                    recipe=job["recipe"],
                    status=job["status"],
                    run_id=data["id"] if data else job.get("run_id"),
                    error=job.get("error"),
                    elapsed_seconds=job.get("elapsed_seconds"),
                    last_phase=read(progress).get("phase")
                    if progress.exists()
                    else None,
                    sampled_peak_device_bytes=job.get("sampled_peak_device_bytes"),
                    measurements=rows,
                )
            )
        winner = entry.get("winner") or child.get("winner") or {}
        selected = saved.get(winner.get("run_id"))
        if selected and selected[0]["status"] == "passed" and entry.get("deployment"):
            data, job, rows = selected
            if data["source"]["tree_sha256"] != state["source"]["tree_sha256"]:
                raise ValueError("winner source differs from campaign: " + entry["id"])
            profile.update(
                selected_run_id=data["id"],
                recipe=job["recipe"],
                kv_dtype=data["metadata"].get("kv_dtype"),
                peak_device_bytes=job.get("sampled_peak_device_bytes"),
                qualifying_status="passed",
                measurements=rows,
            )
            first = next(
                (v for v in saved.values() if v[0]["status"] == "passed"), None
            )
            if first:
                before = {
                    (r["batch"], r["prefix"]): r["burst_output_tok_s"] for r in first[2]
                }
                ratios = [
                    r["burst_output_tok_s"] / before[r["batch"], r["prefix"]]
                    for r in rows
                    if r["burst_output_tok_s"] and before.get((r["batch"], r["prefix"]))
                ]
                profile["reference_run_id"] = first[0]["id"]
                profile["geomean_http_speedup"] = (
                    math.exp(statistics.mean(math.log(v) for v in ratios))
                    if ratios
                    else None
                )
            schedules[entry["id"]] = dict(
                run_id=data["id"], recipe=job["recipe"], phases=data.get("search", [])
            )
        else:
            profile["qualifying_status"] = (
                "unsupported"
                if entry.get("status") == "unsupported"
                else "not_measured"
            )
            # Display available partial measurements with their own run IDs and failed status.
            partial = {}
            for data, job, rows in saved.values():
                for row in rows:
                    if any(row.get(k) is not None for k in METRICS):
                        partial[row["batch"], row["prefix"]] = row
            profile["measurements"] = [
                partial.get((r["batch"], r["prefix"]), r)
                for r in profile["measurements"]
            ]
        profiles.append(profile)
    # Freeze baseline selection to measurements completed before this campaign began.
    index_path = history / "index.json"
    index = read(index_path)["runs"] if index_path.exists() else []
    expected_shapes = {
        (b, p, requested["output_tokens"])
        for b, p in itertools.product(requested["batches"], requested["prefix_tokens"])
    }
    selected = {}
    for item in sorted(index, key=lambda r: r["started_at"]):
        if (
            item.get("purpose") != "baseline"
            or item["started_at"] > state["created_at"]
        ):
            continue
        shape_set = {
            (s["batch"], s["prefix_tokens"], s["output_tokens"])
            for s in item.get("shapes", [])
        }
        if shape_set != expected_shapes:
            continue
        key = (
            item["model"],
            item["runtime"],
            item["provider"],
            item["weight_format"],
            item["mode"],
        )
        candidate = record(history, item["id"])
        if (
            candidate
            and candidate.get("completed_at")
            and candidate["status"] != "running"
            and candidate["metadata"].get("protocol") == "baseline-burst-v3"
            and (candidate.get("completed_at") or candidate["started_at"])
            <= state["created_at"]
        ):
            selected[key] = item["id"]
    for model_id, model in state["models"].items():
        for runtime, provider, weight, mode in baseline_combinations(state, model):
            key = (model["model"], runtime, provider, weight, mode)
            data = record(history, selected[key]) if key in selected else None
            if data and (
                data["metadata"].get("protocol") != "baseline-burst-v3"
                or (data.get("completed_at") or data["started_at"])
                > state["created_at"]
            ):
                data = None
            profile_id = "baseline-" + "-".join(
                (model_id, runtime, provider, weight, mode)
            )
            profile = dict(
                id=profile_id,
                kind="baseline",
                runtime=runtime,
                model_id=model_id,
                model=model["model"],
                provider=provider,
                weight_format=weight,
                cache_request="default",
                mode=mode,
                status=data["status"] if data else "not_measured",
                error=data["metadata"].get("error")
                if data
                else "No matching full-workload baseline before campaign start",
                selected_run_id=data["id"] if data else None,
                recipe=None,
                qualifying_status="passed"
                if data and data["status"] == "passed"
                else "unqualified"
                if data
                else "not_measured",
                kv_dtype=data["metadata"].get("kv_dtype") if data else None,
                peak_device_bytes=data["metadata"].get("sampled_peak_device_bytes")
                if data
                else None,
                measurements=[
                    measurement(r, "baseline", data["status"] == "passed", data["id"])
                    for r in data["results"]
                ]
                if data
                else blank_rows(requested, "not_measured", "No matching baseline"),
            )
            if data:
                evidence[data["id"]] = metadata_for(data)
            profiles.append(profile)
    return dict(
        schema=SCHEMA,
        generated_at=datetime.now(timezone.utc).isoformat(),
        campaign=state,
        campaign_sha256=digest(campaign),
        workload=requested,
        profiles=profiles,
        attempts=attempts,
        evidence=evidence,
        schedules=schedules,
        precision_comparisons=precision_comparisons(state, history, evidence),
        text_comparisons=text_comparisons(state, history, evidence),
    )


def text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content.rstrip() + "\n" if path.suffix == ".md" else content)
    temporary.replace(path)


def fmt(value):
    return "—" if value is None else f"{value:,.1f}"


def clean(value):
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def evidence_link(run_id, out, history):
    if not run_id:
        return "—"
    path = os.path.relpath(Path(history) / "runs" / run_id / "run.json", out)
    return f"[{run_id}]({path})"


def csv_file(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(
        {**row, "error": str(row["error"]).strip()}
        if row.get("error") is not None
        else row
        for row in rows
    )
    text(path, stream.getvalue())


def table_rows(profiles):
    for p in profiles:
        for r in p["measurements"]:
            yield dict(
                profile=p["id"],
                kind=p["kind"],
                model=p["model"],
                runtime=p["runtime"],
                provider=p["provider"],
                weight_format=p["weight_format"],
                kv_dtype=p.get("kv_dtype"),
                cache_request=p["cache_request"],
                mode=p["mode"],
                **r,
            )


def render(data, out, history):
    validate(data)
    out = Path(out).resolve()
    history = Path(history).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = data["campaign"]
    akas = [p for p in data["profiles"] if p["kind"] == "akas"]
    baselines = [p for p in data["profiles"] if p["kind"] == "baseline"]
    qualified = [p for p in akas if p["qualifying_status"] == "passed"]
    ceiling = state.get("configuration_budget_seconds")
    ceiling_text = "not set" if ceiling is None else f"{ceiling / 3600:.2f} hours"
    notes = "All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput."
    lines = [
        "# Campaign benchmark report",
        "",
        f"UTC: {state['created_at']} → {state.get('updated_at', 'running')}. Status: `{state['status']}`. Active time: {state.get('spent_seconds', 0) / 3600:.2f} hours; per-configuration ceiling: {ceiling_text}.",
        "",
        f"Source: `{state['source'].get('revision')}`; measured source tree: `{state['source']['tree_sha256']}`.",
        "",
        f"Qualified AKAS deployments: **{len(qualified)}/{len(akas)}**. Qualified baseline profiles: **{sum(p['qualifying_status'] == 'passed' for p in baselines)}/{len(baselines)}**.",
        "",
        "- [Side-by-side tables for every requested shape](COMPARISON.md)",
        "- [All AKAS configurations and per-shape measurements](AKAS.md)",
        "- [All baseline profiles and per-shape measurements](BASELINES.md)",
        "- [Every candidate attempt](CANDIDATES.md)",
        "- [Full measurement CSV](measurements.csv) · [Candidate measurement CSV](candidate-measurements.csv) · [Candidate outcomes CSV](candidates.csv)",
        "- [Selected kernel implementations and schedules](SCHEDULES.json) · [Portable report data](report.json) · [Campaign snapshot](campaign.json)",
        "",
        notes,
        "",
        "## Hardware and measurement provenance",
        "",
        "| Hardware | Driver | Baseline/runtime | Version or revision | Source | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for p in qualified + baselines:
        e = data["evidence"].get(p.get("selected_run_id"), {})
        m = e.get("metadata", {})
        device = m.get("device", {})
        name = (
            device.get("name", str(device)) if isinstance(device, dict) else str(device)
        )
        driver = (
            device.get("driver", device.get("driver_version", ""))
            if isinstance(device, dict)
            else ""
        )
        version = m.get("runtime_version") or m.get("runtime_revision")
        if not version and p["runtime"] == "akas":
            version = e.get("dependencies", {}).get("akas", "source checkout")
        lines.append(
            f"| {clean(name)} | {clean(driver)} | {p['runtime']} / {p['model_id']} / {p['weight_format']} / {p['mode']} | {clean(version)} | `{e.get('source', {}).get('revision', 'unknown')}` | {evidence_link(p.get('selected_run_id'), out, history)} |"
        )
    environment = next(
        (
            e["environment"]
            for e in data["evidence"].values()
            if e.get("environment", {}).get("nvidia_smi")
        ),
        {},
    )
    if environment:
        lines += [
            "",
            f"Captured host: {environment.get('cpu_count', 'unrecorded')} logical CPUs; {environment.get('platform', 'unrecorded platform')}.",
            "",
            "```text",
            environment["nvidia_smi"],
            "```",
        ]
    lines += [
        "",
        "Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.",
        "",
        "## Selected configurations",
        "",
        "HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.",
        "",
        "| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for p in qualified:
        recipe = p["recipe"]
        peak = p.get("peak_device_bytes")
        ratio = p.get("geomean_http_speedup")
        lines.append(
            f"| {p['id']} | {p.get('kv_dtype')} | {recipe.get('chunk')} | {recipe.get('prefill_requests')} | {recipe.get('graphs')} | {recipe.get('trials')} | {f'{ratio:.3f}x' if ratio is not None else '—'} | {fmt(peak / 2**30) if peak is not None else '—'} | {evidence_link(p.get('selected_run_id'), out, history)} |"
        )
    lines += [
        "",
        "## Complete requested AKAS coverage",
        "",
        "| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for p in akas:
        outcome = (
            "qualified; " + p["status"]
            if p["qualifying_status"] == "passed"
            else p["status"]
        )
        lines.append(
            f"| {p['model_id']} | {p['weight_format']} | {p['provider']} | {p['cache_request']} | {p['mode']} | [{outcome}](AKAS.md#{p['id']}) | {p.get('spent_seconds', 0) / 60:.1f} | {evidence_link(p.get('selected_run_id'), out, history)} | {clean(p.get('error')) if p['qualifying_status'] != 'passed' else ''} |"
        )
    counts = Counter(a["status"] for a in data["attempts"])
    errors = Counter(
        a.get("error", "") for a in data["attempts"] if a["status"] != "passed"
    )
    lines += [
        "",
        "## Candidate outcomes",
        "",
        f"{len(data['attempts'])} attempts: "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        + ".",
        "",
        "| Rejection or deadline | Attempts |",
        "| --- | ---: |",
    ]
    for reason, count in errors.most_common():
        lines.append(f"| {clean(reason)} | {count} |")
    for comparison in data.get("precision_comparisons", []):
        positions = sum(row["teacher_forced_steps"] for row in comparison["cases"])
        agreement = (
            sum(
                row["top1_agreement"] * row["teacher_forced_steps"]
                for row in comparison["cases"]
            )
            / positions
        )
        divergence = (
            sum(
                row["mean_kl_f16_to_candidate"] * row["teacher_forced_steps"]
                for row in comparison["cases"]
            )
            / positions
        )
        lines += [
            "",
            "## Cache precision and generated outputs",
            "",
            f"Reference: `{comparison['reference_kv_dtype']}`; candidate: `{comparison['candidate_kv_dtype']}`. "
            f"{sum(row['exact'] for row in comparison['cases'])}/{len(comparison['cases'])} greedy outputs match exactly.",
            f"Across {positions} shared-history positions, top-1 agreement is **{agreement:.2%}** and mean KL divergence is **{divergence:.6f}**.",
            "",
            comparison["qualification"],
            "",
            "Common prefix counts matching tokens before the first changed token. Shared-history top-1 and KL "
            "compare both caches after supplying the same F16-generated continuation. Later free-generation "
            "differences can compound after an earlier changed token; they are not an accuracy score.",
            "",
            f"Reference evidence: {evidence_link(comparison['reference_run_id'], out, history)}; "
            f"candidate evidence: {evidence_link(comparison['run_id'], out, history)}. "
            "Both generated texts and token IDs are retained in report.json; prompt IDs and per-step metrics remain in the checksummed run artifacts.",
            "",
            "| Prompt | Prefix | F16 outputs | Candidate outputs | Exact | Common prefix | Shared-history steps | Top-1 agreement | Mean KL |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in comparison["cases"]:
            lines.append(
                f"| {clean(row['name'])} | {row['prefix_tokens']} | {row['reference_tokens']} | {row['candidate_tokens']} | "
                f"{'yes' if row['exact'] else 'no'} | {row['common_prefix_tokens']} | {row['teacher_forced_steps']} | "
                f"{row['top1_agreement']:.1%} | {row['mean_kl_f16_to_candidate']:.6f} |"
            )
    if data.get("text_comparisons"):
        lines += ["", "## Real-text speculative decoding", "",
                  "Matched real prompts distinguish draft acceptance from synthetic shape tests. "
                  "Calibration and held-out examples are separate. Long cases with one sample are diagnostic measurements. "
                  "All rates are tokens/second for one request generating 128 tokens with EOS disabled.",
                  "", "AR uses device token feedback. Speculative decode includes proposal, verification, commit, repair and coordinator readbacks. "
                  "Short cases use one warmup and three samples; long cases use one sample. "
                  "Dataset revisions, example IDs, prompt hashes, crop recipes, full methodology and both outputs remain in report.json and the checksummed workload artifacts."]
    for comparison in data.get("text_comparisons", []):
        variant = f"; {comparison['variant']}" if comparison.get("variant") else ""
        lines += ["", f"### {comparison['split']}; confidence {comparison['confidence_threshold']}{variant}", "",
                  f"Target/draft KV: `{comparison['kv_dtype']}` / `{comparison['draft_kv_dtype']}`. "
                  f"Evidence: {evidence_link(comparison['run_id'], out, history)}; "
                  f"AR reference: {evidence_link(comparison.get('ar_reference_run_id', comparison['run_id']), out, history)}.", "",
                  "| Example | Category | Prefix | AR prefill | Speculative prefill | AR decode | Speculative decode | Decode speedup | Acceptance | Tokens/round | Exact output | Common prefix |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
        for row in comparison["cases"]:
            agreement = row["output_agreement"]
            lines.append(f"| {clean(row['id'])} | {row['category']} | {row['prefix_tokens']} | "
                         f"{row['ar_prefill_tps']:.2f} | {row['speculative_prefill_tps']:.2f} | "
                         f"{row['ar_decode_tps']:.2f} | {row['speculative_decode_tps']:.2f} | "
                         f"{row['decode_speedup']:.3f}x | {row['acceptance']:.1%} | "
                         f"{row['tokens_per_round']:.2f} | {'yes' if agreement['exact'] else 'no'} | {agreement['common_prefix_tokens']} |")
    text(out / "README.md", "\n".join(lines) + "\n")
    for kind, profiles, filename in [
        ("akas", akas, "AKAS.md"),
        ("baseline", baselines, "BASELINES.md"),
    ]:
        lines = [
            "# " + ("AKAS configurations" if kind == "akas" else "Baseline profiles"),
            "",
            "[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)",
            "",
            notes,
            "",
        ]
        for p in profiles:
            meta = (
                data["evidence"].get(p.get("selected_run_id"), {}).get("metadata", {})
            )
            host = meta.get("cpu_peak_rss_bytes")
            device = p.get("peak_device_bytes")
            lines += [
                f'<a id="{p["id"]}"></a>',
                f"## {p['model_id']} / {p['runtime']} / {p['weight_format']} / {p['provider']} / {p['cache_request']} / {p['mode']}",
                "",
                f"Profile status: **{p['status']}**; qualification: **{p['qualifying_status']}**; effective KV dtype: `{p.get('kv_dtype') or 'unrecorded'}`. Evidence: {evidence_link(p.get('selected_run_id'), out, history)}.",
                "",
                f"Peak device GiB: {fmt(device / 2**30) if device is not None else '—'}; peak host process RSS GiB: {fmt(host / 2**30) if host is not None else '—'}. Loading and in-process compilation are included in host RSS.",
                "",
            ]
            if p.get("error"):
                lines += [clean(p["error"]), ""]
            if p.get("recipe"):
                lines += [
                    "Selected recipe: `"
                    + json.dumps(p["recipe"], sort_keys=True)
                    + "`.",
                    "",
                ]
            lines += [
                "| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |",
                "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
            for r in p["measurements"]:
                values = " | ".join(fmt(r.get(k)) for k in METRICS)
                lines.append(
                    f"| {r['batch']} | {r['prefix']} | {r['output']} | {r.get('phase_protocol') or '—'} | {values} | {r['repetitions']} | {r['qualification']} | {evidence_link(r.get('run_id'), out, history)} |"
                )
            lines.append("")
        text(out / filename, "\n".join(lines) + "\n")
    comparisons(data, out, history, notes)
    lines = [
        "# Candidate attempts",
        "",
        "[Campaign overview](README.md) · [All measured candidate shapes](candidate-measurements.csv) · [CSV](candidates.csv)",
        "",
        "| Configuration | Recipe | Trials allowance | Outcome | Last phase | Seconds | Peak device GiB | Selected | Evidence | Reason |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    candidate_rows = []
    candidate_values = []
    selected_ids = {p.get("selected_run_id") for p in qualified}
    for a in data["attempts"]:
        peak = a.get("sampled_peak_device_bytes")
        recipe = a["recipe"]
        selected = a.get("run_id") in selected_ids
        lines.append(
            f"| {a['configuration']} | `{recipe['id']}` | {recipe['trials']} | {a['status']} | {a.get('last_phase') or '—'} | {fmt(a.get('elapsed_seconds'))} | {fmt(peak / 2**30) if peak is not None else '—'} | {'yes' if selected else ''} | {evidence_link(a.get('run_id'), out, history)} | {clean(a.get('error'))} |"
        )
        candidate_rows.append(
            dict(
                configuration=a["configuration"],
                recipe=recipe["id"],
                trials=recipe["trials"],
                status=a["status"],
                last_phase=a.get("last_phase"),
                elapsed_seconds=a.get("elapsed_seconds"),
                peak_device_bytes=peak,
                selected=selected,
                run_id=a.get("run_id"),
                error=a.get("error"),
            )
        )
        candidate_values.extend(
            dict(
                configuration=a["configuration"],
                recipe=recipe["id"],
                candidate_status=a["status"],
                **r,
            )
            for r in a["measurements"]
            if any(r.get(k) is not None for k in METRICS)
        )
    text(out / "CANDIDATES.md", "\n".join(lines) + "\n")
    csv_file(
        out / "measurements.csv",
        table_rows(data["profiles"]),
        [
            "profile",
            "kind",
            "model",
            "runtime",
            "provider",
            "weight_format",
            "cache_request",
            "kv_dtype",
            "mode",
            "batch",
            "prefix",
            "output",
            "status",
            "qualification",
            "phase_protocol",
            *METRICS,
            "repetitions",
            "run_id",
            "error",
        ],
    )
    csv_file(
        out / "candidate-measurements.csv",
        candidate_values,
        [
            "configuration",
            "recipe",
            "candidate_status",
            "batch",
            "prefix",
            "output",
            "status",
            "qualification",
            "phase_protocol",
            *METRICS,
            "repetitions",
            "run_id",
            "error",
        ],
    )
    csv_file(
        out / "candidates.csv",
        candidate_rows,
        [
            "configuration",
            "recipe",
            "trials",
            "status",
            "last_phase",
            "elapsed_seconds",
            "peak_device_bytes",
            "selected",
            "run_id",
            "error",
        ],
    )
    write(
        out / "SCHEDULES.json",
        dict(
            schema="akas.campaign-schedules.v1",
            configurations={
                key: {
                    **value,
                    "record": os.path.relpath(
                        Path(history) / "runs" / value["run_id"] / "run.json", out
                    ),
                }
                for key, value in data["schedules"].items()
            },
        ),
    )
    write(out / "campaign.json", state)
    write(out / "report.json", data)
    index(history)


def comparisons(data, out, history, notes):
    lines = [
        "# Side-by-side throughput",
        "",
        "[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)",
        "",
        notes,
        "",
        "Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.",
        "",
    ]
    profiles = data["profiles"]
    spec = data["campaign"]["specification"]
    requested = data["workload"]
    for model_id, model in data["campaign"]["models"].items():
        for weight in spec["formats"]:
            for mode in model["modes"]:
                columns = [
                    p
                    for p in profiles
                    if p["kind"] == "akas"
                    and p["model_id"] == model_id
                    and p["weight_format"] == weight
                    and p["mode"] == mode
                ]
                columns += [
                    p
                    for p in profiles
                    if p["kind"] == "baseline"
                    and p["model_id"] == model_id
                    and p["weight_format"] == weight
                    and p["mode"] == mode
                ]
                lines += [f"## {model_id} / {weight} / {mode}", ""]
                lookups = [
                    {(r["batch"], r["prefix"]): r for r in p["measurements"]}
                    for p in columns
                ]
                for metric, title in (
                    ("prefill_tok_s", "Prefill tok/s — reported phase timing"),
                    ("decode_tok_s", "Decode tok/s — reported phase timing"),
                    ("burst_output_tok_s", "Complete-burst output tok/s"),
                ):
                    labels = []
                    for p in columns:
                        boundary = (
                            (
                                "HTTP"
                                if p["kind"] == "akas" or p["runtime"] == "llama"
                                else "native call"
                            )
                            if metric == "burst_output_tok_s"
                            else next(
                                (
                                    r["phase_protocol"]
                                    for r in p["measurements"]
                                    if r.get("phase_protocol")
                                ),
                                "unmeasured",
                            )
                        )
                        labels.append(
                            f"[{p['runtime']} {p['provider']} {p['cache_request']} / {boundary}]({'AKAS.md' if p['kind'] == 'akas' else 'BASELINES.md'}#{p['id']})"
                        )
                    lines += [
                        f"### {title}",
                        "",
                        "| B | Prefix | " + " | ".join(labels) + " |",
                        "| ---: | ---: | " + " | ".join("---:" for p in columns) + " |",
                    ]
                    for b, prefix in itertools.product(
                        requested["batches"], requested["prefix_tokens"]
                    ):
                        cells = []
                        for lookup in lookups:
                            r = lookup.get((b, prefix), {})
                            value = r.get(metric)
                            cell = fmt(value)
                            if value is not None and r.get("qualification") != "passed":
                                cell += " †"
                            cells.append(cell)
                        lines.append(f"| {b} | {prefix} | " + " | ".join(cells) + " |")
                    lines.append("")
    text(out / "COMPARISON.md", "\n".join(lines) + "\n")


def index(history):
    history = Path(history)
    reports = history / "reports"
    lines = [
        "# Campaign reports",
        "",
        "Each report contains the full requested matrix, all available AKAS and baseline measurements, failed or unsupported rows, runtime settings, source provenance and selected kernel schedules. Raw measurements remain under `../runs/`.",
        "",
        "| Started UTC | Workload | Source | Qualified AKAS | Qualified baselines | Report |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for path in sorted(reports.glob("*/report.json"), reverse=True):
        data = read(path)
        profiles = data["profiles"]
        a = [p for p in profiles if p["kind"] == "akas"]
        b = [p for p in profiles if p["kind"] == "baseline"]
        state = data["campaign"]
        lines.append(
            f"| {state['created_at']} | {data['workload'].get('id', 'custom')} | `{state['source'].get('revision')}` | {sum(p['qualifying_status'] == 'passed' for p in a)}/{len(a)} | {sum(p['qualifying_status'] == 'passed' for p in b)}/{len(b)} | [full tables]({path.parent.name}/README.md) |"
        )
    text(reports / "README.md", "\n".join(lines) + "\n")


def export(campaign, *, history="benchmarks", out=None):
    data = collect(campaign, history)
    destination = Path(out) if out else report_directory(data["campaign"], history)
    render(data, destination, history)
    return destination


def validate(data):
    """Require the versioned report contract before rendering or publishing."""
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported report schema: {data.get('schema')!r}")
    required = {
        "generated_at",
        "campaign",
        "campaign_sha256",
        "workload",
        "profiles",
        "attempts",
        "evidence",
        "schedules",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing report fields: {sorted(missing)}")
    requested = data["workload"]
    expected = {
        (b, p, requested["output_tokens"])
        for b, p in itertools.product(requested["batches"], requested["prefix_tokens"])
    }
    identifiers = set()
    for profile in data["profiles"]:
        name = profile["id"]
        if name in identifiers:
            raise ValueError(f"Duplicate report profile: {name}")
        identifiers.add(name)
        rows = profile["measurements"]
        actual = {(r["batch"], r["prefix"], r["output"]) for r in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError(f"Incomplete or duplicate workload shapes: {name}")
        for row in rows:
            required_row = {*METRICS, "status", "qualification", "phase_protocol"}
            if required_row - row.keys():
                raise ValueError(f"Missing measurement fields: {name}")
            if any(
                row[k] is not None
                and (
                    not isinstance(row[k], (int, float))
                    or isinstance(row[k], bool)
                    or not math.isfinite(row[k])
                    or row[k] < 0
                )
                for k in METRICS
            ):
                raise ValueError(f"Invalid measurement value: {name}")
    expected_akas = set(identity_map(data["campaign"]))
    actual_akas = {p["id"] for p in data["profiles"] if p["kind"] == "akas"}
    if expected_akas != actual_akas:
        raise ValueError("Report does not cover the complete requested AKAS matrix")
    expected_baselines = {
        "baseline-" + "-".join((model_id, *combination))
        for model_id, model in data["campaign"]["models"].items()
        for combination in baseline_combinations(data["campaign"], model)
    }
    actual_baselines = {p["id"] for p in data["profiles"] if p["kind"] == "baseline"}
    if expected_baselines != actual_baselines:
        raise ValueError("Report does not cover the complete baseline matrix")
    if any(p["kind"] not in {"akas", "baseline"} for p in data["profiles"]):
        raise ValueError("Unsupported report profile kind")


def check(history="benchmarks"):
    """Read-only audit of the canonical report layout and generated content."""
    history = Path(history).resolve()
    reports = history / "reports"
    allowed_guides = {history / "README.md", history / "RESULTS.md"}
    for path in history.rglob("*.md"):
        relative = path.relative_to(history)
        if ".ipynb_checkpoints" in relative.parts:
            continue
        if relative.parts[0] in {"runs", "reports"} or path in allowed_guides:
            continue
        raise ValueError(
            f"Performance documentation belongs in generated reports: {path}"
        )
    with tempfile.TemporaryDirectory(prefix="akas-report-check-") as temporary:
        staged_history = Path(temporary) / "benchmarks"
        for folder in sorted(reports.iterdir()):
            if folder.name == ".ipynb_checkpoints":
                continue
            if folder.name == "README.md" and folder.is_file():
                continue
            if not folder.is_dir() or not (folder / "report.json").is_file():
                raise ValueError(f"Unexpected report entry: {folder}")
            if {
                p.name for p in folder.iterdir() if p.name != ".ipynb_checkpoints"
            } != REPORT_FILES:
                raise ValueError(
                    f"Report files differ from the standard bundle: {folder}"
                )
            data = read(folder / "report.json")
            if folder != report_directory(data["campaign"], history):
                raise ValueError(
                    f"Report directory does not match campaign identity: {folder}"
                )
            staged = staged_history / "reports" / folder.name
            render(data, staged, staged_history)
            for name in sorted(REPORT_FILES):
                if (folder / name).read_bytes() != (staged / name).read_bytes():
                    raise ValueError(
                        f"Generated report differs; regenerate it: {folder / name}"
                    )
        index(staged_history)
        if (reports / "README.md").read_bytes() != (
            staged_history / "reports/README.md"
        ).read_bytes():
            raise ValueError(
                "Report index differs; regenerate it with tools.campaign_report"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--check",
        action="store_true",
        help="verify the canonical report archive without writing",
    )
    source.add_argument(
        "--campaign", type=Path, help="completed or interrupted matrix campaign.json"
    )
    source.add_argument(
        "--render",
        type=Path,
        help="portable report.json; compiler/build directories are unnecessary",
    )
    parser.add_argument("--history", type=Path, default=Path("benchmarks"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.check:
        check(args.history)
        print("Benchmark report format and generated content verified.")
        return
    if args.render:
        destination = args.out or args.render.parent
        render(read(args.render), destination, args.history)
    else:
        destination = export(args.campaign, history=args.history, out=args.out)
    print(destination / "README.md")


if __name__ == "__main__":
    main()
