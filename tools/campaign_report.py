"""Export portable campaign tables from retained measurements and baseline history."""

import argparse
from collections import Counter
import csv
import hashlib
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
from akas.benchmarking import archive
from akas.benchmarking import enhancement, transcription
from akas.benchmarking.workload import prompt_digest
from akas.optimization.context import workload

SCHEMA = "akas.campaign-report.v1"
SPEECH_SCHEMA = "akas.speech-report.v1"
NATIVE_STREAM_SCHEMA = "akas.native-stream-baseline-report.v1"
STREAM_COMPARISON_SCHEMA = "akas.native-stream-comparison-report.v1"
STREAM_COMPARISON_FILES = frozenset(("README.md", "measurements.csv", "campaign.json", "report.json"))
SPEECH_FILES = frozenset((
    "README.md", "COMPARISON.md", "AKAS.md", "BASELINES.md",
    "measurements.csv", "campaign.json", "report.json",
))
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
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    streamed = kind == "akas" and protocol == "client-stream-burst-v1"
    qualified = qualified and row.get("model_throughput_qualified") is not False
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
        ttft_ms=median(samples, "ttft_seconds", 1000) if streamed else median(serving, "ttft_ms")
        if kind == "akas"
        else median(samples, "ttft_seconds", 1000),
        tpot_ms=median(samples, "tpot_seconds", 1000) if streamed else median(serving, "tpot_ms")
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
    try:
        data = json.loads(archive.read_file(history, "runs", run_id, "run.json"))
    except FileNotFoundError:
        return None
    for item in data.get("artifacts", []):
        if (
            item["path"] == "report.json"
            and hashlib.sha256(archive.read_file(
                history, "runs", run_id, item["path"])).hexdigest() != item["sha256"]
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
    requested = state["specification"].get("baselines")
    if requested is not None:
        return [
            (item["runtime"], item["provider"], item["weight_format"], item["mode"])
            for item in requested
            if item["weight_format"] in formats and item["mode"] in model["modes"]
        ]
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
            path = archive.directory(history, "runs", item["id"]) / artifact["path"]
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
        path = archive.directory(history, "runs", run_id) / "benchmark/text-workload.json"
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
                           draft_weight_dtype=meta.get("draft_weight_dtype"),
                           split=meta["dataset_split"], kv_dtype=meta["kv_dtype"],
                           draft_kv_dtype=meta["draft_kv_dtype"], sources=workload["sources"],
                           ar_reference_run_id=meta.get("ar_reference_run_id", run_id),
                           qualification=meta["qualification"], cases=selected))
        evidence[run_id] = metadata_for(data)
    return groups


def speculative_baselines(state, history, evidence):
    """Retain upstream acceptance counters separately from native AR comparisons."""
    groups = []
    expected = {item.get("checkpoint_sha256") for item in evidence.values()
                if item.get("checkpoint_sha256")}
    for run_id in state.get("speculative_baseline_runs", []):
        data = record(history, run_id)
        if not data or data["status"] != "passed" or not data.get("completed_at"):
            raise ValueError("speculative baseline requires a sealed, passed run")
        meta = data["metadata"]
        if meta.get("checkpoint_sha256") not in expected:
            raise ValueError("speculative baseline checkpoint differs from campaign")
        artifact = next((a for a in data["artifacts"]
                         if a["path"] == "benchmark/text-workload.json"), None)
        path = archive.directory(history, "runs", run_id) / "benchmark/text-workload.json"
        if (not artifact or digest(path) != artifact["sha256"] or
            artifact["sha256"] != meta.get("text_workload_sha256")):
            raise ValueError("speculative baseline workload checksum mismatch")
        workload = read(path)
        if workload.get("tokenizer_checkpoint_sha256") != meta["checkpoint_sha256"]:
            raise ValueError("speculative baseline tokenizer identity differs")
        cases = {item["id"]: item for item in workload["cases"]}
        selected, seen = [], set()
        for row in data["results"]:
            case = cases.get(row.get("text_case", {}).get("id"))
            if (not case or case["id"] in seen or row["status"] != "ok" or
                prompt_digest([case["token_ids"]]) != case["prompt_sha256"] or
                row.get("prompt_sha256") != case["prompt_sha256"] or
                row["case"]["prompt"] != case["prefix_tokens"] or
                row["case"]["generated"] != workload["output_tokens"] or
                len(row["output_ids"]) != 1 or
                len(row["output_ids"][0]) != workload["output_tokens"]):
                raise ValueError("speculative baseline inputs or outputs differ")
            seen.add(case["id"])
            samples = row["samples"]
            if len(samples) != meta["repeats"]:
                raise ValueError("incomplete speculative baseline repetitions")
            for sample in samples:
                proposed, accepted = sample["proposed"], sample["accepted"]
                counters = sample["counters"]
                if (proposed < 1 or not 0 <= accepted <= proposed or
                    not math.isclose(sample["acceptance"], accepted / proposed) or
                    counters["spec_decode_num_draft_tokens_total"] != proposed or
                    counters["spec_decode_num_accepted_tokens_total"] != accepted or
                    counters["spec_decode_num_drafts_total"] < 1 or
                    len(sample["output_ids"]) != workload["output_tokens"] or
                    any(not math.isfinite(sample[key]) or sample[key] <= 0
                        for key in ("prefill_tokens_per_second", "decode_tokens_per_second"))):
                    raise ValueError("invalid speculative baseline accounting")
            selected.append(dict(
                id=case["id"], category=case["category"], split=case["split"],
                prefix_tokens=case["prefix_tokens"], prompt_sha256=case["prompt_sha256"],
                accepted=sum(s["accepted"] for s in samples),
                proposed=sum(s["proposed"] for s in samples),
                acceptance=sum(s["accepted"] for s in samples) / sum(s["proposed"] for s in samples),
                prefill_tps=median(samples, "prefill_tokens_per_second"),
                decode_tps=median(samples, "decode_tokens_per_second"),
                output_ids=row["output_ids"], output_text=row["output_text"], samples=samples))
        if seen != set(cases):
            raise ValueError("incomplete speculative baseline prompt coverage")
        groups.append(dict(run_id=run_id, runtime=meta["runtime"],
                           runtime_revision=meta["runtime_revision"],
                           confidence_threshold=meta["confidence_threshold"],
                           draft_weight_dtype=meta["draft_weight_dtype"],
                           kv_dtype=meta["kv_dtype"], draft_kv_dtype=meta["draft_kv_dtype"],
                           output_tokens=workload["output_tokens"], warmup=meta["warmup"],
                           repeats=meta["repeats"], qualification=meta["qualification"], cases=selected))
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
                        data["status"] == "passed" and data["metadata"].get("model_throughput_qualified") is not False,
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
        if (selected and selected[0]["status"] == "passed" and entry.get("deployment")
                and selected[0]["metadata"].get("model_throughput_qualified") is not False):
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
            if partial:
                profile["qualifying_status"] = "unqualified"
                profile["kv_dtype"] = next(
                    (data["metadata"].get("kv_dtype") for data, _, _ in reversed(tuple(saved.values()))
                     if data["metadata"].get("kv_dtype")), profile["kv_dtype"])
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
            and candidate["metadata"].get("protocol") in ("baseline-burst-v3", "baseline-stream-burst-v1")
            and (candidate.get("completed_at") or candidate["started_at"])
            <= state["created_at"]
        ):
            selected[key] = item["id"]
    locked = state.get("baseline_lock")
    if locked is not None:
        if locked.get("schema") != "akas.baseline-lock.v1":
            raise ValueError("unsupported baseline lock schema")
        selected = {}
        for runtime, entry in locked["runs"].items():
            run_id = entry["run_id"]
            contents = archive.read_file(history, "runs", run_id, "run.json")
            if hashlib.sha256(contents).hexdigest() != entry["sha256"]:
                raise ValueError("frozen baseline checksum mismatch: " + run_id)
            candidate = record(history, run_id)
            meta = candidate["metadata"]
            shapes = {(r["case"]["batch"], r["case"]["prompt"], r["case"]["generated"])
                      for r in candidate["results"]}
            if (meta["runtime"] != runtime or shapes != expected_shapes
                    or not candidate.get("completed_at")
                    or candidate["completed_at"] > state["created_at"]):
                raise ValueError("frozen baseline is incompatible with campaign: " + run_id)
            for name, checksum in entry.get("artifact_checksums", {}).items():
                if hashlib.sha256(archive.read_file(history, "runs", run_id, name)).hexdigest() != checksum:
                    raise ValueError("frozen baseline artifact checksum mismatch: " + run_id + "/" + name)
            selected[meta["model"], runtime, meta["provider"], meta["weight_format"], meta["mode"]] = run_id
    for model_id, model in state["models"].items():
        for runtime, provider, weight, mode in baseline_combinations(state, model):
            key = (model["model"], runtime, provider, weight, mode)
            data = record(history, selected[key]) if key in selected else None
            if data and (
                data["metadata"].get("protocol") not in ("baseline-burst-v3", "baseline-stream-burst-v1")
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
        speculative_baselines=speculative_baselines(state, history, evidence),
    )


def text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content.rstrip() + "\n" if path.suffix == ".md" else content,
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def fmt(value):
    return "—" if value is None else f"{value:,.1f}"


def clean(value):
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def evidence_path(run_id, out, history, name):
    return (
        archive.link(history, "runs", run_id, name)
        or Path(os.path.relpath(Path(history) / "runs" / run_id / name, out)).as_posix()
    )


def evidence_link(run_id, out, history, *, name="run.json", label=None):
    if not run_id:
        return "—"
    return f"[{label or run_id}]({evidence_path(run_id, out, history, name)})"


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
    if data.get("schema") == STREAM_COMPARISON_SCHEMA:
        return render_native_stream_comparison(data, Path(out).resolve(), Path(history).resolve())
    if data.get("schema") == NATIVE_STREAM_SCHEMA:
        return render_native_stream_baseline(data, Path(out).resolve(), Path(history).resolve())
    if data.get("schema") == SPEECH_SCHEMA:
        if data["workload"].get("protocol") == transcription.PROTOCOL:
            return render_transcription(data, Path(out).resolve(), Path(history).resolve())
        if data["workload"].get("protocol") == enhancement.PROTOCOL:
            return render_enhancement(data, Path(out).resolve(), Path(history).resolve())
        return render_speech(data, Path(out).resolve(), Path(history).resolve())
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
    notes = data.get("measurement_notes", notes)
    lines = [
        "# Campaign benchmark report",
        "",
        f"UTC: {state['created_at']} → {state.get('updated_at', 'running')}. Status: `{state['status']}`. Active time: {state.get('spent_seconds', 0) / 3600:.2f} hours; per-configuration ceiling: {ceiling_text}.",
        "",
        f"Source: `{state['source'].get('revision')}`; measured source tree: `{state['source']['tree_sha256']}`.",
        "",
        f"Qualified AKAS {'measurements' if data.get('measurement_verification') else 'deployments'}: **{len(qualified)}/{len(akas)}**. Qualified baseline profiles: **{sum(p['qualifying_status'] == 'passed' for p in baselines)}/{len(baselines)}**.",
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
    if "performance_acceptance" in data:
        acceptance = data["performance_acceptance"]
        write(out / "PARITY.json", acceptance)
        lines.insert(
            lines.index(notes),
            ("Decode median criterion: **" if data.get("measurement_notes")
             and acceptance.get("phases") == ["decode"] else "Both-phase median criterion: **")
            + ("passed" if acceptance["passed"] else "failed")
            + "**. [Per-prefix acceptance results](PARITY.json).\n",
        )
    if data.get("prefill_observations"):
        prefill = ["## Prefill observations", "",
            "One completed-wall observation per prefix, including first-use phase setup where encountered. These are not warmed medians or repeated prefill benchmarks. The maximum-prefix AKAS observation comes from the original retained prefill producer; only decode gates acceptance.", "",
            "| Prefix tokens | AKAS observed prefill tok/s | llama.cpp observed prefill tok/s | Samples per runtime |",
            "| ---: | ---: | ---: | ---: |"]
        for row in data["prefill_observations"]:
            prefill.append(f"| {row['prefix']:,} | {row['akas']['tokens_per_second']:,.2f} | {row['baseline']['tokens_per_second']:,.2f} | 1 |")
        prefill.append("")
        position = lines.index("## Hardware and measurement provenance")
        lines[position:position] = prefill
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
                  "Warmup and repetition counts are recorded per run; single-sample sweeps are diagnostic. "
                  "Dataset revisions, example IDs, prompt hashes, crop recipes, full methodology and both outputs remain in report.json and the checksummed workload artifacts."]
    for comparison in data.get("text_comparisons", []):
        variant = f"; {comparison['variant']}" if comparison.get("variant") else ""
        draft_weights = (f"Draft weights: `{comparison['draft_weight_dtype']}`. "
                         if comparison.get("draft_weight_dtype") else "")
        lines += ["", f"### {comparison['split']}; confidence {comparison['confidence_threshold']}{variant}", "",
                  draft_weights + f"Target/draft KV: `{comparison['kv_dtype']}` / `{comparison['draft_kv_dtype']}`. "
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
    if data.get("speculative_baselines"):
        lines += ["", "## Upstream speculative acceptance", "",
                  "These runs use the same pinned real-text token IDs and target checkpoint as the native tests. "
                  "Acceptance is accepted draft tokens divided by proposed draft tokens, pooled over measured samples. "
                  "It excludes the target bonus token and depends on confidence filtering. "
                  "Prefill and decode rates use llama.cpp's reported server timing boundaries; cross-runtime speedups are not calculated. "
                  "Different target arithmetic can change generated text and subsequent acceptance. "
                  "Full outputs, per-sample counters, commands, checkpoint hashes and upstream source revision are retained in the linked evidence."]
    for baseline in data.get("speculative_baselines", []):
        lines += ["", f"### {baseline['runtime']}; draft {baseline['draft_weight_dtype']}; confidence {baseline['confidence_threshold']}", "",
                  f"Target/draft KV: `{baseline['kv_dtype']}` / `{baseline['draft_kv_dtype']}`. "
                  f"Output tokens: {baseline['output_tokens']}; warmups: {baseline['warmup']}; repetitions: {baseline['repeats']}. "
                  f"Upstream revision: `{baseline['runtime_revision']}`. "
                  f"Evidence: {evidence_link(baseline['run_id'], out, history)}.", "",
                  "| Example | Split | Category | Prefix | Accepted / proposed | Acceptance | Prefill tok/s | Decode tok/s |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in baseline["cases"]:
            lines.append(f"| {clean(row['id'])} | {row['split']} | {row['category']} | {row['prefix_tokens']} | "
                         f"{row['accepted']} / {row['proposed']} | {row['acceptance']:.1%} | "
                         f"{row['prefill_tps']:.2f} | {row['decode_tps']:.2f} |")
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
                    "record": archive.link(history, "runs", value["run_id"], "run.json") or os.path.relpath(
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
        "Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings. † marks an incomplete correctness qualification or a failed correctness check; the reason is stated below. Timing samples are retained. Missing values are not replaced by zero.",
        "",
    ]
    if data.get('qualification_notes'):
        lines += ['Qualification notes:', '']
        lines += ['- '+note for note in data['qualification_notes']]
        lines.append('')
    profiles = data["profiles"]
    spec = data["campaign"]["specification"]
    requested = data["workload"]

    def transport_label(profile):
        run_id = profile.get('selected_run_id') or next(
            (row.get('run_id') for row in profile['measurements'] if row.get('run_id')), None)
        meta = data.get('evidence', {}).get(run_id, {}).get('metadata', {})
        transport = str(meta.get('transport', '')).lower()
        if 'http' in transport:
            return 'HTTP/SSE'
        if 'gpu' in transport and 'phase' in transport:
            return 'CUDA events'
        if meta.get('protocol') == 'client-stream-burst-v1':
            return 'native engine token stream'
        return 'reported transport'
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
                            transport_label(p)
                            if metric == "burst_output_tok_s"
                            else next(
                                (
                                    r["phase_protocol"]
                                    for r in p["measurements"]
                                    if r.get("phase_protocol")
                                ),
                                "unmeasured",
                            ) + ' / ' + transport_label(p)
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
    if data.get('efficiency_estimates'):
        efficiency = data['efficiency_estimates']
        lines += ['## Utilization estimates and targets', '', efficiency['notes'], '',
                  '| C | vLLM MFU / MBU | SGLang MFU / MBU | AKAS MFU / MBU |',
                  '| ---: | ---: | ---: | ---: |']
        for batch in requested['batches']:
            cells = []
            for runtime in ('vllm','sglang','akas'):
                row = next((r for r in efficiency['rows'] if r['runtime']==runtime and r['concurrency']==batch), None)
                cells.append(f"{row['prefill_mfu_percent']:.1f}% / {row['decode_mbu_percent']:.1f}%" if row else '—')
            lines.append(f"| {batch} | " + ' | '.join(cells) + ' |')
        lines += ['', efficiency['target_status'], '']
    if data.get("calibration"):
        pilot = data["calibration"]
        lines += ["## Native prefill tuning pilots", "",
                  f"Reference: {evidence_link(pilot['reference_run_id'], out, history)}; "
                  f"selected pilot: {evidence_link(pilot['selected_run_id'], out, history)}. "
                  f"Chunks: {pilot['before_chunk']} → {pilot['after_chunk']}. "
                  f"Each uses {pilot['warmup']} warmup and {pilot['repeats']} retained repetitions. "
                  "These calibration prefixes use identical weights, inputs, cache precision and native timing protocol. "
                  "The speedup below compares native prefill within AKAS; it does not compare runtimes.", "",
                  "| Prefix | Before prefill | After prefill | Prefill speedup | Before decode | After decode | Exact output IDs |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for row in pilot["cases"]:
            lines.append(f"| {row['prefix']} | {fmt(row['before_prefill'])} | {fmt(row['after_prefill'])} | "
                         f"{row['prefill_speedup']:.3f}x | {fmt(row['before_decode'])} | {fmt(row['after_decode'])} | "
                         f"{'yes' if row['exact_output'] else 'no'} |")
        lines.append("")
    if data.get("tuning_evidence"):
        tuning = data["tuning_evidence"]
        lines += ["## Tuning evidence", "",
                  f"{evidence_link(tuning['run_id'], out, history)}: {clean(tuning['scope'])}", ""]
    if data.get("retuning"):
        tuned = data["retuning"]
        lines += ["## Previous AKAS sweep", "",
                  f"Previous: {evidence_link(tuned['reference_run_id'], out, history)}; "
                  f"selected: {evidence_link(tuned['selected_run_id'], out, history)}. "
                  "Weights, input IDs, cache precision, shape, repetition counts and native timing protocol match. "
                  "These ratios compare AKAS with its own previous sweep, not with another runtime.", "",
                  "| Prefix | Before prefill | After prefill | Prefill ratio | Before decode | After decode | Decode ratio | Exact output IDs |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for row in tuned["cases"]:
            lines.append(f"| {row['prefix']} | {fmt(row['before_prefill'])} | {fmt(row['after_prefill'])} | "
                         f"{row['after_prefill'] / row['before_prefill']:.3f}x | {fmt(row['before_decode'])} | "
                         f"{fmt(row['after_decode'])} | {row['after_decode'] / row['before_decode']:.3f}x | "
                         f"{'yes' if row['exact_output'] else 'no'} |")
        lines.append("")
    if data.get("output_comparisons"):
        lines += ["## Greedy output agreement", "",
                  "The untimed AKAS greedy pass is compared with the first retained llama.cpp response. "
                  "Both use the same input IDs, checkpoint and cache precision. Different arithmetic can "
                  "change the trajectory; agreement is not a corpus accuracy score.", "",
                  "| Prefix | Output tokens | Exact IDs | Common prefix tokens |",
                  "| ---: | ---: | --- | ---: |"]
        for row in data["output_comparisons"]:
            lines.append(f"| {row['prefix']} | {row['output']} | {'yes' if row['exact'] else 'no'} | {row['common_prefix_tokens']} |")
        lines.append("")
    text(out / "COMPARISON.md", "\n".join(lines) + "\n")


def index(history):
    history = Path(history)
    reports = history / "reports"
    lines = [
        "# Campaign reports",
        "",
        "Each report contains the full requested matrix, all available AKAS and baseline measurements, failed or unsupported rows, runtime settings, source provenance and selected kernel schedules. Archived reports and runs are pinned by commit in [the artifact catalog](../archive.json). Native streaming baseline reports contain native diagnostics only.",
        "",
        "| Started UTC | Workload | Source | Qualified AKAS | Qualified baselines | Report |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    archived = archive.catalog(history)
    entries = {identity: (entry["summary"], archive.url(archived, entry, "README.md"))
               for identity, entry in archived["reports"].items()}
    from tools.benchmark_archive import report_summary
    for path in reports.glob("*/report.json"):
        if path.parent.name in entries:
            continue
        data = read(path)
        entries[path.parent.name] = (report_summary(data), path.parent.name + "/README.md")
    for identity in sorted(entries, reverse=True):
        summary, target = entries[identity]
        lines.append(
            f"| {summary['started_at']} | {summary['workload']} | `{summary['revision']}` | {summary['akas']} | {summary['baseline']} | [full tables]({target}) |"
        )
    text(reports / "README.md", "\n".join(lines) + "\n")


def export(campaign, *, history="benchmarks", out=None):
    data = collect(campaign, history)
    destination = Path(out) if out else report_directory(data["campaign"], history)
    render(data, destination, history)
    return destination


def export_native_stream_baseline(run_ids, history, out):
    """Render native BF16 streaming diagnostics and their archived source dependencies."""
    runs = [record(history, run_id) for run_id in run_ids]
    if any(data is None or data["status"] != "passed" for data in runs):
        raise ValueError("native stream baseline requires sealed, passed runs")
    measurements = sorted(
        ({**row, "run_id": data["id"]} for data in runs for row in data["results"]),
        key=lambda row: (row["case"]["concurrency"], row["case"]["id"]),
    )
    if not measurements:
        raise ValueError("native stream baseline has no measurements")
    for data in runs:
        meta = data["metadata"]
        if (
            meta.get("protocol") != "client-speech-pcm-stream-v1"
            or meta.get("weight_format") != "bf16"
        ):
            raise ValueError("invalid native stream baseline metadata: " + data["id"])
        workload = meta["speech_workload"]["definition"]
        if workload["request"]["max_tokens"] != 384 or workload["request"]["stream_decode"] != "prefix":
            raise ValueError("native stream baseline protocol differs")
        for row in data["results"]:
            if (
                row["case"]["concurrency"] != row["requests"]
                or row["completed"] != row["requests"]
                or row["failed"]
                or len(row["results"]) != row["requests"]
                or not all(item["ok"] for item in row["results"])
            ):
                raise ValueError("invalid native stream baseline measurement: " + data["id"])
    identity = hashlib.sha256("\n".join(run_ids).encode()).hexdigest()
    state = dict(
        identity=identity,
        created_at=min(data["started_at"] for data in runs),
        updated_at=max(data["completed_at"] for data in runs),
        run_ids=list(run_ids),
        codec_qualification_run=None,
        codec_qualification=None,
        compact_integration_run=None,
        source_provenance_run=None,
    )
    data = dict(
        schema=NATIVE_STREAM_SCHEMA,
        generated_at=datetime.now(timezone.utc).isoformat(),
        campaign=state,
        workload=runs[0]["metadata"]["speech_workload"]["definition"],
        measurements=measurements,
        evidence={data["id"]: metadata_for(data) for data in runs},
    )
    destination = Path(out) if out else report_directory(state, history)
    render_native_stream_baseline(data, destination, history)
    return destination


def attach_native_stream_codec(report, qualification_run_id, compact_run_id, history):
    """Link packaging/integration evidence without rewriting the original baseline matrix."""
    data = read(Path(report) / "report.json")
    if data.get("schema") != NATIVE_STREAM_SCHEMA:
        raise ValueError("not a native stream baseline report")
    qualification = record(history, qualification_run_id)
    integration = record(history, compact_run_id)
    if not qualification or qualification["status"] != "passed" or not integration or integration["status"] != "passed":
        raise ValueError("compact codec evidence must be sealed and passed")
    if not qualification["metadata"].get("checks") or not all(
        check["bitwise"] for check in qualification["metadata"]["checks"]
    ):
        raise ValueError("compact codec requires bitwise qualification")
    if not all(row["status"] == "ok" for row in integration["results"]):
        raise ValueError("compact integration has failed requests")
    data["campaign"]["codec_qualification_run"] = qualification_run_id
    data["campaign"]["compact_integration_run"] = compact_run_id
    data["campaign"]["codec_qualification"] = qualification["metadata"]["compact_codec"]
    data["evidence"][qualification_run_id] = metadata_for(qualification)
    data["evidence"][compact_run_id] = metadata_for(integration)
    provenance_id = data["campaign"].get("source_provenance_run")
    if provenance_id:
        data["evidence"][provenance_id] = metadata_for(record(history, provenance_id))
    render_native_stream_baseline(data, Path(report).resolve(), Path(history).resolve())


def attach_native_stream_provenance(report, provenance_run_id, history):
    """Link the actual baseline source without changing the original measurements."""
    data = read(Path(report) / "report.json")
    if data.get("schema") != NATIVE_STREAM_SCHEMA:
        raise ValueError("not a native stream baseline report")
    provenance = record(history, provenance_run_id)
    if not provenance or provenance["status"] != "passed":
        raise ValueError("baseline source provenance must be sealed and passed")
    data["campaign"]["source_provenance_run"] = provenance_run_id
    data["evidence"][provenance_run_id] = metadata_for(provenance)
    render_native_stream_baseline(data, Path(report).resolve(), Path(history).resolve())


def render_native_stream_baseline(data, out, history):
    if data.get("schema") != NATIVE_STREAM_SCHEMA:
        raise ValueError("invalid native stream baseline schema")
    out.mkdir(parents=True, exist_ok=True)
    state = data["campaign"]
    measurements = sorted(data["measurements"], key=lambda row: row["case"]["concurrency"])
    workload = data["workload"]
    prompt = workload["utterances"][0]["text"]
    if len(prompt.encode()) != 177 or len(prompt.split()) != 30:
        raise ValueError("native stream baseline input differs")
    if any(row["case"]["text"] != prompt for row in measurements):
        raise ValueError("native stream baseline prompt differs between runs")
    qualification_id = state.get("codec_qualification_run")
    integration_id = state.get("compact_integration_run")
    lines = [
        "# Native BF16 streaming baseline",
        "",
        "These are Zyphra native CUDA **baseline diagnostics**, not qualified AKAS measurements. "
        "The GGUF/llama.cpp path is not part of this comparison.",
        "",
        f"Model: **{measurements[0]['case'].get('model', 'Zyphra/ZONOS3-Mini')}**; weights: **BF16 LM** with the FP32 "
        "`keep2_plain` codec. Fixed English input: **30 words / 177 UTF-8 bytes**. "
        "Each request uses **50 prompt frames** (47 model-input frames + 3 silence-prefill frames), "
        "up to **384 generated tokens** within a **512-frame** limit.",
        "",
        "Flat admission up to the requested request limit; no request-class bucketing. Prefix streaming is enabled. "
        "The scheduler fix prevents request-slot recycling while earlier decodes remain active. "
        "C64 had the highest measured audio throughput; C256 is a successful capacity result, not a latency target.",
        "",
        "| Concurrency | Valid audio | Wall time (s) | Audio seconds/s | Peak GPU (GiB) | Evidence |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in measurements:
        peak = row["gpu_memory_peak"]["memory_used_mib"] / 1024
        lines.append(
            f"| {row['concurrency']} | {row['completed']}/{row['requests']} | {row['wall_s']:.2f} | "
            f"{row['audio_seconds_per_second']:.2f} | {peak:.2f} | {evidence_link(row['run_id'], out, history)} |"
        )
    lines += [
        "",
        "## Source and weight packaging",
        "",
        "For the native baseline, the full training generator can be replaced by a **compact, dtype-preserving "
        "codec checkpoint** with identical output. The matrix used the original assets; the separately qualified "
        "compact package is linked below. The LM remains BF16; no weights are quantized for size. "
        "The local scheduler and loading changes "
        "are retained in each run's **`dependencies/<dependency>/source.tar.gz`** artifact. "
        "Inspect the linked run's `run.json` and dependency snapshot for the exact source used; do not use a mutable "
        "checkout as provenance.",
        "",
    ]
    if qualification_id:
        codec = state["codec_qualification"]
        lines += [
            f"Codec packaging: **{codec['bytes'] / 1e9:.3f} GB**, replacing **8.48 GB** of full generator + override "
            f"assets. With the 1.772 GB BF16 LM, the required native model data is approximately **2.35 GB** "
            "instead of **10.25 GB**.",
            "",
            f"Bitwise codec qualification: {evidence_link(qualification_id, out, history, name='qualification.json', label='1-, 7-, and 32-frame identical-output checks')}. "
            f"Compact C1/C16 HTTP integration: {evidence_link(integration_id, out, history)}.",
            "",
            "The compact checkpoint remains a local build artifact; it is intentionally not uploaded to external storage. "
            "Its size, format and SHA-256 are retained in the qualification record.",
            "",
        ]
    else:
        lines += [
            "The original baseline matrix used the full native codec assets. A separately qualified compact-codec "
            "package may be linked here after validation.",
            "",
        ]
    if state.get("source_provenance_run"):
        lines += [
            "Exact baseline source bundle: "
            + evidence_link(
                state["source_provenance_run"],
                out,
                history,
                name="dependencies/zonos3/source.tar.gz",
                label="dependencies/zonos3/source.tar.gz",
            )
            + ". This supplemental bundle includes `model/` and `servers/` and is the authoritative source for "
            "the compact codec loader and safe slot-lifecycle scheduler change.",
            "",
        ]
    lines += [
        "Run artifacts, source snapshots, exact dependency revisions, per-request audio hashes and server logs are "
        "retained under the normal benchmark archive. These measurements establish native baseline functionality "
        "and capacity; they do not claim AKAS BF16 correctness or qualified speedups.",
        "",
    ]
    text(out / "README.md", "\n".join(lines))
    text(
        out / "COMPARISON.md",
        "# Native baseline concurrency comparison\n\n"
        + "\n".join(line for line in lines if line.startswith("| ") or line.startswith("# Native")),
    )
    text(
        out / "AKAS.md",
        "# AKAS status\n\nNo AKAS BF16 implementation has been measured in this baseline report. "
        "The matrix contains native Zyphra CUDA diagnostics only.\n",
    )
    text(
        out / "BASELINES.md",
        "# Native baseline provenance\n\n"
        "Each linked run retains its exact source and dependency snapshots under "
        "`dependencies/<dependency>/source.tar.gz`, plus per-request audio hashes and server logs. "
        "Use those artifacts, not the mutable checkout, when inspecting the baseline source.\n",
    )
    csv_file(
        out / "measurements.csv",
        measurements,
        (
            "run_id",
            "concurrency",
            "requests",
            "completed",
            "failed",
            "wall_s",
            "throughput_qps",
            "audio_seconds_per_second",
            "gpu_memory_initial",
            "gpu_memory_peak",
        ),
    )
    write(out / "report.json", data)
    write(out / "campaign.json", state)
    index(history)
    return out


def export_native_stream_comparison(akas_run, native_report, correctness_run, figure_run, history, out=None):
    """Export the matched BF16 streaming comparison from sealed archive records."""
    from tools.plot_comparison import _native_stream_points

    akas, correctness, figure = (record(history, run) for run in (akas_run, correctness_run, figure_run))
    if any(run is None or run["status"] != "passed" for run in (akas, correctness, figure)):
        raise ValueError("streaming comparison requires three sealed, passed runs")
    native_raw = archive.read_file(history, "reports", native_report, "report.json")
    native = json.loads(native_raw)
    for key in ("akas", "native", "correctness"):
        source = figure["metadata"]["inputs"][key]
        raw = archive.read_file(history, source["kind"], source["id"], "report.json")
        if source["id"] != {"akas": akas_run, "native": native_report, "correctness": correctness_run}[key] or (
            hashlib.sha256(raw).hexdigest() != source["sha256"]
        ):
            raise ValueError(f"figure inputs do not match the comparison: {key}")
    if (
        akas["metadata"]["program_sha256"] != correctness["metadata"]["program_sha256"]
        or akas["source"]["tree_sha256"] != correctness["source"]["tree_sha256"]
    ):
        raise ValueError("correctness evidence belongs to a different program or source tree")
    points = _native_stream_points({
        "schema": "akas.benchmark.v1", "metadata": akas["metadata"], "results": akas["results"],
    }, native)
    rows = []
    for count, a, n in points:
        row = {"concurrency": count}
        for label, value in (("akas", a), ("native", n)):
            row.update({
                f"{label}_completed": value["completed"],
                f"{label}_wall_s": value["wall_s"],
                f"{label}_audio_s_per_s": value["audio_seconds_per_second"],
                f"{label}_ttfa_p50_ms": 1000 * value["ttfa_s"]["p50"],
                f"{label}_peak_gpu_gib": value["gpu_memory_peak"]["memory_used_mib"] / 1024,
            })
        row["throughput_speedup"] = row["akas_audio_s_per_s"] / row["native_audio_s_per_s"]
        rows.append(row)
    checks = correctness["metadata"]["checks"]
    replay_shapes = {r["case"]["batch"] for r in correctness["results"] if r["slot_reuse_bitwise"]}
    if not checks or replay_shapes != {row["concurrency"] for row in rows}:
        raise ValueError("streaming correctness evidence does not cover the measured matrix")
    state = dict(
        identity=hashlib.sha256("\n".join((akas_run, native_report, correctness_run, figure_run)).encode()).hexdigest(),
        created_at=akas["started_at"], updated_at=figure["completed_at"],
        akas_run=akas_run, native_report=native_report, correctness_run=correctness_run, figure_run=figure_run,
        source={"revision": akas["source"]["revision"], "tree_sha256": akas["source"]["tree_sha256"]},
    )
    data = dict(
        schema=STREAM_COMPARISON_SCHEMA, campaign=state, workload=native["workload"], measurements=rows,
        correctness=dict(codec_checks=len(checks), max_rms=max(c["rms"] for c in checks),
                         max_abs=max(c["max_abs"] for c in checks),
                         replays=sum(r["completed"] for r in correctness["results"])),
        program_sha256=akas["metadata"]["program_sha256"],
    )
    return render(data, Path(out) if out else report_directory(state, history), history)


def render_native_stream_comparison(data, out, history):
    out.mkdir(parents=True, exist_ok=True)
    state, rows, correctness = (data[key] for key in ("campaign", "measurements", "correctness"))
    native = archive.link(history, "reports", state["native_report"], "README.md")
    native = native or f"../{state['native_report']}/README.md"
    akas = evidence_link(state["akas_run"], out, history, name="report.json", label="AKAS measurements")
    checks = evidence_link(state["correctness_run"], out, history, name="report.json", label="Correctness probes")
    plot = evidence_link(state["figure_run"], out, history, name="figures/throughput.svg", label="Three-panel figure")
    best = max(rows, key=lambda row: row["native_audio_s_per_s"])
    largest = max(rows, key=lambda row: row["concurrency"])
    ratio = largest["akas_audio_s_per_s"] / best["native_audio_s_per_s"]
    lines = [
        "# ZONOS3-Mini: native BF16 streaming on NVIDIA L40S",
        "",
        "Native BF16 LM weights, FP32 `keep2_plain`, and 44.1 kHz float32 PCM on CUDA. "
        "Both runtimes measured one HTTP prefix-streaming burst at each concurrency with the fixed "
        "30-word (177-byte) English prompt, 50 prompt frames, and `max_tokens=384`.",
        "",
        f"{plot} · {akas} · [Zyphra native baseline]({native}) · {checks}",
        "",
        "| Concurrency | Valid AKAS/native | AKAS wall s | Native wall s | AKAS audio s/s | Native audio s/s | Speedup | AKAS/native median TTFA (ms) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['concurrency']} | {row['akas_completed']}/{row['native_completed']} | "
            f"{row['akas_wall_s']:.2f} | {row['native_wall_s']:.2f} | {row['akas_audio_s_per_s']:.2f} | "
            f"{row['native_audio_s_per_s']:.2f} | {row['throughput_speedup']:.2f}× | "
            f"{row['akas_ttfa_p50_ms']:.0f}/{row['native_ttfa_p50_ms']:.0f} |"
        )
    lines.extend((
        "",
        "Throughput is aggregate generated audio duration divided by burst wall time. "
        "TTFA is client time to first audio. Compilation and model loading are excluded.",
        "",
        f"AKAS at C{largest['concurrency']} delivered **{ratio:.2f}×** Zyphra's best measured throughput "
        f"(C{best['concurrency']}). Zyphra's C256 prefill budget serialized admission into waves; "
        "the matched C256 speedup includes the effect of that scheduling limit.",
        "",
        "## Correctness and provenance",
        "",
        f"The measured CUDA bundle passed {correctness['codec_checks']} codec probes against Zyphra's "
        f"`ISTFTStream` (worst RMS {correctness['max_rms']:.2e}, max absolute {correctness['max_abs']:.2e}). "
        f"Greedy and seeded sampled requests reproduced token rows and PCM bitwise on slot reuse across "
        f"{correctness['replays']} replayed utterances. These are workload checks, not independent "
        "end-to-end equivalence or an intelligibility assessment.",
        "",
        f"CUDA bundle SHA-256: `{data['program_sha256']}`. The linked run bundles preserve source snapshots, "
        "dependency revisions, and phase manifests. The compact codec checkpoint is not part of the archive.",
        "",
    ))
    text(out / "README.md", "\n".join(lines))
    csv_file(out / "measurements.csv", rows, tuple(rows[0]))
    write(out / "report.json", data)
    write(out / "campaign.json", state)
    index(history)
    return out


def enhancement_memory(history, run_id, native):
    """Import a sealed, bitwise-checked memory A/B on the measured binary."""
    retained = record(history, run_id)
    if retained is None or retained["status"] != "passed":
        raise ValueError("enhancement memory run must be sealed and passed")
    raw = archive.read_file(history, "runs", run_id, "report.json")
    checksum = hashlib.sha256(raw).hexdigest()
    evidence = json.loads(raw)["metadata"]
    if (
        checksum != next(a["sha256"] for a in retained["artifacts"] if a["path"] == "report.json")
        or evidence != retained["metadata"]
        or not evidence.get("checks")
        or not all(check["bitwise"] for check in evidence["checks"])
    ):
        raise ValueError("enhancement memory run lacks sealed bitwise checks")
    profiles = {row["label"]: row for row in evidence["profiles"]}
    if set(profiles) != {"separate", "shared"}:
        raise ValueError("expected separate and shared scratch memory profiles")
    measured = native["metadata"]
    rows = {}
    for label, profile in profiles.items():
        model = profile["model"]
        sample = profile["memory_peak_by_stage"]["golden-11s"]
        shared = model["buffer_bytes"]["shared_scratch"]
        if (
            profile["program_sha256"] != measured["program_sha256"]
            or model["device"]["adapter"] != measured["device"]["adapter"]
            or not profile["qualified"]
            or bool(shared) != (label == "shared")
            or (shared and model["buffer_bytes"] != measured["buffer_bytes"])
            or not sample["gpu_samples"]
        ):
            raise ValueError("enhancement memory profile differs from the measured program")
        rows[label] = dict(
            dedicated_gpu_bytes=sample["dedicated_gpu_bytes"],
            host_working_set_bytes=sample["host_working_set_bytes"],
            buffer_bytes=model["buffer_bytes"],
            checks=profile["checks"],
        )
    return dict(run_id=run_id, report_sha256=checksum, profiles=rows)


def export_speech(native, model, workload_id, history, out, baseline_lock=None, *,
                  memory_run=None):
    from akas.benchmarking.workload import load

    metadata = native["metadata"]
    workload = load(workload_id)
    if (
        metadata["model"] != model["model"]
        or metadata["speech_workload"]["definition"] != workload
        or not native["completed_at"]
        or native["status"] == "invalidated"
    ):
        raise ValueError(
            "the retained speech run does not match the requested workload"
        )
    reference = metadata.get("reference")
    vulkan = record(history, reference["run_id"]) if reference else None
    if reference and vulkan is None:
        raise ValueError("the frozen speech baseline is unavailable")
    if workload["protocol"] == enhancement.PROTOCOL and reference:
        raw, _ = enhancement.reference_report(
            history, reference["run_id"], workload, metadata["checkpoint_sha256"],
            golden_only=True,
        )
        if hashlib.sha256(raw).hexdigest() != reference["report_sha256"]:
            raise ValueError("the frozen enhancement baseline checksum differs")
    cpu = None
    if baseline_lock is not None:
        if workload["protocol"] == transcription.PROTOCOL:
            raise ValueError("ASR export uses the Vulkan reference frozen in the AKAS run")
        if (
            not vulkan
            or baseline_lock.get("schema") != "akas.baseline-lock.v1"
            or set(baseline_lock["runs"]) != {"llama"}
        ):
            raise ValueError(
                "a supplemental speech lock must name one llama.cpp CPU run"
            )
        entry = baseline_lock["runs"]["llama"]
        run_id = entry["run_id"]
        for name, checksum in {
            "run.json": entry["sha256"],
            **entry.get("artifact_checksums", {}),
        }.items():
            if (
                hashlib.sha256(
                    archive.read_file(history, "runs", run_id, name)
                ).hexdigest()
                != checksum
            ):
                raise ValueError(f"frozen speech CPU checksum differs: {run_id}/{name}")
        cpu = record(history, run_id)
        cpu_meta = cpu["metadata"]
        gpu_meta = vulkan["metadata"]
        matched = (
            "runtime",
            "binary_sha256",
            "runtime_revision",
            "runtime_files",
            "model",
            "protocol",
            "speech_workload",
            "weight_format",
            "checkpoint_sha256",
            "voice",
            "sample_rate_hz",
            "warmup",
            "repeats",
        )
        if (
            not cpu["completed_at"]
            or cpu["status"] not in ("passed", "failed")
            or cpu_meta["provider"] != "cpu"
            or any(cpu_meta[key] != gpu_meta[key] for key in matched)
            or cpu_meta["projector"]["sha256"] != gpu_meta["projector"]["sha256"]
            or [row["case"] for row in cpu["results"]] != workload["utterances"]
        ):
            raise ValueError(
                "the frozen CPU run does not match the Vulkan speech workload"
            )
    profiles = []
    for run in (native, vulkan, cpu):
        if run is None:
            continue
        kind = "akas" if run is native else "baseline"
        evidence = json.loads(
            archive.read_file(history, "runs", run["id"], "report.json")
        )
        meta = evidence["metadata"]
        if meta != run["metadata"] or evidence["results"] != run["results"]:
            raise ValueError("speech run record differs from its sealed report")
        checksum = next(
            item["sha256"] for item in run["artifacts"] if item["path"] == "report.json"
        )
        if run is vulkan and checksum != reference["report_sha256"]:
            raise ValueError("the frozen speech baseline checksum differs")
        profiles.append(
            dict(
                id="llama-cpu" if run is cpu else meta["runtime"],
                kind=kind,
                status=run["status"],
                qualifying_status="unqualified",
                metadata=meta,
                source=run["source"],
                run_id=run["id"],
                report_sha256=checksum,
                measurements=evidence["results"],
            )
        )
    state = dict(
        source=native["source"],
        created_at=native["started_at"],
        updated_at=native["completed_at"],
        status=native["status"],
        runs={profile["run_id"]: profile["report_sha256"] for profile in profiles},
    )
    if cpu is not None:
        state.update(
            updated_at=max(state["updated_at"], cpu["completed_at"]),
            baseline_lock=baseline_lock,
        )
    if memory_run:
        if workload["protocol"] != enhancement.PROTOCOL:
            raise ValueError("--memory-run applies only to voice enhancement")
        state["memory"] = enhancement_memory(history, memory_run, native)
    state["identity"] = hashlib.sha256(
        json.dumps(state, sort_keys=True).encode()
    ).hexdigest()
    data = dict(
        schema=SPEECH_SCHEMA, campaign=state, workload=workload, profiles=profiles
    )
    destination = Path(out) if out else report_directory(state, history)
    render(data, destination, history)
    return destination


def render_transcription(data, out, history):
    state, workload, profiles = data["campaign"], data["workload"], data["profiles"]
    transcription.validate(workload)
    if [p["kind"] for p in profiles] not in (["akas"], ["akas", "baseline"]):
        raise ValueError("invalid ASR profiles")
    cases = {item["id"]: item for item in workload["utterances"]}
    banded = all(item.get("duration_band") for item in cases.values())
    nav = "[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)"
    notes = (
        "Warm wall time covers the request through receipt of the complete transcript over local process pipes. "
        "The timer includes WAV loading, feature extraction and greedy decoding. Model loading and context reset are outside the timer. "
        "RTF is wall seconds per second of input audio. WER and CER use case-folded NFKC text with punctuation removed. "
        + (
            f"The clips are recorded speech from {clean(workload['source']['name'])}. "
            if workload.get("source", {}).get("kind") == "recorded"
            else "The demonstration uses synthesized speech. "
        )
        + "Both profiles are unqualified for independent full-model accuracy."
    )
    overview = [
        "# Qwen3 ASR benchmark",
        "",
        nav,
        "",
        f"Workload `{workload['id']}` uses {len(cases)} fixed clips. Protocol `{transcription.PROTOCOL}`.",
        "",
        notes,
        "",
        "| Runtime | GPU | Complete and deterministic cases | Run |",
        "| --- | --- | ---: | --- |",
    ]
    stats = {}
    samples = []
    baseline_page = ["# Baseline transcription measurements", "", nav, ""]
    for profile in profiles:
        meta = profile["metadata"]
        if (
            meta["speech_workload"]["definition"] != workload
            or meta["protocol"] != transcription.PROTOCOL
            or meta["mode"] != "asr"
        ):
            raise ValueError("ASR profile metadata differs from the workload")
        kind = profile["kind"]
        name = "AKAS" if kind == "akas" else "llama.cpp"
        link = evidence_link(profile["run_id"], out, history)
        rows = profile["measurements"]
        indexed = {row["case"]["id"]: row for row in rows}
        if len(indexed) != len(rows) or any(
            cases.get(key) != row["case"] for key, row in indexed.items()
        ):
            raise ValueError("undeclared or duplicate ASR measurements")
        stats[kind] = indexed
        passed = sum(row["status"] == "ok" for row in rows)
        overview.append(
            f"| {name} ({meta['provider']}) | {clean(meta['device']['name'])} | {passed}/{len(cases)} | {link} |"
        )
        lines = [
            f"# {name} transcription measurements",
            "",
            nav,
            "",
            notes,
            "",
            f"Run {link}. Report SHA-256 `{profile['report_sha256']}`.",
            f"Model {meta['model']}. Format `{meta['weight_format']}`. Context {meta['context']} tokens.",
            f"Decoder SHA-256 `{meta['checkpoint_sha256']}`. Projector SHA-256 `{meta['projector']['sha256']}`.",
            f"Each clip has {meta['warmup']} warmup and {meta['repeats']} measured requests.",
        ]
        runtime = meta.get("runtime_version")
        program = meta.get("program_sha256")
        if runtime:
            lines.append(f"Runtime `{clean(runtime)}`.")
        if program:
            lines.append(f"Compiled program SHA-256 `{program}`.")
        lines.extend(
            [
                "",
                "| Case | Status | Wall ms | Audio s | RTF | Error rate | Transcript |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for case, item in cases.items():
            row = indexed.get(case, {})
            summary = row.get("summary", {})
            retained = row.get("samples", [])
            rate = summary.get("error_rate")
            display_rate = (
                "—" if rate is None else f"{item['metric'].upper()} {100 * rate:.1f}%"
            )
            display_text = (
                retained[0]["transcript"] if retained else row.get("error", "—")
            )
            wall = summary.get("wall_seconds")
            rtf = summary.get("real_time_factor")
            display_rtf = "—" if rtf is None else f"{rtf:.3f}"
            lines.append(
                f"| {case} | {row.get('status', 'not measured')} | "
                f"{fmt(wall * 1000 if wall is not None else None)} | {fmt(summary.get('audio_seconds'))} | "
                f"{display_rtf} | {display_rate} | {clean(display_text)} |"
            )
            for sample in retained or ({},):
                samples.append(
                    dict(
                        runtime=name,
                        provider=meta["provider"],
                        case=case,
                        language=item["language"],
                        reference=item["text"],
                        metric=item["metric"],
                        status=row.get("status", "not measured"),
                        error=row.get("error"),
                        **sample,
                    )
                )
        if kind == "akas":
            text(out / "AKAS.md", "\n".join(lines) + "\n")
        else:
            baseline_page = lines
    comparison = [
        "# Transcription comparison",
        "",
        nav,
        "",
        notes,
        "",
        "| Case | "
        + ("Duration band | Audio s | " if banded else "")
        + "AKAS wall ms | llama.cpp wall ms | llama.cpp / AKAS | Exact transcript match |",
        "| --- | "
        + ("--- | ---: | " if banded else "")
        + "---: | ---: | ---: | --- |",
    ]
    ratios = []
    bands = {}
    for case, item in cases.items():
        left, right = (
            stats.get(kind, {}).get(case, {}) for kind in ("akas", "baseline")
        )
        native = left.get("summary", {}).get("wall_seconds")
        baseline = right.get("summary", {}).get("wall_seconds")
        duration = left.get("summary", {}).get("audio_seconds") or right.get(
            "summary", {}
        ).get("audio_seconds")
        rows = bands.setdefault(item["duration_band"], []) if banded else None
        ratio, match = None, "—"
        if left.get("status") == right.get("status") == "ok":
            ratio = baseline / native
            ratios.append(ratio)
            match = (
                "yes"
                if left["samples"][0]["transcript"] == right["samples"][0]["transcript"]
                else "no"
            )
        display_ratio = "—" if ratio is None else f"{ratio:.3f}"
        duration_cells = (
            f"{clean(item['duration_band'])} | {fmt(duration)} | " if banded else ""
        )
        if rows is not None:
            rows.append((duration, native, baseline, ratio))
        comparison.append(
            f"| {case} | {duration_cells}{fmt(native * 1000 if native is not None else None)} | "
            f"{fmt(baseline * 1000 if baseline is not None else None)} | {display_ratio} | {match} |"
        )
    if bands:
        comparison.extend(
            (
                "",
                "Band medians use clips completed deterministically by both runtimes. "
                "Ratios are geometric means of the paired per-clip ratios.",
                "",
                "| Duration band | Completed clips | Audio s | AKAS wall ms | llama.cpp wall ms | llama.cpp / AKAS |",
                "| --- | ---: | --- | ---: | ---: | ---: |",
            )
        )
        for band, rows in bands.items():
            paired = [(a, b, ratio) for _, a, b, ratio in rows if ratio is not None]
            durations = [duration for duration, *_ in rows if duration is not None]
            span = f"{min(durations):.2f}–{max(durations):.2f}" if durations else "-"
            a, b, speedups = zip(*paired) if paired else ((), (), ())
            comparison.append(
                f"| {clean(band)} | {len(paired)}/{len(rows)} | {span} | "
                f"{fmt(1000 * statistics.median(a) if a else None)} | "
                f"{fmt(1000 * statistics.median(b) if b else None)} | "
                f"{fmt(statistics.geometric_mean(speedups) if speedups else None)} |"
            )
    if ratios:
        comparison.extend(
            [
                "",
                f"Geometric mean wall-time ratio across {len(ratios)} matched clips is {statistics.geometric_mean(ratios):.3f}×.",
            ]
        )
    text(out / "README.md", "\n".join(overview) + "\n")
    text(out / "BASELINES.md", "\n".join(baseline_page) + "\n")
    text(out / "COMPARISON.md", "\n".join(comparison) + "\n")
    csv_file(
        out / "measurements.csv",
        samples,
        (
            "runtime",
            "provider",
            "case",
            "language",
            "reference",
            "metric",
            "status",
            "error",
            "repeat",
            "transcript",
            "wall_seconds",
            "audio_seconds",
            "real_time_factor",
            "error_rate",
            "finish_reason",
        ),
    )
    write(out / "campaign.json", state)
    write(out / "report.json", data)
    index(history)


def render_enhancement(data, out, history):
    def number(value):
        return "—" if value is None else f"{value:.3f}"

    state, workload, profiles = data["campaign"], data["workload"], data["profiles"]
    enhancement.validate(workload)
    out = Path(out).resolve()
    history = Path(history).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = {item["id"]: item for item in workload["clips"]}
    nav = "[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)"
    notes = (
        "Warm wall time covers WAV decoding and complete offline enhancement. "
        "Model loading and WAV writing are excluded. RTF is wall seconds per input audio second. "
        "These public Inworld demo/golden clips are not the private VIBE evaluation set. "
        "Golden error is against the independent training-reference output; other errors are against the upstream CPU runtime. "
        "Operation and golden checks are not an independent VIBE quality evaluation. "
        "The upstream CPU comparison covers only cases completed and validated by both runtimes."
    )
    overview = [
        "# Inworld Voice Isolator: offline enhancement",
        "",
        nav,
        "",
        f"Workload `{workload['id']}` uses {len(cases)} fixed clips. Protocol `{enhancement.PROTOCOL}`.",
        "",
        notes,
        "",
        "| Runtime | Device | Complete and deterministic cases | Golden parity | Run |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    stats, samples = {}, []
    baseline_page = []
    for profile in profiles:
        meta = profile["metadata"]
        is_native = profile["kind"] == "akas"
        if is_native and meta["speech_workload"]["definition"] == workload and meta["protocol"] == enhancement.PROTOCOL:
            expected_cases = set(cases)
        else:
            evidence = dict(metadata=meta, results=profile["measurements"])
            checkpoint = profiles[0]["metadata"]["checkpoint_sha256"]
            if is_native:
                raise ValueError("enhancement profile metadata differs from workload")
            if enhancement.reference_matches(evidence, workload, checkpoint, golden_only=True):
                expected_cases = {key for key, item in cases.items() if item.get("golden")}
            elif enhancement.reference_matches(evidence, workload, checkpoint):
                expected_cases = set(cases)
            else:
                raise ValueError("enhancement baseline differs from workload")
        name = "AKAS" if is_native else "Inworld upstream"
        rows = profile["measurements"]
        indexed = {row["case"]["id"]: row for row in rows}
        if len(indexed) != len(rows) or any(cases.get(key) != row["case"] for key, row in indexed.items()):
            raise ValueError("undeclared or duplicate enhancement measurements")
        stats[profile["id"]] = (indexed, profile)
        golden = sum(row.get("correctness", {}).get("full_model_reference", False) for row in rows)
        passed = sum(row["status"] == "ok" for row in rows)
        qualified = profile["status"] == "passed" and passed == len(expected_cases) and golden == 1
        link = evidence_link(profile["run_id"], out, history)
        overview.append(
            f"| {name} ({meta['provider']}) | {clean(meta['device']['name'])} | {passed}/{len(expected_cases)} | {golden}/1 | {link} |"
        )
        lines = [
            f"# {name} enhancement measurements",
            "",
            nav,
            "",
            notes,
            "",
            f"Run {link}. Report SHA-256 `{profile['report_sha256']}`.",
            f"Model {meta['model']}. Source checkpoint `{meta['weight_format']}`. Execution storage `{meta.get('weight_dtype', meta['weight_format'])}`.",
            f"Checkpoint SHA-256 `{meta['checkpoint_sha256']}`. Each clip has {meta['warmup']} warmup and {meta['repeats']} measured requests.",
        ]
        runtime_version = meta.get("runtime_version") or meta.get("inworld-voice-isolation")
        if runtime_version:
            lines.append(f"Upstream runtime `{runtime_version}`; Torch `{meta.get('torch_version', 'unrecorded')}`.")
        if meta.get("cpu_threads"):
            lines.append(f"CPU threads: {meta['cpu_threads']}.")
        if meta.get("upstream_revision"):
            lines.append(f"Upstream source revision `{meta['upstream_revision']}`.")
        if meta.get("program_sha256"):
            lines.append(f"Compiled program SHA-256 `{meta['program_sha256']}`.")
        lines.extend(
            [
                "",
                "| Case | Audio s | Wall s | RTF | Deterministic | Golden max abs | Upstream RMS | Repeats |",
                "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        audio_links = []
        for case in cases:
            if case not in expected_cases:
                continue
            row = indexed.get(case, {})
            summary = row.get("summary", {})
            repeats = row.get("samples", [])
            golden_errors = [s["golden"]["max_abs_error"] for s in repeats if s.get("golden", {}).get("finite")]
            upstream_errors = [s["upstream_comparison"]["rms_error"] for s in repeats if s.get("upstream_comparison", {}).get("finite")]
            golden_text = f"{max(golden_errors):.3e}" if golden_errors else "—"
            upstream_text = f"{max(upstream_errors):.3e}" if upstream_errors else "—"
            lines.append(
                f"| {case} | {number(summary.get('audio_seconds'))} | "
                f"{number(summary.get('wall_seconds') if qualified else None)} | "
                f"{number(summary.get('real_time_factor') if qualified else None)} | {'yes' if row.get('correctness', {}).get('deterministic') else 'no'} | "
                f"{golden_text} | {upstream_text} | {len(repeats)} |"
            )
            if repeats:
                wav = repeats[0]
                paths = [
                    evidence_link(profile["run_id"], out, history, name=wav[f"{part}_wav"], label=part)
                    for part in ("voice", "background")
                ]
                audio_links.append(f"- `{case}`: " + " · ".join(paths))
            for repeat, sample in enumerate(repeats, 1):
                samples.append(
                    dict(
                        runtime=profile["id"],
                        provider=meta["provider"],
                        run_id=profile["run_id"],
                        case_id=case,
                        repeat=repeat,
                        wall_seconds=sample["wall_seconds"] if qualified else None,
                        audio_seconds=sample["audio_seconds"],
                        real_time_factor=sample["real_time_factor"] if qualified else None,
                        voice_sha256=sample["voice_sha256"],
                        golden_max_abs=sample.get("golden", {}).get("max_abs_error"),
                        upstream_rms=sample.get("upstream_comparison", {}).get("rms_error"),
                        voice_wav=evidence_path(profile["run_id"], out, history, sample["voice_wav"]),
                        background_wav=evidence_path(profile["run_id"], out, history, sample["background_wav"]),
                    )
                )
        if audio_links:
            lines.extend(["", "## Retained audio (first measured repeat)", "", *audio_links])
        if is_native:
            text(out / "AKAS.md", "\n".join(lines))
        else:
            baseline_page.append("\n".join(lines))
    text(
        out / "BASELINES.md",
        "\n\n---\n\n".join(baseline_page)
        if baseline_page
        else f"# Upstream enhancement measurements\n\n{nav}\n\nNo CPU run attached.",
    )
    comparison = [
        "# Offline enhancement comparison",
        "",
        nav,
        "",
        notes,
        "",
    ]
    baseline, cpu_profile = stats.get("inworld", ({}, {}))
    native, gpu_profile = stats.get("akas", ({}, {}))
    if baseline and set(baseline) != set(cases) and workload["id"] == "voice-enhancement-demo-v1":
        comparison.extend([
            "",
            "CPU timing is reported only for the 3-second golden. Upstream ran out of "
            "memory on the 10.77-second clip; the 24.73-second CPU clip had no valid "
            "measurement. See [upstream issue #5](https://github.com/inworld-ai/voice-isolator/issues/5).",
        ])
    comparison.extend([
        "",
        "| Case | Audio s | Upstream wall s | AKAS wall s | Wall speedup | Upstream RTF | AKAS RTF | Golden parity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    gpu_qualified = (
        gpu_profile.get("status") == "passed"
        and set(native) == set(cases)
        and all(row["status"] == "ok" for row in native.values())
        and any(row.get("correctness", {}).get("full_model_reference") for row in native.values())
    )
    for case in cases:
        old, new = baseline.get(case, {}), native.get(case, {})
        left = old.get("summary", {}) if cpu_profile.get("status") == "passed" and old.get("status") == "ok" else {}
        right = new.get("summary", {}) if gpu_qualified else {}
        ratio = (
            number(left["wall_seconds"] / right["wall_seconds"]) + "×"
            if left and right and left["audio_seconds"] == right["audio_seconds"]
            else "—"
        )
        golden = (
            "yes" if new.get("correctness", {}).get("full_model_reference")
            else "no" if cases[case].get("golden") else "not checked"
        )
        comparison.append(
            f"| {case} | {number(right.get('audio_seconds', left.get('audio_seconds')))} | "
            f"{number(left.get('wall_seconds'))} | {number(right.get('wall_seconds'))} | {ratio} | "
            f"{number(left.get('real_time_factor'))} | {number(right.get('real_time_factor'))} | {golden} |"
        )
    if memory := state.get("memory"):
        separate, shared = (memory["profiles"][name] for name in ("separate", "shared"))
        before, after = (row["dedicated_gpu_bytes"] for row in (separate, shared))
        details = evidence_link(memory["run_id"], out, history, name="report.json", label="Memory run")
        comparison.extend([
            "", "## Peak memory on the 11-second golden", "",
            "Separate from latency timing: Windows per-process GPU memory counters, "
            "with both exact-length graphs loaded. Identical compiled kernels and "
            "bit-for-bit identical voice and background WAVs across short → long → short.",
            "",
            "| Runtime allocation | Dedicated GPU peak (GiB) | Host working-set peak (MiB) |",
            "| --- | ---: | ---: |",
            *(
                f"| {label} | {row['dedicated_gpu_bytes'] / 2**30:.3f} | {row['host_working_set_bytes'] / 2**20:.1f} |"
                for label, row in (("Separate scratch", separate), ("Shared scratch", shared))
            ),
            "",
            f"Dedicated GPU memory fell by {100 * (1 - after / before):.1f}%. {details}.",
            "",
            "The shared-scratch build passed the independent 3-second and 11-second "
            "golden checks at `1e-4`. These checks do not establish accuracy on VIBE.",
        ])
        overview.extend(["", "See [Comparison](COMPARISON.md#peak-memory-on-the-11-second-golden) for memory and golden evidence."])
    if workload["id"] in {"voice-enhancement-demo-v1", "voice-enhancement-golden-3s-v1"}:
        upstream = "https://github.com/inworld-ai/voice-isolator/blob/" + workload["source"]["revision"]
        overview.extend([
            "",
            "## Audio sources and license",
            "",
            "The 3-second golden and the hiss demo are derived from LibriSpeech "
            "dev-clean utterance `2277-149896-0032` (*Sister Carrie*, read by zinniz; "
            "LibriSpeech © 2014 Vassil Panayotov, "
            "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). "
            "Inworld added hiss and prepared the golden fixture. The retained outputs "
            "are the voice and background separated by each runtime.",
            "",
            f"[Source recording notice]({upstream}/demo/examples/sources/LICENSE.txt) · "
            f"[Demo mix recipes]({upstream}/demo/build_examples.py) · "
            f"[Golden fixture attribution]({upstream}/tests/fixtures/README.md) · "
            f"[Upstream notices]({upstream}/NOTICE)",
            "",
            *(f"- [{case}]({item['source']})" for case, item in cases.items()),
        ])
    text(out / "README.md", "\n".join(overview))
    text(out / "COMPARISON.md", "\n".join(comparison))
    csv_file(
        out / "measurements.csv",
        samples,
        (
            "runtime",
            "provider",
            "run_id",
            "case_id",
            "repeat",
            "wall_seconds",
            "audio_seconds",
            "real_time_factor",
            "voice_sha256",
            "golden_max_abs",
            "upstream_rms",
            "voice_wav",
            "background_wav",
        ),
    )
    write(out / "report.json", data)
    write(out / "campaign.json", state)
    index(history)
    return out


def render_speech(data, out, history):
    state, workload, profiles = data["campaign"], data["workload"], data["profiles"]
    protocol = "client-speech-pcm-v1"
    if workload.get("protocol") != protocol or [p["kind"] for p in profiles] not in (
        ["akas"],
        ["akas", "baseline"],
        ["akas", "baseline", "baseline"],
    ):
        raise ValueError("invalid speech report protocol or profiles")
    utterances = workload["utterances"]
    cases = {item["id"]: item for item in utterances}
    default_lang = workload["request"].get("lang")
    languages = workload.get("languages") or {default_lang: dict(name=default_lang)}
    if len(cases) != len(utterances) or set(languages) != {
        item.get("lang", default_lang) for item in utterances
    }:
        raise ValueError("invalid speech case or language coverage")
    nav = "[Overview](README.md) · [Comparison](COMPARISON.md) · [AKAS](AKAS.md) · [Baselines](BASELINES.md) · [Measurements](measurements.csv)"
    explicit_voices = any("voice" in item for item in utterances)
    has_cpu = any(profile["id"] == "llama-cpu" for profile in profiles)
    runtime_count = "All runtimes" if has_cpu else "Both runtimes"
    voice_note = (
        f"{runtime_count} used the same case-specific presets. "
        if explicit_voices
        else f"{runtime_count} used one fixed voice for all languages. "
    )
    notes = (
        "Timings cover one warm localhost request through receipt of complete PCM16. "
        "Wall time and real-time factor (RTF, wall time divided by audio duration) are per-utterance medians. "
        "Comparison ratios use geometric means of matched successful utterances. "
        "Pass counts cover request completion, format and non-silent audio. AKAS also reports end of speech. "
        "WAV byte determinism is assessed within each runtime. "
        + voice_note
        + "Intelligibility, pronunciation and MOS were outside the measurement scope."
    )
    overview = [
        "# Speech benchmark report",
        "",
        nav,
        "",
        f"Workload `{workload['id']}` covers **{len(languages)} languages**, **{len(cases)} utterances** "
        f"and {workload['repeats']} measured requests per utterance after {workload['warmup']} warmup.",
        "",
        f"Protocol `{protocol}`. UTC {state['created_at']} to {state['updated_at']}.",
        "",
        notes,
        "",
    ]
    if "corpus" in workload:
        corpus = workload["corpus"]
        overview.extend(
            [
                f"Language coverage follows the [model card]({workload['language_source']}). "
                f"Prompts are from [{corpus['name']}]({corpus['url']}), "
                f"licensed [{corpus['license']}]({corpus['license_url']}). {corpus['selection']}",
                "",
            ]
        )
    if explicit_voices:
        overview.extend(
            [
                "| Language | Preset voice |",
                "| --- | --- |",
            ]
        )
        for lang, info in languages.items():
            voices = {
                item.get("voice", profiles[0]["metadata"]["voice"])
                for item in utterances
                if item.get("lang", default_lang) == lang
            }
            overview.append(
                f"| {info['name']} (`{lang}`) | {', '.join(sorted(voices))} |"
            )
        overview.append("")
    overview.extend(
        [
            "| Runtime | Device | Successful cases | Deterministic cases | Run |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    measurements = {}
    samples = []
    baseline_sections = []
    for profile in profiles:
        meta = profile["metadata"]
        if (
            meta["speech_workload"]["definition"] != workload
            or meta["protocol"] != protocol
        ):
            raise ValueError("speech profile metadata differs from the workload")
        rows = profile["measurements"]
        by_id = {row["case"]["id"]: row for row in rows}
        if len(by_id) != len(rows) or any(
            cases.get(row["case"]["id"]) != row["case"] for row in rows
        ):
            raise ValueError("undeclared or duplicate speech measurements")
        runtime = profile["id"]
        measurements[runtime] = by_id
        count = sum(r["status"] == "ok" for r in rows)
        stable = sum(
            r.get("correctness", {}).get("deterministic_pcm", False) for r in rows
        )
        link = evidence_link(profile["run_id"], out, history)
        overview.append(
            f"| {runtime} ({meta['provider']}) | {clean(meta['device']['name'])} | "
            f"{count}/{len(cases)} | {stable}/{len(cases)} | {link} |"
        )
        lines = [
            f"# {runtime} speech measurements",
            "",
            nav,
            "",
            notes,
            "",
            f"Model {meta['model']} · {meta['weight_format']} · {meta['sample_rate_hz']:,} Hz · "
            + (
                "case-specific voices are listed in the overview."
                if explicit_voices
                else f"voice `{meta['voice']}`."
            ),
            f"Run {link} · report SHA-256 `{profile['report_sha256']}`.",
            f"Checkpoint SHA-256 `{meta['checkpoint_sha256']}`. Projector SHA-256 `{meta['projector']['sha256']}`.",
        ]
        if meta.get("program_sha256"):
            lines.append(f"Compiled program SHA-256 `{meta['program_sha256']}`.")
        if meta.get("compiled_codec_frames"):
            lines.append(
                f"Compiled codec frame capacities: {', '.join(map(str, meta['compiled_codec_frames']))}."
            )
        if meta.get("runtime_version"):
            lines.append(f"Runtime version `{clean(meta['runtime_version'])}`.")
        lines.extend(
            [
                "",
                "| Language | Case | Tier | Status | Repeats | Wall ms | Audio s | RTF | PCM repeatable | Example | Error |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for item in utterances:
            case_id = item["id"]
            lang = item.get("lang", default_lang)
            row = by_id.get(case_id, {})
            summary = row.get("summary", {})
            reps = row.get("samples", ())
            status = row.get("status", "not measured")
            repeatable = row.get("correctness", {}).get("deterministic_pcm")
            wall = summary.get("wall_seconds")
            audio = summary.get("audio_seconds")
            rtf = summary.get("real_time_factor")
            wav = (
                evidence_link(
                    profile["run_id"], out, history, name=reps[0]["wav"], label="WAV"
                )
                if reps
                else "—"
            )
            lines.append(
                f"| {lang} | {case_id} | {item['tier']} | {status} | {len(reps)} | "
                f"{f'{wall * 1000:.1f}' if wall is not None else '—'} | "
                f"{f'{audio:.3f}' if audio is not None else '—'} | "
                f"{f'{rtf:.3f}' if rtf is not None else '—'} | "
                f"{'—' if repeatable is None else 'yes' if repeatable else 'no'} | "
                f"{wav} | {clean(row.get('error'))} |"
            )
            for repeat, sample in enumerate(reps or ({},), 1):
                samples.append(
                    dict(
                        runtime=runtime,
                        provider=meta["provider"],
                        run_id=profile["run_id"],
                        language=lang,
                        case_id=case_id,
                        tier=item["tier"],
                        source_row=item.get("source_row"),
                        text=item["text"],
                        status=status,
                        error=row.get("error"),
                        **(
                            {"voice": item.get("voice", meta["voice"])}
                            if explicit_voices
                            else {}
                        ),
                        **({"repeat": repeat, **sample} if sample else {}),
                    )
                )
        if profile["kind"] == "akas":
            text(out / "AKAS.md", "\n".join(lines))
        else:
            baseline_sections.append("\n".join(lines))
    text(
        out / "BASELINES.md",
        "\n\n---\n\n".join(baseline_sections)
        if baseline_sections
        else f"# Speech baselines\n\n{nav}\n\nNo frozen baseline was retained.",
    )
    comparison = [
        "# Speech benchmark comparison",
        "",
        nav,
        "",
        notes,
        "",
        "| Language | Cases | AKAS passed | Baseline passed | Paired | Wall wins | Wall speedup | Audio throughput speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    native, baseline = (measurements.get(name, {}) for name in ("akas", "llama"))
    for lang, name in [
        *[(code, row["name"]) for code, row in languages.items()],
        (None, "All languages"),
    ]:
        ids = [
            key
            for key, item in cases.items()
            if lang is None or item.get("lang", default_lang) == lang
        ]
        good = [
            {k for k in ids if rows.get(k, {}).get("status") == "ok"}
            for rows in (native, baseline)
        ]
        paired = sorted(good[0] & good[1])
        wall, rtf = (
            [
                baseline[k]["summary"][metric] / native[k]["summary"][metric]
                for k in paired
            ]
            for metric in ("wall_seconds", "real_time_factor")
        )
        label = f"{name} (`{lang}`)" if lang else f"**{name}**"
        ratios = [
            f"{statistics.geometric_mean(values):.3f}×" if values else "—"
            for values in (wall, rtf)
        ]
        comparison.append(
            f"| {label} | {len(ids)} | {len(good[0])} | {len(good[1])} | {len(paired)} | "
            f"{sum(value > 1 for value in wall)} | {ratios[0]} | {ratios[1]} |"
        )
    if has_cpu:
        cpu = measurements["llama-cpu"]
        comparison.extend(
            [
                "",
                "## Relative to llama.cpp CPU",
                "",
                "All three runtimes must complete a prompt for it to enter the ratios. "
                "Wall speedup is CPU wall time divided by the corresponding GPU wall time.",
                "",
                "| Language | Cases | CPU passed | Matched on all three | AKAS wall speedup | llama.cpp Vulkan wall speedup |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for lang, name in [
            *[(key, row["name"]) for key, row in languages.items()],
            (None, "All languages"),
        ]:
            ids = [
                key
                for key, item in cases.items()
                if lang is None or item.get("lang", default_lang) == lang
            ]
            cpu_good = [key for key in ids if cpu.get(key, {}).get("status") == "ok"]
            paired = [
                key
                for key in cpu_good
                if native.get(key, {}).get("status")
                == baseline.get(key, {}).get("status")
                == "ok"
            ]
            ratios = []
            for rows in (native, baseline):
                values = [
                    cpu[key]["summary"]["wall_seconds"]
                    / rows[key]["summary"]["wall_seconds"]
                    for key in paired
                ]
                ratios.append(
                    f"{statistics.geometric_mean(values):.3f}×" if values else "—"
                )
            comparison.append(
                f"| {name} | {len(ids)} | {len(cpu_good)} | {len(paired)} | {ratios[0]} | {ratios[1]} |"
            )
    text(out / "README.md", "\n".join(overview))
    text(out / "COMPARISON.md", "\n".join(comparison))
    csv_file(
        out / "measurements.csv",
        samples,
        (
            "runtime",
            "provider",
            "run_id",
            "language",
            *(("voice",) if explicit_voices else ()),
            "case_id",
            "tier",
            "source_row",
            "text",
            "status",
            "error",
            "repeat",
            "wall_seconds",
            "audio_seconds",
            "real_time_factor",
            "samples",
            "peak_pcm16",
            "audio_frames",
            "terminated",
            "wav",
            "wav_sha256",
        ),
    )
    write(out / "campaign.json", state)
    write(out / "report.json", data)
    index(history)


def export_runs(run_ids, *, model_id, workload_id, history="benchmarks", out=None,
                baseline_lock=None, memory_run=None):
    """Render retained native measurements through the existing campaign exporter.

    Importing a run does not create a qualified deployment or mutate its evidence.
    Baselines retain the ordinary campaign-start cutoff and workload matching.
    """
    history = Path(history).resolve()
    root = Path(__file__).resolve().parents[1]
    model = read(root / "benchmarks/models" / (model_id + ".json"))
    runs = [record(history, identity) for identity in run_ids]
    if not runs or any(data is None for data in runs):
        raise ValueError("retained native runs must exist")
    if any(data["metadata"].get("runtime") != "akas" for data in runs):
        raise ValueError("--runs accepts native AKAS records; baselines use normal selection")
    if any(
        run["metadata"].get("protocol")
        in ("client-speech-pcm-v1", transcription.PROTOCOL, enhancement.PROTOCOL)
        for run in runs
    ):
        if len(runs) != 1:
            raise ValueError("speech export accepts one AKAS run with its frozen reference")
        lock = read(baseline_lock) if isinstance(baseline_lock, (str, Path)) else baseline_lock
        return export_speech(
            runs[0], model, workload_id, history, out, lock, memory_run=memory_run
        )
    if memory_run:
        raise ValueError("--memory-run applies only to voice enhancement")
    requested = workload(workload_id)
    runs.sort(key=lambda data: data["started_at"])
    formats = set()
    for data in runs:
        meta = data["metadata"]
        if meta.get("model") not in (model_id, model["model"]):
            raise ValueError("retained run belongs to another model")
        weight = meta.get("weight_format")
        if weight is None and len(model["weights"]) == 1:
            weight = next(iter(model["weights"]))
        if weight not in model["weights"]:
            raise ValueError("retained run needs a declared weight format")
        formats.add(weight)
    spec = dict(models=[model_id], formats=sorted(formats),
                backends=sorted({data["metadata"]["provider"] for data in runs}),
                caches=sorted({data["metadata"].get("kv_dtype", "default") for data in runs}),
                workload=requested,
                baselines=[dict(runtime=runtime, provider="cuda", weight_format=weight, mode=mode)
                           for runtime, weight, mode in itertools.product(
                               ("vllm", "sglang"), sorted(formats), model["modes"])])
    state = dict(schema="akas.matrix-campaign.v1", specification=spec,
                 models={model_id: model}, source=runs[0]["source"],
                 created_at=runs[0]["started_at"],
                 updated_at=max(data.get("completed_at") or data["started_at"] for data in runs),
                 status="retained measurements", spent_seconds=0, entries=[],
                 retained_run_ids=[data["id"] for data in runs])
    if baseline_lock is not None:
        state["baseline_lock"] = (read(baseline_lock) if isinstance(baseline_lock, (str, Path))
                                  else baseline_lock)
    state["identity"] = hashlib.sha256(json.dumps(dict(
        runs=state["retained_run_ids"], specification=spec,
        **({"baseline_lock": state["baseline_lock"]} if baseline_lock is not None else {})), sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for identity, fields in identity_map(state).items():
            jobs = []
            for data in runs:
                meta = data["metadata"]
                weight = meta.get("weight_format", next(iter(model["weights"])))
                if (weight, meta["provider"], meta.get("kv_dtype", "default"), meta.get("mode", "ar")) != (
                    fields["weight_format"], fields["provider"], fields["cache_request"], fields["mode"]):
                    continue
                recipe = {"id": "retained-" + data["id"], "trials": 0,
                          **meta.get("recipe", meta.get("engine_config", {}))}
                jobs.append(dict(run_id=data["id"], status=data["status"],
                                 recipe=recipe,
                                 directory=str(history / "runs" / data["id"])))
            state["entries"].append(dict(id=identity, status="unqualified" if jobs else "not_measured",
                                          error="Retained measurements; no qualified deployment imported"))
            write(directory / identity / "campaign.json", dict(jobs=jobs))
        path = directory / "campaign.json"
        write(path, state)
        data = collect(path, history)
        data["measurement_notes"] = (
            "Retained measurements; importing evidence does not qualify an AKAS deployment. "
            "All values are medians. Phase protocol labels describe the measured boundaries. "
            "client-stream-burst-v1 uses concurrent token-arrival timestamps: TTFT includes "
            "queueing and prefill rate is total prompt tokens divided by maximum TTFT. "
            "native-gpu-prefill-ar127-v1 uses CUDA events around native prefill and autoregressive decode. "
            "Column labels distinguish native engine delivery, CUDA events and HTTP/SSE transports. "
            "Decode counts the remaining output tokens after the first. Unqualified values retain †. "
            "Original run records and source snapshots are unchanged.")
        destination = Path(out) if out else report_directory(state, history)
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
    archived = archive.catalog(history)
    allowed_guides = {history / name for name in ("README.md", "RESULTS.md", "ARCHIVE.md")}
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
        if (history / "archive.json").exists():
            write(staged_history / "archive.json", archived)
        for folder in sorted(reports.iterdir()):
            if folder.name == ".ipynb_checkpoints":
                continue
            if folder.name == "README.md" and folder.is_file():
                continue
            if folder.name in archived["reports"]:
                from tools.benchmark_archive import file_manifest
                expected = archived["reports"][folder.name]
                expected = expected.get("source_files", expected["files"])
                if file_manifest(folder) != expected:
                    raise ValueError(f"Archived report checksum mismatch: {folder}")
                continue
            if not folder.is_dir() or not (folder / "report.json").is_file():
                raise ValueError(f"Unexpected report entry: {folder}")
            data = read(folder / "report.json")
            expected_files = (
                STREAM_COMPARISON_FILES if data.get("schema") == STREAM_COMPARISON_SCHEMA else
                SPEECH_FILES
                if data.get("schema") in (SPEECH_SCHEMA, NATIVE_STREAM_SCHEMA)
                else REPORT_FILES
            ) | (
                {"PARITY.json"} if "performance_acceptance" in data else set()
            )
            if {
                p.name for p in folder.iterdir() if p.name != ".ipynb_checkpoints"
            } != expected_files:
                raise ValueError(
                    f"Report files differ from the standard bundle: {folder}"
                )
            if folder != report_directory(data["campaign"], history):
                raise ValueError(
                    f"Report directory does not match campaign identity: {folder}"
                )
            staged = staged_history / "reports" / folder.name
            render(data, staged, staged_history)
            for name in sorted(expected_files):
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
    source.add_argument("--runs", nargs="+", help="native run IDs to export without promoting qualification")
    source.add_argument(
        "--native-stream-comparison", nargs=4, metavar=("AKAS", "BASELINE", "CHECKS", "FIGURE"),
        help="export comparison from an AKAS run, native report, correctness run, and figure run",
    )
    parser.add_argument("--model", help="benchmark model declaration ID for --runs")
    parser.add_argument("--workload", help="versioned workload ID or JSON path for --runs")
    parser.add_argument("--history", type=Path, default=Path("benchmarks"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--baseline-lock", type=Path, help="freeze baseline IDs and verify their recorded checksums for --runs")
    parser.add_argument("--memory-run", help="sealed Voice Isolator memory A/B run to include with --runs")
    args = parser.parse_args()
    if args.memory_run and not args.runs:
        parser.error("--memory-run requires --runs")
    if args.check:
        check(args.history)
        print("Benchmark report format and generated content verified.")
        return
    if args.runs:
        if not args.model or not args.workload:
            parser.error("--runs requires --model and --workload")
        destination = export_runs(args.runs, model_id=args.model, workload_id=args.workload,
                                  history=args.history, out=args.out,
                                  baseline_lock=args.baseline_lock, memory_run=args.memory_run)
    elif args.native_stream_comparison:
        destination = export_native_stream_comparison(
            *args.native_stream_comparison, history=args.history, out=args.out,
        )
    elif args.render:
        destination = args.out or args.render.parent
        render(read(args.render), destination, args.history)
    else:
        destination = export(args.campaign, history=args.history, out=args.out)
    print(destination / "README.md")


if __name__ == "__main__":
    main()
