"""Voice-enhancement workload, waveform comparison and error summaries."""

import json

import numpy as np
from akas.benchmarking.workload import ENHANCEMENT_PROTOCOL as PROTOCOL


def validate(workload):
    request = workload["request"]
    if (
        workload["schema"] != "akas.voice-enhancement-workload.v1"
        or workload["protocol"] != PROTOCOL
        or request != {
            "mode": "offline",
            "sample_rate": 48000,
            "chunk_seconds": 8,
            "overlap_seconds": 2,
            "passes": 1,
        }
        or workload["warmup"] < 1
        or workload["repeats"] < 1
        or not workload["clips"]
        or len({item["id"] for item in workload["clips"]}) != len(workload["clips"])
        or len({item["id"] for item in workload["clips"] if item.get("golden") is not None}) != 1
        or type(workload["warmup"]) is not int
        or type(workload["repeats"]) is not int
        or any(
            len(item["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in item["sha256"])
            for item in workload["clips"]
        )
    ):
        raise ValueError("voice enhancement requires fixed 48 kHz inputs and warm repeats")


def errors(actual, reference):
    actual, reference = np.asarray(actual, dtype="float64"), np.asarray(reference, dtype="float64")
    if actual.shape != reference.shape or not actual.size or not np.isfinite(actual).all():
        return dict(finite=False, max_abs_error=None, rms_error=None, relative_rms_error=None, cosine=None)
    delta = actual - reference
    rms = float(np.sqrt(np.mean(delta**2)))
    reference_rms = float(np.sqrt(np.mean(reference**2)))
    cosine = float(np.dot(actual, reference) / max(np.linalg.norm(actual) * np.linalg.norm(reference), 1e-30))
    return dict(
        finite=True,
        max_abs_error=float(np.max(np.abs(delta))),
        rms_error=rms,
        relative_rms_error=rms / max(reference_rms, 1e-30),
        cosine=cosine,
    )


def reference_matches(reference, workload, checkpoint_sha256, *, golden_only=False):
    """Permit only the identical golden case when exporting a partial comparison."""
    source = reference["metadata"]
    declared = source["speech_workload"]["definition"]
    expected = workload
    if golden_only:
        expected = {**workload, "clips": [c for c in workload["clips"] if c.get("golden")]}
        declared = {**declared, "id": workload["id"]}
    cases = expected["clips"]
    rows = reference["results"]
    return (
        declared == expected
        and source["runtime"] == "inworld"
        and source["provider"] == "cpu"
        and source["protocol"] == workload["protocol"]
        and source["checkpoint_sha256"] == checkpoint_sha256
        and source.get("weight_dtype") == "float32"
        and source["warmup"] == workload["warmup"]
        and source["repeats"] == workload["repeats"]
        and [row["case"] for row in rows] == cases
        and all(
            row["status"] == "ok"
            and len(row["samples"]) == workload["repeats"]
            and row["correctness"]["deterministic"]
            and (not row["case"].get("golden") or row["correctness"]["full_model_reference"])
            for row in rows
        )
    )


def reference_report(history, run_id, workload, checkpoint_sha256, *, golden_only=False):
    from hashlib import sha256

    from akas.benchmarking.archive import read_file
    from akas.benchmarking.history import load_record

    left = load_record(history, run_id)
    raw = read_file(history, "runs", run_id, "report.json")
    reference = json.loads(raw)
    if (
        left["metadata"] != reference["metadata"]
        or left["results"] != reference["results"]
        or left["status"] != "passed"
        or sha256(raw).hexdigest() != next(a["sha256"] for a in left["artifacts"] if a["path"] == "report.json")
        or not (
            reference_matches(reference, workload, checkpoint_sha256)
            or (golden_only and reference_matches(
                reference, workload, checkpoint_sha256, golden_only=True
            ))
        )
    ):
        raise ValueError("frozen upstream enhancement reference does not match")
    return raw, reference


def deterministic(samples):
    return bool(samples) and len({sample["voice_sha256"] for sample in samples}) == 1
