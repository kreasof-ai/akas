"""Checksummed import of an actual prefill state into a compatible decode graph."""

import json
from pathlib import Path
import numpy as np
from akas.benchmarking.history import digest
from akas.benchmarking.state import RecurrentSnapshot
from akas.benchmarking.workload import prompt_digest
from tools.lfm3_vulkan_benchmark import PROFILE
from tools.model_benchmark import prompts


def load(model, directory, prefix, context, slot, producer_record):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "state.json").read_text())
    if (
        metadata["checkpoint_sha256"] != PROFILE.checkpoint_sha256
        or metadata["prompt_sha256"] != prompt_digest(prompts(prefix, PROFILE.vocab))
        or metadata["prefix"] != prefix
        or metadata["context"] != context
        or metadata["positions"] != [prefix]
        or slot != 0
        or model.pending
    ):
        raise ValueError("real-prefill state model/prompt/context/lifecycle mismatch")
    producer = metadata["producer_program"]
    rows = [r for r in producer_record["results"] if r["case"]["prompt"] == prefix]
    pm = producer_record["metadata"]
    if (
        producer_record["status"] != "passed"
        or metadata.get("producer_run_id", producer_record["id"])
        != producer_record["id"]
        or pm.get("runtime") != "akas"
        or pm.get("protocol") != "real-prefill-state-restored-wall-decode-v1"
        or pm.get("checkpoint_sha256") != metadata["checkpoint_sha256"]
        or pm.get("max_context") != context
        or len(rows) != 1
        or rows[0]["prompt_sha256"] != metadata["prompt_sha256"]
        or rows[0]["program_sha256"] != producer["sha256"]
        or rows[0]["output_ids"][0] != metadata["first"][0]
    ):
        raise ValueError("cached prefill does not match its sealed producer run")
    manifest_path = Path(producer["path"]) / "program.json"
    if digest(manifest_path) != producer["sha256"]:
        raise ValueError("prefill producer program hash changed")
    producer_manifest = json.loads(manifest_path.read_text())
    current = model.program
    for field in ("config", "max_requests", "max_context", "provider"):
        if current[field] != producer_manifest[field]:
            raise ValueError(f"prefill/decode program {field} differs")
    # Only attention decode scheduling may change; prefill arithmetic, cache
    # storage, packed weight layout and rotary conventions must be identical.
    ignored = {"decode_splits", "cooperative_decode_attention"}

    def options(program):
        return {k: v for k, v in program["model_options"].items() if k not in ignored}

    if options(current) != options(producer_manifest):
        raise ValueError("prefill/decode numerical or storage options differ")
    if metadata["files"].keys() != model.batches.states.keys():
        raise ValueError("real-prefill state buffer names differ")
    arrays = {}
    for name, entry in metadata["files"].items():
        path = (directory / entry["file"]).resolve()
        if path.parent != directory or digest(path) != entry["sha256"]:
            raise ValueError("real-prefill state path/hash mismatch")
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        target = model.batches.states[name]
        if (
            list(array.shape) != entry["shape"]
            or str(array.dtype) != entry["dtype"]
            or tuple(array.shape) != tuple(target.shape)
            or str(array.dtype) != str(target.dtype)
        ):
            raise ValueError(f"real-prefill state shape/dtype mismatch: {name}")
        arrays[name] = array
    saved = RecurrentSnapshot(
        model,
        np.asarray(metadata["positions"], dtype=model.positions.dtype),
        model.state.generations.copy(),
        model.occupied.copy(),
        arrays,
    )
    provenance = dict(
        producer_run_id=producer_record["id"],
        state_manifest_sha256=digest(directory / "state.json"),
        producer_program_sha256=producer["sha256"],
        prefill_observation=rows[0]["prefill_observation"],
    )
    return saved, metadata["first"], provenance
