"""Research preset using one compiled program and the shared performance harness.

Legacy separate bundle/installer flags have been replaced by --policy and
--model-options. --workload retains actual request replay; --windows qualifies
multiple speculative geometries in one program. --prepared reuses its manifest.
"""

from akas.benchmarking.research import main as research_main

DEFINITION = "akas.models.qwen35.model:Qwen35Batch"
PRESET = {'batches': [8, 16, 32, 64], 'batch': 8, 'chunk': 512, 'windows': [32], 'context': 48000, 'target': 'sm_90a', 'prompt': 32, 'generated': 32, 'model_options': {'kv_dtype': 'fp8'}, 'policy': {}}


import gzip
import json


def workload(slots, path="/workspace/workload-c64.json.gz"):
    from benchmarks.llm_serving.workload import digest, validate

    with gzip.open(path, "rt") as source:
        value = json.load(source)
    validate(value)
    if slots not in (8, 16, 32, 64):
        raise ValueError("unsupported scaling point")
    value["requests"] = value["requests"][:slots]
    value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    return validate(value)


def compare_records(expected, actual):
    if len(expected) != len(actual):
        raise ValueError("slot coverage mismatch")
    differences = [
        dict(slot=s, components=[k for k in a if a[k] != b.get(k)])
        for s, (a, b) in enumerate(zip(expected, actual))
        if a != b
    ]
    return dict(
        slots=len(actual),
        differences=differences,
        finite=all(r["finite"] for r in actual),
        kv_prefix_unchanged=all(r["kv_prefix_unchanged"] for r in actual),
        passed=not differences
        and all(r["finite"] and r["kv_prefix_unchanged"] for r in actual),
        canonical_model_qualified=False,
        model_throughput_qualified=False,
    )


def main():
    research_main(model_definition=DEFINITION, preset=PRESET, description=__doc__)


if __name__ == "__main__":
    main()
