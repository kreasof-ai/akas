"""Precision tables reject mismatched prompts and corrupted or failed evidence."""

import pytest
from akas.benchmarking.history import digest, write
from tools.campaign_report import precision_comparisons


@pytest.fixture
def paired(tmp_path):
    history = tmp_path / "benchmarks"

    def save(name, dtype, parent=None):
        root = history / "runs" / name
        row = dict(
            name="question",
            prefix=32,
            prompt_sha256="prompt",
            output_ids=[1, 2],
            output_text="answer",
        )
        if parent:
            row.update(
                output_ids=[1, 3],
                output_text="changed",
                greedy_comparison=dict(f16_output_ids=[1, 2]),
                teacher_forced=dict(
                    steps=[
                        dict(
                            top1_equal=True,
                            kl_f16_to_fp8=0.01,
                            reference_token_nll_delta=0.02,
                        ),
                        dict(
                            top1_equal=False,
                            kl_f16_to_fp8=0.03,
                            reference_token_nll_delta=-0.01,
                        ),
                    ]
                ),
            )
        write(root / "precision/outputs.json", dict(run_id=name, cases=[row]))
        artifact = dict(
            path="precision/outputs.json",
            sha256=digest(root / "precision/outputs.json"),
        )
        data = dict(
            id=name,
            status="passed",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T01:00:00Z",
            source={},
            metadata=dict(
                protocol="kv-output-comparison-v1",
                checkpoint_sha256="checkpoint",
                kv_dtype=dtype,
                qualification="Controlled probe, not accuracy.",
            ),
            artifacts=[artifact],
            change=dict(parent_run=parent),
        )
        write(root / "run.json", data)

    save("reference", "float16")
    save("candidate", "fp8", "reference")
    return (
        history,
        dict(precision_comparison_runs=["candidate"]),
        {"model": dict(checkpoint_sha256="checkpoint")},
    )


def test_precision_report_keeps_text_ids_and_shared_history_metrics(paired):
    history, state, evidence = paired
    result = precision_comparisons(state, history, evidence)
    row = result[0]["cases"][0]
    assert row["reference_output_ids"] == [1, 2]
    assert row["candidate_output_ids"] == [1, 3]
    assert row["reference_output_text"] == "answer"
    assert row["candidate_output_text"] == "changed"
    assert row["common_prefix_tokens"] == 1 and not row["exact"]
    assert row["top1_agreement"] == 0.5
    assert row["mean_kl_f16_to_candidate"] == 0.02
    assert set(evidence) == {"model", "reference", "candidate"}


def test_precision_report_rejects_corrupt_output_artifact(paired):
    history, state, evidence = paired
    (history / "runs/candidate/precision/outputs.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        precision_comparisons(state, history, evidence)


@pytest.mark.parametrize("change", ["failed", "checkpoint", "prompt", "continuation"])
def test_precision_report_rejects_unmatched_reference(paired, change):
    import json

    history, state, evidence = paired
    root = history / "runs/candidate"
    data = json.loads((root / "run.json").read_text())
    if change == "failed":
        data["status"] = "failed"
    elif change == "checkpoint":
        data["metadata"]["checkpoint_sha256"] = "other"
    else:
        payload = json.loads((root / "precision/outputs.json").read_text())
        if change == "prompt":
            payload["cases"][0]["prompt_sha256"] = "other"
        else:
            payload["cases"][0]["greedy_comparison"]["f16_output_ids"] = [9]
        write(root / "precision/outputs.json", payload)
        data["artifacts"][0]["sha256"] = digest(root / "precision/outputs.json")
    write(root / "run.json", data)
    with pytest.raises(ValueError):
        precision_comparisons(state, history, evidence)
