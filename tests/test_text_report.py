"""Real-text tables preserve example identity and reject unmatched evidence."""

import pytest

from akas.benchmarking.history import digest, write
from akas.benchmarking.workload import prompt_digest
from tools.campaign_report import text_comparisons


@pytest.fixture
def saved(tmp_path):
    folder = tmp_path / "runs/probe"
    case = dict(id="source:17:2", source="source", example_id="17", split="heldout",
                category="coding", prefix_tokens=2, token_ids=[3, 4],
                prompt_sha256=prompt_digest([[3, 4]]), preprocessing=dict(cropped=False))
    workload = dict(tokenizer_checkpoint_sha256="target", cases=[case], sources={}, output_tokens=4)
    path = folder / "benchmark/text-workload.json"
    write(path, workload)
    summary = {key: dict(median=value) for key, value in
               (("decode_tokens_per_second", 20), ("prefill_tokens_per_second", 100))}
    ar = dict(case=dict(batch=1, prompt=2, generated=4),
              prompt_sha256=case["prompt_sha256"], summary=summary)
    row = dict(case=ar["case"], text_case={k: v for k, v in case.items() if k != "token_ids"},
               ar=ar, summary=summary, prompt_sha256=case["prompt_sha256"],
               native_speculation=dict(request_rounds=2, proposed=4, accepted=1, emitted=3),
               decode_tokens=3, output_ids=[[1, 2, 3, 4]], ar_output_ids=[[1, 2, 3, 4]],
               output_text="answer", ar_output_text="answer",
               output_agreement=dict(exact=True, common_prefix_tokens=4), warmup=1, repeats=3)
    data = dict(id="probe", status="passed", completed_at="2026-01-01T00:00:00Z",
                started_at="2026-01-01T00:00:00Z", source={}, results=[row],
                metadata=dict(checkpoint_sha256="target", dataset_split="heldout",
                              text_workload_sha256=digest(path), confidence_threshold=.5,
                              kv_dtype="float16", draft_kv_dtype="float16", qualification="probe"),
                artifacts=[dict(path="benchmark/text-workload.json", sha256=digest(path))])
    write(folder / "run.json", data)
    return tmp_path, folder, data, dict(text_workload_runs=["probe"]), {"winner": dict(checkpoint_sha256="target")}


def test_text_report_preserves_acceptance_outputs_and_prompt_identity(saved):
    history, _, _, state, evidence = saved
    row = text_comparisons(state, history, evidence)[0]["cases"][0]
    assert row["acceptance"] == .25 and row["tokens_per_round"] == 1.5
    assert row["decode_speedup"] == 1 and row["output_ids"] == [[1, 2, 3, 4]]
    assert row["example_id"] == "17" and row["split"] == "heldout"
    assert "probe" in evidence


@pytest.mark.parametrize("change", ["checksum", "failed", "checkpoint", "prompt", "split", "incomplete", "accounting"])
def test_text_report_rejects_changed_inputs_or_incomplete_evidence(saved, change):
    history, folder, data, state, evidence = saved
    if change == "checksum":
        (folder / "benchmark/text-workload.json").write_text("{}")
    elif change == "failed":
        data["status"] = "failed"
    elif change == "checkpoint":
        data["metadata"]["checkpoint_sha256"] = "another-target"
    elif change == "prompt":
        data["results"][0]["ar"]["prompt_sha256"] = "another-prompt"
    elif change == "split":
        data["metadata"]["dataset_split"] = "calibration"
    elif change == "incomplete":
        data["results"] = []
    elif change == "accounting":
        data["results"][0]["native_speculation"]["emitted"] = 5
    write(folder / "run.json", data)
    with pytest.raises(ValueError):
        text_comparisons(state, history, evidence)
