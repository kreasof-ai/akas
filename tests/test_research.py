"""Research presets reuse the compiler, scheduler and qualified program identities."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from akas.benchmarking import research


def prepared(tmp_path):
    (tmp_path / "program.json").write_text('{}\n')
    return dict(schema="akas.research.v1", definition="test:Model", checkpoint="weights",
                program=str(tmp_path), provider="cuda", target="sm_89", batches=[1], chunk=4,
                windows=[4], context=100, policy={}, model_options={}, checkpoint_options={},
                source_identity="source", program_sha256=hashlib.sha256((tmp_path / "program.json").read_bytes()).hexdigest())


def test_geometry_matrix_contains_each_required_phase_without_duplicate_proposals():
    rows = research.geometries([1, 4], 8, [4, 8])
    assert len(rows) == len(set(rows))
    for batch in (1, 4):
        assert (batch, 1) in rows and (batch, 8) in rows
        for chunk in (1, 4, 8):
            assert (batch, chunk, "proposal") in rows
        for chunk in (4, 8):
            assert (batch, chunk, "verify") in rows


def test_prepared_identity_rejects_changed_and_legacy_manifests(tmp_path):
    settings = prepared(tmp_path)
    assert research.validate(settings) is settings
    (tmp_path / "program.json").write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="changed"):
        research.validate(settings)
    with pytest.raises(ValueError, match="recompilation"):
        research.validate(dict(paths={"decode":"old"}))


def test_prepare_uses_declared_model_and_common_compile_pipeline(tmp_path, monkeypatch):
    calls = []
    model = SimpleNamespace(checkpoint_type=lambda path, **options: (path, options), compose=object())
    monkeypatch.setattr(research, "definition", lambda name: model)
    monkeypatch.setattr("akas.artifacts.fingerprint.backbone_hashes", lambda: {"source":"digest"})
    def compile(checkpoint, directory, **options):
        calls.append((checkpoint, options))
        Path(directory).mkdir(parents=True)
        (Path(directory) / "program.json").write_text('{}\n')
    monkeypatch.setattr("akas.compilation.compile.compile_program", compile)
    result = research.prepare("new.model:Definition", "weights", tmp_path / "case", batches=[1, 4],
                              windows=[4], chunk=8, checkpoint_options={"precision":"float16"},
                              policy={"linear":{"partitions":2}})
    assert calls[0][0] == ("weights", {"precision":"float16"})
    assert calls[0][1]["definition"] is model.compose
    assert calls[0][1]["geometries"] == research.geometries([1, 4], 8, [4])
    assert calls[0][1]["max_requests"] == 4
    assert result["schema"] == "akas.research.v1"
    research.validate(result)


@pytest.mark.parametrize("window", [None, 4])
def test_workload_replay_uses_actual_ids_and_preserves_cohort_counts(tmp_path, monkeypatch, window):
    from test_qwen_server import Target, Proposer
    import tensor
    settings = prepared(tmp_path)
    class Device:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def synchronize(self): pass
    class Model(Target):
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(tensor, "Device", lambda **options: Device())
    monkeypatch.setattr(research, "definition", lambda name: lambda *args, **kwargs: Model(1))
    monkeypatch.setattr("akas.models.heads.mtp.MTPHead", lambda target: Proposer())
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps(dict(requests=[dict(prompt_token_ids=[12, 24], output_tokens=5),
                                                  dict(prompt_token_ids=[30], output_tokens=2)])))
    report = research.run_workload(settings, workload, window=window)
    assert report["completed_requests"] == 2 and report["generated_tokens"] == 7
    assert report["output_ids"] == [list(range(25, 30)), [31, 32]]
    assert report["aggregate_output_tokens_per_second"] > 0
    assert report["model_throughput_qualified"] is False
