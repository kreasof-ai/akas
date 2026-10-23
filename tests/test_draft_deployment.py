"""Deployment identity covers external draft programs and checkpoint bytes."""

import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from akas.artifacts.write import checksum
from akas.benchmarking.history import write


def program(root, content):
    root.mkdir()
    binary = root / "kernel.tbin"
    binary.write_bytes(content)
    phase = root / "forward.json"
    write(phase, dict(kernels={"op": dict(path="kernel.tbin", sha256=checksum(binary))}))
    write(root / "program.json", dict(phases=[dict(path="forward.json", sha256=checksum(phase))]))
    return root


def test_external_draft_descriptor_retains_all_artifacts_and_relative_paths(tmp_path, monkeypatch):
    import akas.artifacts.read as artifacts
    import akas.optimization.deployment as deployment
    import akas.optimization.planning as planning

    monkeypatch.setattr(artifacts, "read_program", lambda root: json.loads((Path(root) / "program.json").read_text()))
    monkeypatch.setattr(deployment, "runtime_source", lambda: "runtime")
    monkeypatch.setattr(planning, "software", lambda: {"tensor-workspace": "test"})
    target = program(tmp_path / "target", b"target kernel")
    draft = program(tmp_path / "draft", b"draft kernel")
    checkpoint = tmp_path / "draft.gguf"
    checkpoint.write_bytes(b"pinned draft weights")
    recipe = dict(provider="cuda", window=10, graphs=True, round_capture=False,
                  head="akas.models.dspark.proposer:load",
                  head_options=dict(checkpoint=str(checkpoint), bundle=str(draft)))
    descriptor = deployment.describe({}, recipe, target, checkpoint_sha256="target sha",
                                     campaign_identity="campaign", run_id="run")
    options = descriptor["recipe"]["head_options"]
    assert options["bundle"] == "draft" and options["checkpoint"] == "draft.gguf"
    assert options["checkpoint_sha256"] == checksum(checkpoint)
    assert options["program_sha256"] == checksum(draft / "program.json")
    assert recipe["head_options"]["bundle"] == str(draft)
    assert {item["path"] for item in descriptor["artifacts"]} == {
        "target/program.json", "target/forward.json", "target/kernel.tbin",
        "draft/program.json", "draft/forward.json", "draft/kernel.tbin"}
    descriptor["status"] = "qualified"
    destination = tmp_path / "deployment.json"
    write(destination, descriptor)
    assert deployment.load(destination)["run_id"] == "run"
    (draft / "kernel.tbin").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="checksum mismatch"):
        deployment.load(destination)


@pytest.mark.parametrize("changed", ["checkpoint", "program"])
def test_draft_loader_checks_identities_before_allocating(tmp_path, monkeypatch, changed):
    import akas.models.dspark.model as model
    from akas.models.dspark.proposer import load

    checkpoint = tmp_path / "draft.gguf"
    checkpoint.write_bytes(b"weights")
    bundle = program(tmp_path / "draft", b"kernel")
    expected = dict(checkpoint_sha256=checksum(checkpoint),
                    program_sha256=checksum(bundle / "program.json"))
    allocated = []
    monkeypatch.setattr(model, "DSparkDraft", lambda *args, **kw: allocated.append(True))
    path = checkpoint if changed == "checkpoint" else bundle / "program.json"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load(None, checkpoint=checkpoint, bundle=bundle, **expected)
    assert allocated == []


def test_draft_loader_releases_allocations_when_pairing_fails(tmp_path, monkeypatch):
    import akas.models.dspark.model as model
    import akas.models.dspark.proposer as module

    closed, opened = [], []
    target = SimpleNamespace(device=object())
    draft = SimpleNamespace(close=lambda: closed.append(True))

    def create(checkpoint, bundle, device, **options):
        opened.append((checkpoint, bundle, device, options))
        return draft

    def reject(*args, **kwargs):
        raise ValueError("incompatible pairing")

    monkeypatch.setattr(model, "DSparkDraft", create)
    monkeypatch.setattr(module, "DSparkProposer", reject)
    with pytest.raises(ValueError, match="pairing"):
        module.load(target, checkpoint="draft", bundle="program")
    assert opened[0][2] is target.device
    assert opened[0][3]["checkpoint_options"]["target"] is target
    assert closed == [True]
