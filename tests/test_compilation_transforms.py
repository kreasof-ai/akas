"""Compile declaration smoke checks for draft and measured operation transforms."""

from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from akas.compilation.profiles.hybrid import decode, prefill, draft, draft_prefill, verification
from akas.models.qwen35.config import Qwen35Config


@pytest.fixture
def declarations(monkeypatch):
    import akas.compilation.build as pipeline

    calls = []

    def export(module, factory, *args, dependencies=()):
        # Serialized factories must exist in the installed package.
        assert callable(getattr(import_module(module), factory))
        return json.dumps(
            dict(module=module, factory=factory, args=args, dependencies=dependencies)
        )

    def compile(source, destination, **options):
        calls.append(json.loads(Path(source).read_text()))
        Path(destination).write_text(
            json.dumps(
                dict(
                    target=options["target"],
                    source=Path(source).read_text(),
                    files={"kernel.cubin": "0" * 64},
                )
            )
        )

    def read(path):
        return json.loads(Path(path).read_text()), None

    monkeypatch.setattr("tensor.compiler.frontend.source.export_source", export)
    monkeypatch.setattr("tensor.build", compile)
    monkeypatch.setattr(pipeline, "needs_build", lambda *args, **options: True)
    for name in ("hopper", "attention_workspace"):
        monkeypatch.setattr(
            import_module("akas.compilation.transforms." + name), "read_artifact", read
        )
    return calls


@pytest.fixture
def checkpoint():
    return SimpleNamespace(
        config=Qwen35Config(
            ("linear_attention", "full_attention"), 131072, 1e-6, 10000000
        ),
        weight_bytes=128,
    )


def load(path, name="prefill.json"):
    return json.loads((Path(path) / name).read_text())


def test_draft_and_verification_reuse_target_kernels(
    tmp_path, checkpoint, declarations
):
    target = decode.produce(checkpoint, tmp_path / "decode", slots=2, target="sm_90")
    result = draft.produce(checkpoint, target, tmp_path / "draft")
    manifest = load(result, "inference.json")
    assert manifest["draft"] == "qwen35-mtp" and len(manifest["config"]["layers"]) == 1
    assert (
        len(manifest["kernels"]) == len(load(target, "inference.json")["kernels"]) + 3
    )
    prefix = prefill.produce(
        checkpoint,
        tmp_path / "prefill",
        slots=2,
        chunk=4,
        target="sm_90",
        kv_dtype="fp8",
    )
    proposed = draft_prefill.produce(checkpoint, prefix, tmp_path / "draft-prefill")
    assert len(load(proposed)["kernels"]) == len(load(prefix)["kernels"]) + 3
    verified = verification.produce(prefix, tmp_path / "verified")
    rows = list(load(verified)["kernels"].values())
    assert any(
        row["kind"] == "verify_head"
        and row["parameters"]["o"] == checkpoint.config.vocab
        for row in rows
    )
    assert sum(row["kind"] == "restore" for row in rows) == 2


def test_attention_projection_and_hopper_overlays_share_build_pipeline(
    tmp_path, checkpoint, declarations
):
    from akas.compilation.transforms import compact_experts, hopper, hopper_verification, split_attention, split_linear, attention_workspace, bandwidth_projection

    prefix = prefill.produce(
        checkpoint,
        tmp_path / "prefill",
        slots=2,
        chunk=4,
        target="sm_90",
        kv_dtype="fp8",
    )
    verified = verification.produce(prefix, tmp_path / "verified")
    split = split_linear.produce(verified, tmp_path / "split-linear", grouped_rows=True)
    assert load(split)["split_linear"]
    expert = hopper_verification.produce(split, tmp_path / "experts")
    assert load(expert)["compact_experts"]["projections"]
    dense = hopper_verification.produce_dense(split, tmp_path / "dense")
    assert load(dense)["hopper_dense_schedule"]["reordered"]
    narrow = bandwidth_projection.dense_verification(dense, tmp_path / "narrow")
    assert load(narrow)["bandwidth_dense_columns"] == 64
    attention = split_attention.produce(
        verified,
        tmp_path / "attention",
        query_tokens=4,
        kernel_target="sm_90a",
        pad_queries=True,
    )
    workspace = attention_workspace.produce_verification(
        attention, tmp_path / "workspace"
    )
    assert load(workspace, "attention-workspace.json")["scope"] == "verification"
    assert load(workspace, "attention-workspace.json")["expected_attention_calls"] == 1
    compact = compact_experts.produce(prefix, tmp_path / "compact")
    projected = hopper.produce(compact, tmp_path / "hopper", hopper_attention=True)
    prepared = dict(hopper=projected, prefill=compact)
    workspace = attention_workspace.produce(prepared, tmp_path / "prefill-workspace")
    assert load(workspace, "attention-workspace.json")["expected_attention_calls"] == 1
    assert load(workspace, "attention-workspace.json")["kernels"].keys() == {
        "decode",
        "attention",
    }
    narrowed = bandwidth_projection.prefill(prepared, tmp_path / "prefill-narrow")
    assert load(narrowed, "hopper.json")["projection_columns"] == 64


def test_recompute_and_decode_projection_profiles(tmp_path, checkpoint, declarations):
    from akas.compilation.transforms import recompute, decode_profile

    prefix = prefill.produce(
        checkpoint,
        tmp_path / "prefill",
        slots=1,
        chunk=64,
        target="sm_90",
        kv_dtype="fp8",
    )
    verified = verification.produce(prefix, tmp_path / "verified")
    restored = recompute.produce(verified, tmp_path / "recompute")
    assert load(restored)["state_recompute"] == "accepted-prefix-recompute"
    decoder = decode_profile.produce(
        checkpoint, tmp_path / "decoder", slots=2, target="sm_90", kv_dtype="fp8"
    )
    assert load(decoder, "inference.json")["profile_producer_sha256"]
    assert any(call["factory"] == "quantize_kernel" for call in declarations)
