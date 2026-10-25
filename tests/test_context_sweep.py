"""Vulkan sweeps reject changed inputs and preserve finite adapter requests."""

import json
from types import SimpleNamespace

import pytest

from akas.benchmarking.workload import prompt_digest
from tools.context_sweep import inputs, vulkan_device


@pytest.mark.parametrize("corruption", [None, "length", "fingerprint"])
def test_explicit_sweep_inputs_are_checked_before_execution(tmp_path, corruption):
    path = tmp_path / "prompts.json"
    ids = [1, 2, 3]
    row = dict(prefix_tokens=3, token_ids=ids, prompt_sha256=prompt_digest([ids]))
    if corruption == "length":
        row["token_ids"] = [1, 2]
    if corruption == "fingerprint":
        row["token_ids"] = [1, 2, 4]
    path.write_text(json.dumps(dict(cases=[row])))
    args = SimpleNamespace(prompts=path, prefixes=[3])
    if corruption:
        with pytest.raises(ValueError, match="fingerprint"):
            inputs(args)
    else:
        assert inputs(args)[0]["token_ids"] == ids


@pytest.mark.parametrize("limit,expected", [(8, 8), (64, 64), (0xFFFFFFFF, 32)])
def test_only_undefined_vulkan_binding_limit_is_capped(monkeypatch, limit, expected):
    import tensor

    class Device:
        def __init__(self, **options):
            self.info = dict(adapter=dict(backend_type="Vulkan", adapter_type="DiscreteGPU"))
            values = {"max-storage-buffers-per-shader-stage": limit}
            self._adapter = SimpleNamespace(limits=values, _limits=values)
            assert options["max_buffer_size"] >= 256 * 1024**2

        def __enter__(self):
            assert self._adapter.limits["max-storage-buffers-per-shader-stage"] == expected
            return self

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(tensor, "Device", Device)
    with vulkan_device() as device:
        assert device.info["adapter_storage_buffers_reported"] == limit


@pytest.mark.parametrize("mismatch", [None, "checkpoint", "tokens", "cache", "coverage", "ancestry_protocol"])
def test_comparison_requires_matched_runs_and_regenerates(tmp_path, mismatch):
    from akas.benchmarking.history import write
    from tools.campaign_report import check
    from tools.context_sweep import report

    records = []
    for runtime, provider, protocol in (("akas", "webgpu", "native-token-step-v2"),
                                        ("llama", "vulkan", "baseline-burst-v3")):
        meta = dict(runtime=runtime, provider=provider, protocol=protocol,
                    checkpoint_sha256="same-weights", prompt_manifest_sha256="same-manifest",
                    output_tokens=128, kv_dtype="float16", model="LiquidAI/LFM2.5-2.6B",
                    prefill_chunk=128, warmup=0, repeats=1)
        records.append(dict(id=runtime, status="passed", started_at="2026-10-24T17:00:00Z",
            completed_at="2026-10-24T17:01:00Z", metadata=meta,
            source=dict(revision="fixture", tree_sha256="fixture", files={}), search=[],
            results=[dict(case=dict(batch=1, prompt=32, generated=128), status="ok",
                prompt_sha256="same-tokens", samples=[dict(prefill_tokens_per_second=100,
                decode_tokens_per_second=50, phase_protocol=protocol)])]))
    if mismatch == "checkpoint":
        records[1]["metadata"]["checkpoint_sha256"] = "other-weights"
    elif mismatch == "cache":
        records[1]["metadata"]["kv_dtype"] = "fp8"
    elif mismatch == "tokens":
        records[1]["results"][0]["prompt_sha256"] = "other-tokens"
    elif mismatch == "coverage":
        records[1]["results"][0]["case"]["prompt"] = 512
    elif mismatch == "ancestry_protocol":
        from copy import deepcopy
        parent, grandparent = deepcopy(records[0]), deepcopy(records[0])
        records[0]["change"] = dict(parent_run="parent")
        parent.update(id="parent", change=dict(parent_run="grandparent"))
        grandparent.update(id="grandparent", change=dict(parent_run=None))
        parent["metadata"].update(warmup=1, repeats=3)
        records += [parent, grandparent]
    for data in records:
        write(tmp_path / "runs" / data["id"] / "run.json", data)
    args = SimpleNamespace(history=tmp_path, runs=["akas", "llama"], out=tmp_path / "build")
    if mismatch and mismatch != "ancestry_protocol":
        with pytest.raises(ValueError, match="mismatched"):
            report(args)
    else:
        args.reference_run = "akas"
        report(args)
        saved = next((tmp_path / "reports").glob("*/report.json"))
        data = json.loads(saved.read_text())
        assert data["retuning"]["cases"][0]["before_decode"] == 50
        if mismatch == "ancestry_protocol":
            assert data["calibration_unavailable"]["run_ids"] == ["parent", "grandparent"]
        check(tmp_path)
