"""H200 campaign and optimization flow must use standard AKAS modules and preserve verified schedules."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_h200_uses_standard_campaign_and_no_redundant_tools():
    from akas.optimization.context import workload
    from tools.campaign import modal_command

    root = Path(__file__).resolve().parents[1]
    assert not list((root / "tools").glob("h200_*.py"))
    spec_path = Path("benchmarks/campaigns/lfm2.5-24b-a2b-fp8-h200.json")
    spec = json.loads((root / spec_path).read_text())
    requested = workload(spec["workload"])
    assert requested["batches"] == [1, 4, 16, 64, 256]
    assert requested["prefix_tokens"] == [2048]
    assert requested["output_tokens"] == 128
    assert spec["formats"] == ["fp8"] and spec["caches"] == ["float16"]
    assert (root / spec["baseline_lock"]).is_file()
    assert not (root / "benchmarks/configs").exists()
    args = modal_command(
        spec_path,
        "/data/campaigns/test",
        budget="300s",
        slice_duration="300s",
        history="/data/standard-benchmarks",
        python="python",
    )
    assert args[:5] == ["python", "-u", "-m", "tools.campaign", "--spec"]
    assert "--no-autotune" not in args


def test_hopper_recipe_and_historical_seeding_match_verified_h200_schedules(monkeypatch):
    from akas.compilation.compile import plan
    from akas.compilation.policy import lower
    from akas.compilation.rewrites import token_feedback
    from akas.models.lfm2.model import LFM2MoE
    from akas.optimization import incumbents, planning
    from akas.optimization.context import Context, workload

    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "build/checkpoint"
    if not (checkpoint / "config.json").is_file():
        pytest.skip("local LFM2.5-24B-A2B-FP8 checkpoint metadata not present")
    monkeypatch.setattr(
        planning,
        "hardware",
        lambda provider, ordinal: {"name": "NVIDIA H200", "arch": "sm_90"},
    )
    ctx = Context(
        str(checkpoint),
        "akas.models.lfm2.model:LFM2MoE",
        workload("concurrent-prefill-v1"),
        backend="cuda",
        kv_dtype="float16",
        mode="ar",
    ).metadata()
    probe = planning.probe(ctx, "cuda")
    recipe = next(r for r in planning.recipes(ctx, [probe]) if r["graphs"])
    assert recipe["target"] == "sm_90a"
    assert recipe["chunk"] == 2048
    assert recipe["prefill_requests"] == 16
    geoms = planning.geometries(ctx["workload"], recipe)
    assert geoms == [
        (1, 1),
        (4, 1),
        (16, 1),
        (64, 1),
        (256, 1),
        (1, 2048),
        (4, 2048),
        (16, 2048),
    ]
    loaded = LFM2MoE.checkpoint_type(ctx["checkpoint"])
    graphs = []
    for b, t, *ph in geoms:
        purpose = ph[0] if ph else None
        composed = LFM2MoE.compose(
            loaded,
            rows=b,
            chunk=t,
            max_requests=256,
            max_context=2176,
            provider="cuda",
            phase=purpose,
            **recipe["model_options"],
        )
        graph, _ = lower(
            composed.graph,
            composed.commit,
            policy=recipe["policy"],
            provider="cuda",
            target="sm_90a",
        )
        graphs.append((purpose or ("decode" if t == 1 else "prefill"), graph))
    ckpt_sha = "dff2336439834c4ce4e2c6f0f7b6cd829987f7e9d57ec8074354e0b2cacb2830"
    profile, origins = incumbents.seed(
        root / "benchmarks",
        graphs,
        provider="cuda",
        target="sm_90a",
        checkpoint_sha256=ckpt_sha,
        hardware="NVIDIA H200",
    )
    assert profile is not None and len(origins) == 8
    schedules_path = (
        root / "benchmarks/reports/20261103T052243Z-8ea11a78fe72/SCHEDULES.json"
    )
    published = json.loads(schedules_path.read_text())["configurations"][
        "lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar"
    ]
    by_phase = {(p["phase"], p["batch"], p["chunk"]): p for p in published["phases"]}
    for (phase_name, graph), (b, t, *_) in zip(graphs, geoms):
        if phase_name == "decode":
            token_feedback(
                graph, graph.request["inputs"], graph.request["prediction"], b
            )
        if graph.phase == "forward":
            graph.phase = phase_name
        bundle = plan(graph, provider="cuda", target="sm_90a", schedules=profile)
        expected = by_phase[(phase_name, b, t)]["kernels"]
        actual = {k.key: k.schedule for k in bundle.kernels}
        assert set(actual) == set(expected)
        for key, sched in actual.items():
            assert sched["implementation"] == expected[key]["schedule"]["implementation"]


def test_early_phase_regression_gate_rejects_degraded_candidate(monkeypatch):
    from dataclasses import asdict
    import numpy as np
    from akas.optimization import evaluation

    class Session:
        allocated_bytes = 1

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    seen_batches = []

    def measured(engine, batch, prefix, output, **kwargs):
        seen_batches.append(batch)
        return dict(
            seconds=2.0,
            output_tokens_per_second=64.0,
            ttft_ms=50.0,
            tpot_ms=15.0,
            output_ids=[[1, 2, 3, 4]] * batch,
        )

    monkeypatch.setenv("AKAS_SKIP_AUTOTUNE", "1")
    monkeypatch.setattr(evaluation, "Session", Session)
    monkeypatch.setattr(evaluation, "diagnostic", lambda *a, **k: np.ones((3, 1, 4)))
    monkeypatch.setattr(
        evaluation,
        "Engine",
        lambda *a, **k: SimpleNamespace(
            scheduler=SimpleNamespace(stats={}), close=lambda: None
        ),
    )
    monkeypatch.setattr(evaluation, "measured_requests", measured)
    reference = dict(
        metadata=dict(run_id="incumbent"),
        results=[
            dict(
                case=dict(batch=b, prompt=2048),
                status="ok",
                output_ids=[[1, 2, 3, 4]] * b,
                serving=dict(seconds=1.0),
            )
            for b in (1, 4, 16)
        ],
    )
    with pytest.raises(ValueError, match="early phase regression gate failed"):
        evaluation.evaluate(
            SimpleNamespace(),
            dict(chunk=2048, window=None, prefill_requests=16),
            dict(batches=[1, 4, 16], prefix_tokens=[2048], output_tokens=4),
            reference=reference,
        )
    assert seen_batches == [1]
