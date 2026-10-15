"""Operation policies preserve optimized physical pipelines in the common compiler."""

import os
import pytest


def projection_graph():
    from akas.graph import Graph

    graph = Graph(contract="block-fp8", phase="decode")
    x = graph.buffer("x", (4, 256), "bfloat16", role="input")
    w = graph.buffer("w", (128, 256), "uint8", role="weight")
    scales = graph.buffer("scales", (1, 2), "bfloat16", role="weight")
    for name in ("a", "b"):
        out = graph.buffer(name, (4, 128), "float32", role="output")
        graph("fp8_linear", dict(r=4, k=256, o=128), x=x, w=w, scales=scales, out=out)
    return graph


def test_projection_policy_declares_shared_quantization_and_split_merge():
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    graph, _ = lower(projection_graph(), policy={"linear": {"partitions": 2}})
    assert [call.kind for call in graph.calls] == [
        "quantize",
        "projection_partial",
        "projection_merge",
        "projection_partial",
        "projection_merge",
    ]
    assert graph.buffers["pipeline.partial.0"].shape == (4, 2, 128)
    plan(graph)


def test_recompute_policy_keeps_operands_and_removes_full_state_snapshots():
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan
    from akas.models.qwen35._composition import compose_moe_program
    from test_qwen_graph import moe_checkpoint

    graph, commit = compose_moe_program(
        moe_checkpoint(),
        rows=2,
        chunk=4,
        max_requests=4,
        max_context=32,
        phase="verify",
    )
    graph, commit = lower(graph, commit, policy={"gdn": {"commit": "recompute"}})
    assert not any(name.startswith("verify.saved") for name in graph.buffers)
    assert [call.kind for call in commit.calls] == ["gdn_conv", "gdn_scan", "advance"]
    for call in commit.calls[:2]:
        assert call.parameters.get("commit", True)
    assert all(
        call.parameters.get("commit") is False
        for call in graph.calls
        if call.kind in ("gdn_conv", "gdn_scan")
    )
    assert graph.buffers["verify.operands.state.0.1.x"].role == "journal"
    plan(graph)
    plan(commit)


@pytest.mark.skipif(
    os.environ.get("TENSOR_QWEN_CUDA") != "1",
    reason="native physical pipeline qualification",
)
def test_split_projection_pipeline_is_numerically_qualified_by_default(tmp_path):
    from akas.compilation.policy import lower
    from akas.compilation.compile import produce

    graph, _ = lower(projection_graph(), policy={"linear": {"partitions": 2}})
    manifest = produce(graph, tmp_path)
    assert manifest["tuning"]["operations"]
    assert all(row["selected"] for row in manifest["tuning"]["operations"])


def expert_graph():
    from akas.graph import Graph

    graph = Graph(contract="block-fp8", phase="prefill")
    specs = {
        "x": ((4, 256), "uint8"),
        "activation_scales": ((4, 2), "float32"),
        "w": ((256, 128, 256), "uint8"),
        "scales": ((256, 1, 2), "bfloat16"),
        "counts": ((256,), "int32"),
        "routes": ((256, 4), "int32"),
        "out": ((4, 8, 128), "float32"),
    }
    buffers = {
        name: graph.buffer(
            name, shape, dtype, role="output" if name == "out" else "input"
        )
        for name, (shape, dtype) in specs.items()
    }
    graph(
        "fp8_experts",
        dict(r=4, k=256, o=128, experts=256, top=8, routed_input=False),
        **buffers,
    )
    return graph


@pytest.mark.parametrize("settings", [{"compact": True}, {"partitions": 2}])
def test_routed_policy_declares_physical_expert_pipeline(settings):
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    graph, _ = lower(expert_graph(), policy={"moe": settings})
    kinds = [call.kind for call in graph.calls]
    assert "expert_partial" in kinds
    assert ("expert_tiles" in kinds) == settings.get("compact", False)
    assert ("expert_merge" in kinds) == (settings.get("partitions", 1) > 1)
    plan(graph)


@pytest.mark.skipif(
    os.environ.get("TENSOR_QWEN_CUDA") != "1",
    reason="native routed pipeline qualification",
)
@pytest.mark.parametrize("settings", [{"compact": True}, {"partitions": 2}])
def test_routed_pipeline_is_numerically_qualified_by_default(tmp_path, settings):
    from akas.compilation.policy import lower
    from akas.compilation.compile import produce

    graph, _ = lower(expert_graph(), policy={"moe": settings})
    manifest = produce(graph, tmp_path)
    assert all(row["selected"] for row in manifest["tuning"]["operations"])


def test_resident_expert_storage_policy_is_uniform_across_all_model_phases():
    from akas.compilation.policy import lower
    from akas.models.qwen35._composition import compose_moe_program
    from test_moe_speculative_program import checkpoint

    loaded = checkpoint()
    settings = {
        "hopper": True,
        "resident_weights": True,
        "bf16_mma": True,
        "mma_reorder": True,
        "mma_reduction": 32,
    }
    conversions = None
    for rows, chunk, phase in (
        (2, 1, None),
        (2, 4, None),
        (2, 4, "verify"),
        (2, 1, "proposal"),
        (2, 4, "proposal"),
    ):
        composed = compose_moe_program(
            loaded, rows=rows, chunk=chunk, max_requests=4, max_context=32, phase=phase
        )
        graph, commit = composed if isinstance(composed, tuple) else (composed, None)
        graph, commit = lower(graph, commit, policy={"moe": settings}, target="sm_90a")
        weights = {
            name: spec for name, spec in graph.buffers.items() if spec.role == "weight"
        }
        options = loaded.program_upload_options(weights, model_options={})
        assert options["conversions"]
        assert all(dtype == "float16" for dtype in options["conversions"].values())
        if phase != "proposal":
            if conversions is None:
                conversions = options["conversions"]
            else:
                assert conversions == options["conversions"]
        assert all(
            graph.buffers[call.bindings["w"]].dtype == "float16"
            for call in graph.calls
            if call.kind == "expert_partial"
        )


def fixed_expert_graph():
    from akas.graph import Graph

    graph = Graph(contract="block-fp8", phase="decode")
    x = graph.buffer("x", (4, 256), "bfloat16", role="input")
    w = graph.buffer("w", (256, 128, 256), "uint8", role="weight")
    scales = graph.buffer("scales", (256, 1, 2), "bfloat16", role="weight")
    ids = graph.buffer("ids", (4, 8), "int32", role="input")
    experts = graph.buffer("experts", (32,), "int32")
    routes = graph.buffer("routes", (32, 4), "int32")
    out = graph.buffer("out", (4, 8, 128), "float32", role="output")
    graph("moe_groups", dict(r=4, top=8), ids, experts, routes)
    graph(
        "fp8_experts",
        dict(r=4, k=256, o=128, experts=256, top=8, routed_input=False),
        x,
        w,
        scales,
        experts,
        routes,
        out,
    )
    return graph


def test_fixed_row_group_policy_preserves_native_tensor_core_layout():
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    graph, _ = lower(
        fixed_expert_graph(), policy={"moe": {"fixed_rows": True, "partitions": 2}}
    )
    assert [call.kind for call in graph.calls] == [
        "moe_groups",
        "quantize",
        "expert_partial",
        "expert_merge",
    ]
    projection = graph.calls[2]
    assert "experts" in projection.bindings and "counts" not in projection.bindings
    assert graph.buffers[projection.bindings["routes"]].shape == (32, 4)
    plan(graph)


@pytest.mark.skipif(
    os.environ.get("TENSOR_QWEN_CUDA") != "1",
    reason="native fixed group pipeline qualification",
)
def test_fixed_row_group_pipeline_is_numerically_qualified_by_default(tmp_path):
    from akas.compilation.policy import lower
    from akas.compilation.compile import produce

    graph, _ = lower(
        fixed_expert_graph(), policy={"moe": {"fixed_rows": True, "partitions": 2}}
    )
    manifest = produce(graph, tmp_path)
    assert all(row["selected"] for row in manifest["tuning"]["operations"])


@pytest.mark.skipif(
    os.environ.get("TENSOR_QWEN_CUDA") != "1",
    reason="native large batch projection qualification",
)
@pytest.mark.parametrize("rows", [16, 256])
def test_large_bf16_projection_qualifies_tiled_and_scalar_choices(tmp_path, rows):
    from akas.graph import Graph
    from akas.compilation.compile import produce

    graph = Graph(contract="block-fp8", phase="decode")
    x = graph.buffer("x", (rows, 64), "bfloat16", role="input")
    w = graph.buffer("w", (32, 64), "bfloat16", role="weight")
    out = graph.buffer("out", (rows, 32), "float32", role="output")
    graph("bf16_linear", dict(r=rows, k=64, o=32), x, w, out)
    result = produce(graph, tmp_path)
    assert result["tuning"]["operations"][0]["selected"]
