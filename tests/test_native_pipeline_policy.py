"""Operation policies preserve optimized physical pipelines in the common compiler."""

import os
import pytest


def projection_graph(contract="block-fp8", input_dtype="bfloat16"):
    from akas.graph import Graph

    graph = Graph(contract=contract, phase="decode")
    x = graph.buffer("x", (4, 256), input_dtype, role="input")
    w = graph.buffer("w", (128, 256), "uint8", role="weight")
    scales = graph.buffer("scales", (1, 2), "bfloat16", role="weight")
    for name in ("a", "b"):
        out = graph.buffer(name, (4, 128), "float32", role="output")
        parameters = dict(r=4, k=256, o=128)
        if input_dtype != "bfloat16":
            parameters["input_dtype"] = input_dtype
        graph("fp8_linear", parameters, x=x, w=w, scales=scales, out=out)
    return graph


@pytest.mark.parametrize("contract,input_dtype", [
    ("block-fp8", "bfloat16"), ("reference", "float32"),
])
def test_projection_policy_declares_shared_quantization_and_split_merge(contract, input_dtype):
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    graph, _ = lower(projection_graph(contract, input_dtype), policy={"linear": {"partitions": 2}})
    assert [call.kind for call in graph.calls] == [
        "quantize",
        "projection_partial",
        "projection_merge",
        "projection_partial",
        "projection_merge",
    ]
    assert graph.buffers["pipeline.partial.0"].shape == (4, 2, 128)
    assert graph.calls[0].parameters.get("input_dtype", "bfloat16") == input_dtype
    plan(graph)


def test_projection_pipeline_can_retain_small_batch_gemv_dispatch():
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    graph, _ = lower(projection_graph("reference", "float32"),
                     policy={"linear": {"partitions": 1, "minimum_rows": 9}})
    assert [call.kind for call in graph.calls] == ["fp8_linear", "fp8_linear"]
    assert all(kernel.schedule["implementation"] == "linear.cuda.fp8-gemv:reference"
               for kernel in plan(graph).kernels)


@pytest.mark.parametrize("rows,depth,columns,parts", [
    (4, 2048, 11776, 1), (4, 11776, 2048, 4), (4, 2048, 2048, 4),
    (4, 2048, 512, 8), (8192, 2048, 512, 1),
])
def test_projection_policy_splits_underoccupied_launches_with_legal_k_ranges(rows, depth, columns, parts):
    from akas.graph import Graph
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    graph = Graph(contract="reference", phase="forward")
    x = graph.buffer("x", (rows, depth), "float32", role="input")
    w = graph.buffer("w", (columns, depth), "uint8", role="weight")
    scale = graph.buffer("scale", (columns // 128, depth // 128), "bfloat16", role="weight")
    out = graph.buffer("out", (rows, columns), "float32", role="output")
    graph("fp8_linear", dict(r=rows, k=depth, o=columns, input_dtype="float32"),
          x=x, w=w, scales=scale, out=out)
    graph, _ = lower(graph, policy={"linear": dict(partitions=8, minimum_split_groups=64,
                     columns=128, block_m=32)})
    projection = graph.calls[1]
    assert projection.parameters["partitions"] == parts
    assert "minimum_split_groups" not in projection.parameters
    assert (graph.calls[-1].kind == "projection_merge") == (parts > 1)
    assert graph.buffers[projection.bindings["out"]].shape == (
        (rows, columns) if parts == 1 else (rows, parts, columns))
    plan(graph)


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5])
def test_projection_policy_rejects_invalid_split_group_targets(minimum):
    from akas.compilation.policy import lower

    with pytest.raises(ValueError, match="minimum split groups"):
        lower(projection_graph("reference", "float32"),
              policy={"linear": dict(partitions=8, minimum_split_groups=minimum)})


@pytest.mark.parametrize("minimum,kind", [(2, "matmul"), (9, "linear")])
def test_floating_and_fp8_projection_thresholds_are_independent(minimum, kind):
    from akas.compilation.policy import lower

    graph, _ = lower(floating_projection_graph(4, "float32"), policy={"linear": dict(
        floating_matmul=True, minimum_rows=2, floating_matmul_minimum_rows=minimum)})
    assert graph.calls[0].kind == kind


@pytest.mark.parametrize("maximum", [64, 128])
@pytest.mark.parametrize("kind", ["expert_partial", "fp8_expert_swiglu"])
def test_shared_route_maps_allow_m_tuning_without_changing_storage_contract(kind, maximum):
    from akas.kernels.definition import define
    from tensor.compiler.frontend.signature import frontend_arguments

    p = dict(r=33, rows=33, top=3, experts=7, o=128, k=384,
             routed_input=False, partitions=1, compact=True, tile_map=True, max_block_m=maximum)
    signatures = [frontend_arguments(define(dict(kind=kind, parameters=p, contract="reference"),
                     schedule=dict(block_m=m, threads=128,
                                   **({"columns": 128} if kind == "expert_partial" else {}))), {})
                  for m in (16, 32, 64, *((128,) if maximum == 128 else ()))]
    assert all(signature == signatures[0] for signature in signatures)
    maps = [a for a in signatures[0] if a["name"] in ("tile_experts", "tile_offsets")]
    assert len(maps) == 2 and all(a["shape"][0] == (5 if maximum == 64 else 6) for a in maps)


@pytest.mark.parametrize("maximum", [True, 3, 12, 256, 128.0])
def test_shared_expert_map_rejects_unsupported_row_domains(maximum):
    from akas.ops.moe.definition import tile_layouts

    with pytest.raises(ValueError, match="maximum row tile"):
        tile_layouts({"max_block_m": maximum})


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5])
def test_expert_quantization_fusion_requires_positive_integer_threshold(minimum):
    from akas.compilation.policy import lower

    with pytest.raises(ValueError, match="quantization minimum rows"):
        lower(projection_graph("reference", "float32"), policy={"moe": {
            "fused_swiglu": True, "fused_quantize_minimum_rows": minimum}})


def test_projection_quantization_is_invalidated_by_a_positional_alias_write():
    from akas.compilation.policy import lower

    graph = projection_graph("reference", "float32")
    first, second = graph.calls
    graph.calls = [first]
    alias = graph.view("alias", graph.buffers["x"], (4 * 256,))
    weight = graph.buffer("norm", (256,), "float32", role="weight")
    graph("rms", dict(r=4, c=256, eps=1e-6), alias, weight, alias)
    graph.calls.append(second)
    graph, _ = lower(graph, policy={"linear": {"partitions": 1}})
    assert [call.kind for call in graph.calls] == [
        "quantize", "projection_partial", "rms", "quantize", "projection_partial",
    ]


def test_projection_quantization_reuses_same_geometry_through_nested_aliases():
    from dataclasses import replace
    from akas.compilation.policy import lower, candidates
    from akas.compilation.compile import plan

    graph = projection_graph("reference", "float32")
    flat = graph.view("flat", graph.buffers["x"], (1024,))
    alias = graph.view("nested", flat, (4, 256))
    graph.calls[1] = replace(graph.calls[1], bindings={**graph.calls[1].bindings, "x": alias.name})
    assert {"linear": {"partitions": 1, "minimum_rows": 2}} in candidates(graph, provider="cuda", target="sm_89")
    assert {"linear": {"partitions": 8, "minimum_rows": 2, "minimum_split_groups": 64}} in candidates(
        graph, provider="cuda", target="sm_89")
    graph, _ = lower(graph, policy={"linear": {"partitions": 1}})
    assert [call.kind for call in graph.calls] == ["quantize", "projection_partial", "projection_partial"]
    assert graph.calls[1].bindings["x"] == graph.calls[2].bindings["x"]
    plan(graph)


def test_projection_quantization_keeps_distinct_alias_row_geometries():
    from akas.graph import Graph
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    graph = Graph(contract="reference", phase="decode")
    source = graph.buffer("source", (1024,), "float32", role="input")
    for rows, depth in ((4, 256), (8, 128)):
        x = graph.view(f"x{rows}", source, (rows, depth))
        w = graph.buffer(f"w{rows}", (128, depth), "uint8", role="weight")
        scales = graph.buffer(f"scales{rows}", (1, depth // 128), "bfloat16", role="weight")
        out = graph.buffer(f"out{rows}", (rows, 128), "float32", role="output")
        graph("fp8_linear", dict(r=rows, k=depth, o=128, input_dtype="float32"),
              x=x, w=w, scales=scales, out=out)
    graph, _ = lower(graph, policy={"linear": {"partitions": 1}})
    assert [call.kind for call in graph.calls].count("quantize") == 2
    plan(graph)


@pytest.mark.skipif(os.environ.get("AKAS_LFM_FP8_CUDA") != "1", reason="native FP8 qualification")
def test_reference_projection_pipeline_preserves_f32_inputs_and_split_results(tmp_path):
    import numpy as np
    import tensor
    from akas.compilation.policy import lower
    from akas.compilation.compile import produce
    from akas.execution.forward import Forward
    from akas.kernels.registry import validation
    from akas.kernels.definition import define
    from tensor.compiler.frontend.signature import frontend_arguments

    original = projection_graph("reference", "float32")
    parameters = dict(r=4, k=256, o=128, input_dtype="float32")
    kernel = define(dict(kind="fp8_linear", contract="reference", parameters=parameters))
    case = validation("fp8_linear", parameters, frontend_arguments(kernel, {}))
    graph, _ = lower(original, policy={"linear": {"partitions": 2}})
    produce(graph, tmp_path, search=False)
    with tensor.Device() as device:
        supplied = {
            name: device.from_numpy(case.arguments[name], dtype=spec.dtype)
            for name, spec in graph.buffers.items() if spec.role in ("input", "weight")
        }
        supplied.update({name: device.zeros((4, 128)) for name in ("a", "b")})
        try:
            with Forward(tmp_path, device, supplied) as forward:
                forward.run()
                for name in ("a", "b"):
                    np.testing.assert_allclose(forward.buffers[name].to_numpy(),
                                               case.expected["out"], rtol=.02, atol=.002)
        finally:
            for value in supplied.values():
                value.release()


def floating_projection_graph(rows, weight_dtype="float16", *, named=False):
    from akas.graph import Graph

    graph = Graph(contract="reference", phase="prefill")
    x = graph.buffer("x", (rows * 256,), "float32", role="input")
    w = graph.buffer("w", (128 * 256,), weight_dtype, role="weight")
    out = graph.buffer("out", (rows * 128,), "float32", role="output")
    p = dict(r=rows, k=256, o=128, optimized=True, weight_dtype=weight_dtype)
    if named:
        graph("linear", p, x=x, w=w, out=out)
    else:
        graph("linear", p, x, w, out)
    return graph


@pytest.mark.parametrize("rows", [1, 128])
@pytest.mark.parametrize("named", [False, True])
def test_floating_projection_policy_uses_canonical_matmul_views(rows, named):
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    graph, _ = lower(floating_projection_graph(rows, "float32", named=named),
                     policy={"linear": {"floating_matmul": True}})
    assert graph.calls[0].kind == ("matmul" if rows > 8 else "linear")
    if rows > 8:
        assert graph.calls[0].parameters["reduction_lanes"] == 32
        assert all(graph.buffers[name].alias_of is not None
                   for name in graph.calls[0].bindings.values())
    plan(graph)


@pytest.mark.skipif(os.environ.get("AKAS_CUDA") != "1", reason="native CUDA execution")
@pytest.mark.parametrize("rows,depth,columns", [(33, 65, 19), (2048, 2048, 64)])
def test_canonical_f32_projection_preserves_legacy_warp_reduction_bitwise(tmp_path, rows, depth, columns):
    import numpy as np
    import tensor
    from akas.graph import Graph
    from akas.compilation.policy import lower
    from akas.compilation.compile import produce
    from akas.execution.forward import Forward

    graph = Graph(contract="reference", phase="prefill")
    x = graph.buffer("x", (rows * depth,), "float32", role="input")
    w = graph.buffer("w", (columns * depth,), "float32", role="weight")
    old = graph.buffer("old", (rows * (columns + 1),), "float32", role="output")
    out = graph.buffer("out", (rows * columns,), "float32", role="output")
    graph("linear", dict(r=rows, k=depth, o=columns, output_stride=columns + 1), x, w, old)
    graph("linear", dict(r=rows, k=depth, o=columns), x=x, w=w, out=out)
    graph, _ = lower(graph, policy={"linear": {"floating_matmul": True}})
    assert [call.kind for call in graph.calls] == ["linear", "matmul"]
    produce(graph, tmp_path, search=False)
    rng = np.random.default_rng(91)
    values = rng.normal(size=(rows, depth)).astype("float32")
    weights = rng.normal(size=(columns, depth)).astype("float32")
    with tensor.Device() as device:
        supplied = dict(x=device.from_numpy(values.reshape(-1)), w=device.from_numpy(weights.reshape(-1)),
                        old=device.zeros(old.shape), out=device.zeros(out.shape))
        try:
            with Forward(tmp_path, device, supplied) as forward:
                forward.run()
                actual = supplied["out"].to_numpy().reshape(rows, columns)
                legacy = supplied["old"].to_numpy().reshape(rows, columns + 1)
                np.testing.assert_array_equal(actual, legacy[:, :columns])
                np.testing.assert_array_equal(legacy[:, columns], np.zeros(rows))
                expected = values.astype("float64") @ weights.astype("float64").T
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-5)
                assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-6
        finally:
            for value in supplied.values():
                value.release()


@pytest.mark.parametrize("rows,staged", [(2, False), (128, True)])
def test_cuda_input_staging_uses_existing_mma_operand_precision(rows, staged):
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    graph, _ = lower(floating_projection_graph(rows), policy={"linear": {"stage_inputs": True}},
                     provider="cuda", target="sm_89")
    assert (len(graph.calls) == 2) == staged
    assert graph.calls[-1].parameters.get("input_dtype", "float32") == (
        "float16" if staged else "float32")
    plan(graph)


@pytest.mark.skipif(os.environ.get("AKAS_LFM_FP8_CUDA") != "1", reason="native input staging qualification")
def test_cuda_staged_projection_preserves_original_inputs_and_outputs(tmp_path):
    import numpy as np
    import tensor
    from akas.compilation.policy import lower
    from akas.compilation.compile import produce
    from akas.execution.forward import Forward

    graph, _ = lower(floating_projection_graph(128), policy={"linear": {"stage_inputs": True}},
                     provider="cuda", target="sm_89")
    produce(graph, tmp_path, search=False)
    rng = np.random.default_rng(41)
    x = rng.normal(0, .15, (128, 256)).astype("float32")
    w = rng.normal(0, .15, (128, 256)).astype("float16")
    expected = x.astype("float16").astype("float64") @ w.astype("float64").T
    with tensor.Device() as device:
        supplied = dict(x=device.from_numpy(x.reshape(-1)), w=device.from_numpy(w.reshape(-1)),
                        out=device.zeros((128 * 128,)))
        try:
            with Forward(tmp_path, device, supplied) as forward:
                forward.run()
                np.testing.assert_allclose(supplied["out"].to_numpy().reshape(128, 128),
                                           expected, rtol=.002, atol=.002)
                np.testing.assert_array_equal(supplied["x"].to_numpy(), x.reshape(-1))
        finally:
            for value in supplied.values():
                value.release()


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
