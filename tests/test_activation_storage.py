"""Storage precision is explicit; additions and normalization retain FP32 work."""
import os
import pytest
from test_lfm_moe_native import test_native_operations_match_independent_references as check_operation


@pytest.mark.skipif(os.getenv("AKAS_LFM_FP8_CUDA") != "1", reason="CUDA storage precision qualification")
@pytest.mark.parametrize("kind,parameters", [
    ("add", dict(r=33, c=513, input_dtype="bfloat16", other_dtype="float32", output_dtype="bfloat16")),
    ("add", dict(r=33, c=513, input_dtype="float16", other_dtype="float16", output_dtype="float32")),
    ("rms", dict(r=4, c=2048, eps=1e-5, input_dtype="bfloat16", output_dtype="bfloat16", weight_dtype="float32", stage_values=True)),
    ("rms", dict(r=33, c=513, eps=1e-5, input_dtype="float16", output_dtype="float16", weight_dtype="float32")),
    ("add_rms", dict(r=4, c=2048, eps=1e-5, input_dtype="bfloat16", mixed_dtype="float32", residual_dtype="bfloat16", output_dtype="bfloat16", weight_dtype="float32", algorithm="serial-tree", stage_values=True)),
    ("add_rms", dict(r=33, c=513, eps=1e-5, input_dtype="float16", mixed_dtype="float32", residual_dtype="float16", output_dtype="float16", weight_dtype="float32", algorithm="serial-tree", stage_values=True)),
])
def test_typed_activation_operations_match_independent_equations(tmp_path, kind, parameters):
    check_operation(tmp_path, kind, parameters)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_typed_residual_fusion_keeps_each_declared_storage_boundary(dtype):
    from akas.graph import Graph
    from akas.compilation.policy import lower
    from akas.compilation.compile import plan

    graph = Graph(contract="reference")
    x = graph.buffer("x", (4 * 2048,), dtype, role="input")
    y = graph.buffer("mixed", (4 * 2048,), "float32", role="input")
    residual = graph.buffer("residual", x.shape, dtype, role="output")
    weight = graph.buffer("weight", (2048,), "float32", role="weight")
    out = graph.buffer("normalized", x.shape, dtype, role="output")
    graph("add", dict(r=4, c=2048, input_dtype=dtype, other_dtype="float32", output_dtype=dtype), x, y, residual)
    graph("rms", dict(r=4, c=2048, eps=1e-5, input_dtype=dtype, output_dtype=dtype), residual, weight, out)
    graph, _ = lower(graph, policy={"normalization": dict(fused_residual=True)})
    assert len(graph.calls) == 1
    call = graph.calls[0]
    assert call.kind == "add_rms"
    assert call.parameters["input_dtype"] == call.parameters["residual_dtype"] == call.parameters["output_dtype"] == dtype
    assert call.parameters["mixed_dtype"] == call.parameters["weight_dtype"] == "float32"
    plan(graph, target="sm_89")


@pytest.mark.skipif(os.getenv("AKAS_LFM_FP8_CUDA") != "1", reason="CUDA storage precision qualification")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_typed_residual_fusion_matches_separate_round_then_normalize(tmp_path, dtype):
    from contextlib import ExitStack
    import numpy as np
    import tensor
    from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16
    from akas.graph import Graph
    from akas.compilation.policy import lower
    from akas.compilation.compile import produce
    from akas.execution.forward import Forward

    graph = Graph(contract="reference")
    rows, width = 4, 513
    x = graph.buffer("x", (rows * width,), dtype, role="input")
    y = graph.buffer("mixed", x.shape, "float32", role="input")
    residual = graph.buffer("residual", x.shape, dtype, role="output")
    weight = graph.buffer("weight", (width,), "float32", role="weight")
    out = graph.buffer("normalized", x.shape, dtype, role="output")
    graph("add", dict(r=rows, c=width, input_dtype=dtype, other_dtype="float32", output_dtype=dtype), x, y, residual)
    graph("rms", dict(r=rows, c=width, eps=1e-5, input_dtype=dtype, output_dtype=dtype,
                      algorithm="serial-tree", stage_values=True), residual, weight, out)
    fused, _ = lower(Graph.from_metadata(graph.metadata()), policy={"normalization": dict(fused_residual=True)})
    rng = np.random.default_rng(9157)
    host = dict(x=rng.normal(0, .5, x.shape).astype("float32"),
                mixed=rng.normal(0, .5, y.shape).astype("float32"),
                weight=rng.uniform(.2, 1.2, weight.shape).astype("float32"))
    round_storage = ((lambda v: v.astype("float16").astype("float32")) if dtype == "float16"
                     else (lambda v: decode_bfloat16(encode_bfloat16(v))))
    host["x"] = round_storage(host["x"])
    residual_expected = round_storage(host["x"] + host["mixed"])
    wide = residual_expected.reshape(rows, width).astype("float64")
    expected = round_storage((wide / np.sqrt((wide * wide).mean(-1, keepdims=True) + 1e-5)
                              * host["weight"]).reshape(-1))
    results = []
    with tensor.Device() as device, ExitStack() as lifetime:
        buffers = {name: device.from_numpy(value, dtype=graph.buffers[name].dtype) for name, value in host.items()}
        buffers.update(residual=device.zeros(x.shape, dtype), normalized=device.zeros(x.shape, dtype))
        for value in buffers.values():
            lifetime.callback(value.release)
        for name, candidate in (("separate", graph), ("fused", fused)):
            destination = tmp_path / name
            produce(candidate, destination, search=False, target="sm_89")
            with Forward(destination, device, buffers, graphs=False) as forward:
                forward.run()
                results.append({key: buffers[key].to_numpy() for key in ("residual", "normalized")})
        np.testing.assert_array_equal(results[0]["residual"], residual_expected)
        np.testing.assert_allclose(results[0]["normalized"], expected, rtol=.002, atol=.0002)
        for key in results[0]:
            np.testing.assert_array_equal(results[0][key], results[1][key])
