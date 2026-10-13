"""Single-plane packing has an independent recipe under the common compiler."""

import os
import numpy as np
import pytest


def test_single_plane_reference_preserves_exact_integer_sums_and_finite_scales():
    from akas.quantization.activations.q8 import validation
    specs = [dict(name="x", shape=[256], dtype="float32"),
             dict(name="out", shape=[64], dtype="uint32"),
             dict(name="scales", shape=[8], dtype="float32"),
             dict(name="sums", shape=[8], dtype="int32")]
    case = validation("quantize_q8", dict(k=256), specs)
    quants = case.expected["out"].view("int8").reshape(8, 32)
    np.testing.assert_array_equal(case.expected["sums"], quants.astype("int32").sum(axis=1))
    assert np.isfinite(case.expected["scales"]).all()


@pytest.mark.skipif(os.environ.get("AKAS_FORWARD_WEBGPU") != "1", reason="native WebGPU codec qualification")
def test_single_plane_packing_qualifies_through_default_search(tmp_path):
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings
    graph = Graph(contract="reference")
    x = graph.buffer("x", (256,), "float32", role="input")
    out = graph.buffer("packed", (64,), "uint32", role="output")
    scales = graph.buffer("scales", (8,), "float32", role="output")
    sums = graph.buffer("sums", (8,), "int32", role="output")
    graph("quantize_q8", dict(k=256), x, out, scales, sums)
    report = produce(graph, tmp_path, provider="webgpu", search_settings=SearchSettings(budget=1, samples=1, repeats=1))
    assert report["tuning"]["operations"][0]["selected"]["score"] > 0
