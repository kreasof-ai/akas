"""Single-plane packing has an independent recipe under the common compiler."""

import os
import numpy as np
import pytest


def test_single_plane_reference_preserves_exact_integer_sums_and_finite_scales():
    from akas.quantization.activations.q8.validation import validation

    specs = [
        dict(name="x", shape=[256], dtype="float32"),
        dict(name="out", shape=[64], dtype="uint32"),
        dict(name="scales", shape=[8], dtype="float32"),
        dict(name="sums", shape=[8], dtype="int32"),
    ]
    case = validation("quantize_q8", dict(k=256), specs)
    quants = case.expected["out"].view("int8").reshape(8, 32)
    np.testing.assert_array_equal(
        case.expected["sums"], quants.astype("int32").sum(axis=1)
    )
    assert np.isfinite(case.expected["scales"]).all()


@pytest.mark.skipif(
    os.environ.get("AKAS_FORWARD_WEBGPU") != "1",
    reason="native WebGPU codec qualification",
)
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
    report = produce(
        graph,
        tmp_path,
        provider="webgpu",
        search_settings=SearchSettings(budget=1, samples=1, repeats=1),
    )
    assert report["tuning"]["operations"][0]["selected"]["score"] > 0


@pytest.mark.skipif(
    os.getenv("AKAS_FORWARD_CUDA") != "1", reason="CUDA Q8 codec qualification"
)
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_cuda_q8_exact_packing_ties_zero_and_partial_warp_group(tmp_path, dtype):
    import tensor
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings
    from akas.execution.forward import Forward
    from akas.quantization.activations.q8.validation import validation

    g = Graph(contract="reference")
    specs = [
        ("x", 288, dtype),
        ("out", 72, "uint32"),
        ("scales", 9, "float32"),
        ("sums", 9, "int32"),
    ]
    arguments = [
        g.buffer(n, (count,), dt, role="input" if n == "x" else "output")
        for n, count, dt in specs
    ]
    p = dict(r=3, k=96, input_dtype=dtype)
    g("quantize_q8", p, *arguments)
    signature = [dict(name=n, shape=[count], dtype=dt) for n, count, dt in specs]
    reference = validation("quantize_q8", p, signature)
    with tensor.Device() as device:
        produce(
            g,
            tmp_path,
            device=device,
            search_settings=SearchSettings(budget=1, samples=1, repeats=1),
        )
        buffers = {n: device.from_numpy(reference.arguments[n]) for n, _, _ in specs}
        try:
            with Forward(tmp_path, device, buffers) as forward:
                forward.run()
                for name in ("out", "scales", "sums"):
                    np.testing.assert_array_equal(
                        buffers[name].to_numpy(), reference.expected[name]
                    )
        finally:
            for buffer in buffers.values():
                buffer.release()
