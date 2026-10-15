"""Packed optimizations belong to operations and preserve published intermediates."""

import os

import numpy as np
import pytest

from akas.graph import Graph
from akas.ops.linear.definition import fuse_projections
from akas.ops.linear.definition import fuse_residual_projection
from akas.ops.normalization.definition import fuse_residual_normalization


def projection_graph(encoding=2, rows=1):
    graph = Graph(contract="reference")
    x = graph.buffer("x", (rows * 256,), "float32", role="input")
    w = graph.buffer("w", (512,), "uint8", role="weight")
    gate = graph.buffer("gate", (rows * 64,), "float32")
    up = graph.buffer("up", (rows * 64,), "float32")
    out = graph.buffer("out", (rows * 64,), "float32", role="output")
    parameters = dict(r=rows, k=256, o=64, type=encoding, ggml=True, optimized=True)
    graph("linear", parameters, x, w, gate)
    graph("linear", parameters, x, w, up)
    graph("swiglu", dict(r=rows, c=64), gate, up, out)
    return graph


@pytest.mark.parametrize("encoding,paired", [(0, False), (1, False), (2, True), (12, True), (14, True)])
def test_decode_pair_preserves_quantized_and_float_schedule_policy(encoding, paired):
    graph = fuse_projections(projection_graph(encoding))
    assert [call.kind for call in graph.calls] == (["ffn"] if paired else ["linear", "linear", "swiglu"])


def test_residual_projection_keeps_observable_projection_and_view():
    graph = projection_graph()
    projection = graph.calls[0]
    residual = graph.buffer("residual", (64,), "float32", role="input")
    out = graph.buffers["out"]
    graph.calls = [projection]
    graph("add", dict(r=1, c=64), graph.buffers["gate"], residual, out)
    original = Graph.from_metadata(graph.metadata())
    fused = fuse_residual_projection(graph)
    assert [call.kind for call in fused.calls] == ["linear_add"]
    assert fused.calls[0].arguments == ("x", "w", "residual", "out")
    # A borrowed view means the separate projection remains observable.
    original.view("gate_view", original.buffers["gate"], (1, 64))
    assert [call.kind for call in fuse_residual_projection(original).calls] == ["linear", "add"]


def norm_graph():
    graph = Graph(contract="reference")
    x = graph.buffer("x", (256,), "float32", role="input")
    mixed = graph.buffer("mixed", (256,), "float32", role="input")
    residual = graph.buffer("residual", (256,), "float32", role="output")
    w = graph.buffer("w", (256,), "float32", role="weight")
    out = graph.buffer("out", (256,), "float32", role="output")
    graph("add", dict(r=1, c=256), x, mixed, residual)
    graph("rms", dict(r=1, c=256, eps=1e-6), residual, w, out)
    return graph


def test_normalization_fusion_preserves_residual_and_excludes_aliased_update():
    graph = fuse_residual_normalization(norm_graph())
    assert graph.calls[0].arguments == ("x", "mixed", "residual", "w", "out")
    graph = norm_graph()
    from dataclasses import replace
    graph.buffers["residual"] = replace(graph.buffers["residual"], alias_of="x", role="input")
    assert len(fuse_residual_normalization(graph).calls) == 2


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_native_fused_residual_normalization_matches_separate_outputs(tmp_path, provider):
    if os.environ.get("AKAS_FORWARD_" + provider.upper()) != "1":
        pytest.skip("enable native fusion qualification")
    import tensor
    from akas.compilation.compile import produce
    from akas.execution.forward import Forward
    from akas.compilation.tuning import SearchSettings

    graph = fuse_residual_normalization(norm_graph())
    produce(graph, tmp_path, provider=provider, search_settings=SearchSettings(budget=1, samples=1, repeats=1))
    device = tensor.Device(provider=provider).__enter__()
    rng = np.random.default_rng(17)
    host = {name: rng.normal(0, .1, (256,)).astype("float32") for name in ("x", "mixed", "w")}
    buffers = {name: device.from_numpy(value) for name, value in host.items()}
    buffers.update({name: device.empty((256,), "float32") for name in ("residual", "out")})
    executor = Forward(tmp_path, device, buffers)
    try:
        executor.run()
        expected = host["x"] + host["mixed"]
        np.testing.assert_allclose(executor.buffers["residual"].to_numpy(), expected, rtol=1e-6, atol=1e-7)
        expected_out = expected / np.sqrt(np.mean(expected * expected) + 1e-6) * host["w"]
        np.testing.assert_allclose(executor.buffers["out"].to_numpy(), expected_out, rtol=2e-5, atol=1e-7)
    finally:
        executor.close()
        for buffer in buffers.values():
            buffer.release()
        device.__exit__(None, None, None)
