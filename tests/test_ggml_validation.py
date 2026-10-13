"""Packed weights participate in common numerical qualification and search."""

import os

import numpy as np
import pytest


@pytest.mark.parametrize("kind", [2, 8, 12, 14])
@pytest.mark.parametrize("dtype", ["uint8", "uint32"])
def test_weight_recipe_has_finite_nonconstant_values_and_preserves_storage(kind, dtype):
    from akas.checkpoints.gguf import TYPES, dequantize
    from akas.quantization.weights.ggml import sample_weight

    count = 16 * 256 // TYPES[kind][1] * TYPES[kind][2]
    storage = np.zeros(count // np.dtype(dtype).itemsize, dtype=dtype)
    values = sample_weight(storage, kind)
    assert values.shape == (4096,)
    assert np.isfinite(values).all() and np.std(values) > 0
    np.testing.assert_array_equal(values, dequantize(storage.view("uint8"), kind))


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("encoding", [2, 8, 12, 14])
@pytest.mark.parametrize("rows", [1, 4])
def test_packed_projection_and_embedding_use_default_common_search(tmp_path, provider, encoding, rows):
    if os.environ.get("AKAS_FORWARD_" + provider.upper()) != "1":
        pytest.skip("enable native packed qualification")
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings
    from akas.quantization.weights.ggml import weight_storage

    graph = Graph(contract="packed")
    count, dtype = weight_storage(encoding, 256 * 16, packed_words=provider == "webgpu")
    x = graph.buffer("x", (rows * 256,), "float32", role="input")
    weight = graph.buffer("weight", (count,), dtype, role="weight")
    out = graph.buffer("out", (rows * 16,), "float32", role="output")
    tokens = graph.buffer("tokens", (rows,), "int32", role="input")
    embedded = graph.buffer("embedded", (rows * 256,), "float32", role="output")
    graph("linear", dict(r=rows, k=256, o=16, type=encoding), x, weight, out)
    graph("embedding", dict(r=rows, c=256, v=16, type=encoding), tokens, weight, embedded)
    manifest = produce(graph, tmp_path, provider=provider,
                       search_settings=SearchSettings(budget=2, samples=1, repeats=1))
    assert len(manifest["tuning"]["operations"]) == 2
    for operation in manifest["tuning"]["operations"]:
        assert operation["selected"]["score"] > 0
        assert operation["selected"]["measurements"]
        assert any(candidate["status"] == "passed" for candidate in operation["candidates"])
