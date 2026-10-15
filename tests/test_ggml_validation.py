"""Packed weights participate in common numerical qualification and search."""

import os

import numpy as np
import pytest


@pytest.mark.parametrize("kind", [2, 3, 8, 12, 13, 14])
@pytest.mark.parametrize("dtype", ["uint8", "uint32"])
def test_weight_recipe_has_finite_nonconstant_values_and_preserves_storage(kind, dtype):
    from akas.checkpoints.gguf import TYPES
    from akas.checkpoints.gguf import dequantize
    from akas.quantization.weights.ggml.validation import sample_weight

    count = 16 * 256 // TYPES[kind][1] * TYPES[kind][2]
    storage = np.zeros(count // np.dtype(dtype).itemsize, dtype=dtype)
    values = sample_weight(storage, kind)
    assert values.shape == (4096,)
    assert np.isfinite(values).all() and np.std(values) > 0
    np.testing.assert_array_equal(values, dequantize(storage.view("uint8"), kind))


@pytest.mark.parametrize("contract", ["packed", "reference"])
@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("encoding", [1, 2, 3, 8, 12, 14])
@pytest.mark.parametrize("rows", [1, 4])
def test_packed_projection_and_embedding_use_default_common_search(
    tmp_path, provider, encoding, rows, contract
):
    if os.environ.get("AKAS_FORWARD_" + provider.upper()) != "1":
        pytest.skip("enable native packed qualification")
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings
    from akas.quantization.weights.ggml.definition import weight_storage

    graph = Graph(contract=contract)
    count, dtype = weight_storage(encoding, 256 * 16, packed_words=provider == "webgpu")
    x = graph.buffer("x", (rows * 256,), "float32", role="input")
    weight = graph.buffer("weight", (count,), dtype, role="weight")
    out = graph.buffer("out", (rows * 16,), "float32", role="output")
    tokens = graph.buffer("tokens", (rows,), "int32", role="input")
    embedded = graph.buffer("embedded", (rows * 256,), "float32", role="output")
    graph(
        "linear",
        dict(r=rows, k=256, o=16, type=encoding, ggml=contract == "reference"),
        x,
        weight,
        out,
    )
    graph(
        "embedding",
        dict(r=rows, c=256, v=16, type=encoding, ggml=contract == "reference"),
        tokens,
        weight,
        embedded,
    )
    manifest = produce(
        graph,
        tmp_path,
        provider=provider,
        search_settings=SearchSettings(budget=2, samples=1, repeats=1),
    )
    assert len(manifest["tuning"]["operations"]) == 2
    for operation in manifest["tuning"]["operations"]:
        assert operation["selected"]["score"] > 0
        assert operation["selected"]["measurements"]
        assert any(
            candidate["status"] == "passed" for candidate in operation["candidates"]
        )


def test_q4_1_affine_block_matches_known_values():
    from akas.checkpoints.gguf import dequantize

    # d=0.5, m=-1, low nibbles ascending and high nibbles descending.
    raw = bytes.fromhex("003800bc") + bytes(
        [240, 225, 210, 195, 180, 165, 150, 135, 120, 105, 90, 75, 60, 45, 30, 15]
    )
    expected = np.array(
        [
            -1,
            -0.5,
            0,
            0.5,
            1,
            1.5,
            2,
            2.5,
            3,
            3.5,
            4,
            4.5,
            5,
            5.5,
            6,
            6.5,
            6.5,
            6,
            5.5,
            5,
            4.5,
            4,
            3.5,
            3,
            2.5,
            2,
            1.5,
            1,
            0.5,
            0,
            -0.5,
            -1,
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        dequantize(np.frombuffer(raw, dtype=np.uint8), 3), expected
    )
