"""Ragged suffix controls and row gathers share both provider implementations."""

import os

import numpy as np
import pytest

from akas.ops.sequence.tail import validation


def test_suffix_reference_preserves_empty_short_and_full_rows():
    signature = [dict(name="x", shape=[4 * 8 * 3], dtype="float32"),
                 dict(name="lengths", shape=[4], dtype="int32"),
                 dict(name="out", shape=[4 * 4 * 3], dtype="float32")]
    case = validation("chunk_tail", dict(r=4, c=3, chunk=8, tail=4), signature)
    out = case.expected["out"].reshape(4, 4, 3)
    x = case.arguments["x"].reshape(4, 8, 3)
    np.testing.assert_array_equal(out[0], x[0, 4:])
    np.testing.assert_array_equal(out[1], 0)
    np.testing.assert_array_equal(out[2, :2], x[2, :2])
    np.testing.assert_array_equal(out[2, 2:], 0)


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_suffix_controls_and_gather_native_qualification(tmp_path, provider):
    if os.environ.get("AKAS_FORWARD_" + provider.upper()) != "1":
        pytest.skip("enable native suffix qualification")
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings

    graph = Graph(contract="reference")
    x = graph.buffer("x", (4 * 8 * 3,), "float32", role="input")
    lengths = graph.buffer("lengths", (4,), "int32", role="input")
    positions = graph.buffer("positions", (4 * 8,), "int32", role="input")
    tail = graph.buffer("tail", (4 * 4 * 3,), "float32", role="output")
    tail_positions = graph.buffer("tail_positions", (4 * 4,), "int32", role="output")
    active = graph.buffer("active", (4 * 4,), "int32", role="output")
    tail_lengths = graph.buffer("tail_lengths", (4,), "int32", role="output")
    graph("chunk_tail", dict(r=4, c=3, chunk=8, tail=4), x, lengths, tail)
    graph("tail_controls", dict(r=4, chunk=8, tail=4), positions, lengths, tail_positions, active, tail_lengths)
    result = produce(graph, tmp_path, provider=provider,
                     search_settings=SearchSettings(budget=1, samples=1, repeats=1))
    assert all(operation["selected"]["score"] > 0 for operation in result["tuning"]["operations"])
