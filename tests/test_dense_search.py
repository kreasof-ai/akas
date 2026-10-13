"""Dense projections and fused sampling use the same default search contract."""

import os

import pytest


@pytest.mark.parametrize("rows", [1, 4])
def test_dense_matmul_matvec_and_fused_head_qualify_without_model_tuner(tmp_path, rows):
    if os.environ.get("AKAS_FORWARD_CUDA") != "1":
        pytest.skip("enable native dense search qualification")
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings

    graph = Graph(contract="bf16")
    x = graph.buffer("x", (rows, 256), "bfloat16", role="input")
    weight = graph.buffer("weight", (128, 256), "bfloat16", role="weight")
    for kind in ("linear", "gemv"):
        out = graph.buffer(kind, (rows, 128), "float32", role="output")
        graph(kind, dict(r=rows, k=256, o=128), x, weight, out)
    maximum = graph.buffer("maximum", (rows, 2), "float32")
    index = graph.buffer("index", (rows, 2), "int32")
    active = graph.buffer("active", (rows,), "int32", role="input")
    predicted = graph.buffer("predicted", (rows,), "int32", role="output")
    graph("head_linear", dict(r=rows, k=256, o=128, m=16, n=64, bk=128), x, weight, maximum, index)
    graph("head_argmax", dict(r=rows, tiles=2, vocab=128), maximum, index, active, predicted)
    result = produce(graph, tmp_path, search_settings=SearchSettings(budget=2, samples=1, repeats=1))
    operations = {row["kind"]: row for row in result["tuning"]["operations"]}
    assert set(operations) == {"linear", "gemv", "head_linear", "head_argmax"}
    assert all(row["selected"]["score"] > 0 for row in operations.values())
    assert all(operations[kind]["selected"]["trials"] > 0 for kind in ("linear", "gemv"))
