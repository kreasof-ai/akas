"""Named features remain readable after the complete compiled phase executes."""

from akas.compilation.buffers import scratch_slots
from akas.graph import Graph


def test_exposed_scratch_view_cannot_alias_later_intermediate_values():
    graph = Graph(contract="reference")
    for name, shape, dtype, role in [
        ("tokens", (1,), "int32", "input"),
        ("active", (1,), "int32", "input"),
        ("x", (4,), "float32", "input"),
        ("feature", (4,), "float32", "scratch"),
        ("middle", (4,), "float32", "scratch"),
        ("later", (4,), "float32", "scratch"),
        ("sampled", (1,), "int32", "output"),
    ]:
        graph.buffer(name, shape, dtype, role=role)
    graph("add", dict(r=1, c=4), "x", "x", "feature")
    graph("add", dict(r=1, c=4), "feature", "x", "middle")
    graph("add", dict(r=1, c=4), "middle", "x", "later")
    graph("argmax", dict(r=1, c=4), "later", "sampled", "active")
    before = scratch_slots(graph)
    assert any({"feature", "later"} <= set(group) for group in before)
    view = graph.buffers["feature"].view((2, 2))
    graph.bind_requests(
        inputs={"tokens": "tokens", "lengths": "active"},
        prediction="sampled",
        features={"hidden": view.name},
    )
    assert graph.lifetimes()["feature"] == (0, 4)
    assert graph.lifetimes()[view.name] == (4, 4)
    assert graph.lifetimes()["sampled"] == (3, 4)
    assert not any({"feature", "later"} <= set(group) for group in scratch_slots(graph))
    restored = Graph.from_metadata(graph.metadata())
    assert restored.lifetimes() == graph.lifetimes()
    assert scratch_slots(restored) == scratch_slots(graph)
