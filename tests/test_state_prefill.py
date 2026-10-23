"""State-only prefill retains persistent effects and conservative dependencies."""

from akas.graph import Graph
from akas.compilation.liveness import state_prefill


def graph():
    plan = Graph(contract="reference", phase="prefill")
    for name, shape, dtype, role in [
        ("tokens", (1,), "int32", "input"),
        ("lengths", (1,), "int32", "input"),
        ("positions", (1,), "int32", "state"),
        ("x", (1,), "float32", "input"),
        ("project_weight", (3, 1), "float32", "weight"),
        ("projection", (3,), "float32", "scratch"),
        ("filter", (3,), "float32", "weight"),
        ("state", (1, 2, 1), "float32", "state"),
        ("mixed", (1,), "float32", "scratch"),
        ("head_weight", (2, 1), "float32", "weight"),
        ("logits", (2,), "float32", "output"),
        ("sampled", (1,), "int32", "output"),
    ]:
        plan.buffer(name, shape, dtype, role=role)
    plan("linear", dict(r=1, k=1, o=3), "x", "project_weight", "projection")
    plan(
        "conv",
        dict(r=1, c=1, length=3),
        "projection",
        "filter",
        "state",
        "mixed",
        "lengths",
    )
    plan("linear", dict(r=1, k=1, o=2), "mixed", "head_weight", "logits")
    plan("argmax", dict(r=1, c=2), "logits", "sampled", "lengths")
    plan("next_tokens", dict(r=1), "sampled", "tokens")
    plan("advance", dict(r=1), "positions", "lengths")
    return plan.bind_requests(
        inputs={"tokens": "tokens", "lengths": "lengths"},
        prediction="sampled",
        reset=["positions", "state"],
        features={"logits": "logits"},
    )


def test_state_prefill_removes_head_and_sampling_without_losing_state_writes():
    original = graph()
    derived = state_prefill(original)
    assert [call.kind for call in derived.calls] == ["linear", "conv", "advance"]
    assert "head_weight" not in derived.buffers and "logits" not in derived.buffers
    assert "features" not in derived.request
    assert derived.buffers["state"] == original.buffers["state"]
    assert len(original.calls) == 6


def test_unknown_operation_keeps_all_its_inputs_live():
    original = graph()
    original("foreign", {}, "logits")
    derived = state_prefill(original)
    calls = original.calls if derived is None else derived.calls
    assert any(call.kind == "foreign" for call in calls)
    assert any(call.kind == "linear" and call.parameters["o"] == 2 for call in calls)


def test_requested_intermediate_features_survive_without_the_terminal_head():
    import pytest

    original = graph()
    original.bind_requests(
        inputs=original.request["inputs"],
        prediction="sampled",
        reset=["positions", "state"],
        features={"mixer": "mixed", "logits": "logits"},
        prefill_features=("mixer",),
    )
    restored = Graph.from_metadata(original.metadata())
    derived = state_prefill(restored)
    assert [c.kind for c in derived.calls] == ["linear", "conv", "advance"]
    assert derived.request["features"] == {"mixer": "mixed"}
    assert derived.request["prefill_features"] == ["mixer"]
    assert "logits" not in derived.buffers and "head_weight" not in derived.buffers
    assert derived.buffers["mixed"].shape == (1,)
    with pytest.raises(ValueError, match="prefill features"):
        original.bind_requests(
            inputs=original.request["inputs"],
            prediction="sampled",
            features={"mixer": "mixed"},
            prefill_features=("missing",),
        )
