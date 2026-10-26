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


def test_lifted_state_reader_keeps_output_abi_when_mapping_is_appended(monkeypatch):
    from types import SimpleNamespace
    from akas.artifacts.schema import Kernel, Source
    from akas.compilation import state
    from akas.kernels.registry import graph_outputs

    g = Graph(contract="reference", phase="prefill")
    for name, shape, dtype, role in [
        ("tokens", (2,), "int32", "input"),
        ("enabled", (2,), "int32", "input"),
        ("mapping", (2,), "int32", "input"),
        ("positions", (4,), "int32", "state"),
        ("row_positions", (2,), "int32", "input"),
        ("q", (2, 64), "float32", "input"),
        ("kc", (4, 1, 4, 64), "float16", "state"),
        ("vc", (4, 1, 4, 64), "float16", "state"),
        ("parts", (132,), "float32", "scratch"),
        ("persistent", (4,), "float32", "state"),
        ("head", (2, 132), "float32", "weight"),
        ("logits", (4,), "float32", "output"),
        ("sampled", (2,), "int32", "output"),
    ]:
        g.buffer(name, shape, dtype, role=role)
    p = dict(r=2, h=1, kh=1, d=64, capacity=4, chunk=1, window=0, splits=1)
    call = g(
        "window_attention", p, "q", "kc", "vc", "row_positions", "enabled", "parts"
    )
    g("foreign", {}, "parts", "persistent")
    g("linear", dict(r=2, k=132, o=2), "parts", "head", "logits")
    g("argmax", dict(r=2, c=2), "logits", "sampled", "enabled")
    g("advance", dict(r=2), "positions", "enabled")
    g.bind_requests(
        inputs=dict(tokens="tokens", active="enabled"), prediction="sampled"
    )
    names = ("q", "kc", "vc", "positions", "enabled", "out")
    shapes = ((2, 64), (2, 1, 4, 64), (2, 1, 4, 64), (2,), (2,), (132,))
    params = list(range(6))
    fake = SimpleNamespace(
        params=params,
        buffer_map={
            i: SimpleNamespace(name=n, shape=s)
            for i, n, s in zip(params, names, shapes)
        },
    )
    monkeypatch.setattr(
        state, "import_module", lambda _: SimpleNamespace(make=lambda: fake)
    )
    adapted, _ = state.map_requests(
        g, [Kernel(call.key, "window_attention", p, Source("stub", "make"))]
    )
    assert graph_outputs(adapted.calls[0], contract="reference") == ("parts",)
    assert adapted.calls[0].bindings["mapping"] == "mapping"
    derived = state_prefill(Graph.from_metadata(adapted.metadata()))
    assert [c.kind for c in derived.calls] == ["window_attention", "foreign", "advance"]
