"""Decode fusion preserves observable graph values, codec boundaries and math."""

import os
import numpy as np
import pytest


def graph_fixture(provider="cuda", rows=1):
    from akas.graph import Graph
    from akas.quantization.formats import get
    from akas.quantization.weights import Weight
    from akas.ops.moe.definition import routed_moe

    graph = Graph(contract="reference", phase="decode" if rows == 1 else "prefill")
    x = graph.buffer("x", (rows, 256), "float32", role="input")
    logits = graph.buffer("router", (rows, 6), "float32", role="input")
    bias = graph.buffer("bias", (6,), "float32", role="weight")
    active = graph.buffer("active", (rows,), "int32", role="input")
    weights = {}
    for name, columns, depth in (("gate", 64, 256), ("up", 64, 256), ("down", 256, 64)):
        shape = (6, columns, depth)
        storage = (
            get("weights", "ggml").storage(shape, provider=provider, encoding=2).values
        )
        data = graph.buffer(
            name + ".weight", storage.shape, storage.dtype, role="weight"
        )
        parts = ({"rows": 6 * columns, "format": "ggml", "options": {"encoding": 2}},)
        weights[name] = Weight(shape, (data,), parts)
    routed_moe(
        graph,
        x,
        logits,
        bias,
        weights,
        top=4,
        active=active,
        prefix="moe.",
        provider=provider,
    )
    return graph


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("named", [False, True])
def test_fusion_is_searchable_and_preserves_the_activation_buffer(provider, named):
    from akas.graph import Call
    from akas.ops.moe.definition import lower, search_policies
    from akas.kernels.registry import graph_outputs

    graph = graph_fixture(provider)
    if named:
        for index, call in enumerate(graph.calls):
            names = (
                ("x", "w", "ids", "counts", "out")
                if call.kind == "quantized_experts"
                and not call.parameters["routed_input"]
                else ("g", "u", "out")
                if call.kind == "swiglu_experts"
                else None
            )
            if names:
                graph.calls[index] = Call(
                    call.kind,
                    call.parameters,
                    bindings=dict(zip(names, call.arguments)),
                    label=call.label,
                )
    original = [call.kind for call in graph.calls]
    activation = next(c for c in graph.calls if c.kind == "swiglu_experts")
    dest = activation.bindings.get("out") if named else activation.arguments[-1]
    assert {"fused_swiglu": True} in search_policies(
        graph, None, provider=provider, target="test"
    )
    lowered, commit = lower(
        graph, None, {"fused_swiglu": True}, provider=provider, target="test"
    )
    assert lowered is graph and commit is None
    assert len(graph.calls) == len(original) - 2
    fused = next(c for c in graph.calls if c.kind == "expert_swiglu")
    assert fused.arguments[-1] == dest and graph_outputs(
        fused, contract="reference"
    ) == (dest,)
    assert fused.parameters["round_inputs"] == (provider == "cuda")
    assert len([c for c in graph.calls if c.kind == "quantized_experts"]) == 1


@pytest.mark.parametrize("observable", ["feature", "view", "second-consumer"])
def test_fusion_keeps_projection_outputs_that_are_still_observable(observable):
    from akas.ops.moe.definition import lower, search_policies

    graph = graph_fixture()
    activation = next(c for c in graph.calls if c.kind == "swiglu_experts")
    gate = activation.arguments[0]
    if observable == "feature":
        graph.bind_requests(inputs={}, prediction="moe.out", features={"gate": gate})
    elif observable == "view":
        graph.view("gate.view", graph.buffers[gate], (1, 256))
    else:
        extra = graph.buffer("second.activation", (1, 4, 64), "float32")
        graph("swiglu_experts", activation.parameters, *activation.arguments[:2], extra)
    before = graph.metadata()
    assert search_policies(graph, None, provider="cuda", target="sm_89") == []
    lower(graph, None, {"fused_swiglu": True}, provider="cuda", target="sm_89")
    assert graph.metadata() == before


def test_prefill_is_not_rewritten_and_policy_types_are_checked():
    from akas.ops.moe.definition import lower, search_policies

    graph = graph_fixture(rows=32)
    before = graph.metadata()
    assert search_policies(graph, None, provider="cuda", target="sm_89") == []
    lower(graph, None, {"fused_swiglu": True}, provider="cuda", target="sm_89")
    assert graph.metadata() == before
    with pytest.raises(ValueError, match="boolean"):
        lower(graph, None, {"fused_swiglu": "yes"}, provider="cuda", target="sm_89")
    graph.contract = "block-fp8"
    with pytest.raises(ValueError, match="reference contract"):
        lower(graph, None, {"fused_swiglu": True}, provider="cuda", target="sm_89")


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("up_encoding", [2, 3])
def test_fused_operation_with_independent_codecs_and_inactive_experts(
    tmp_path, provider, up_encoding
):
    if (
        os.environ.get(
            "AKAS_LFM_MOE_CUDA" if provider == "cuda" else "AKAS_LFM_MOE_WEBGPU"
        )
        != "1"
    ):
        pytest.skip("opt-in expert fusion GPU qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.selection import source
    from akas.compilation.emit import build_artifact
    from akas.kernels.registry import validation
    from akas.memory.transfer import write

    p = dict(
        r=1,
        top=4,
        experts=6,
        k=256,
        o=37,
        input_dtype="float32",
        round_inputs=provider == "cuda",
        weight_parts=({"rows": 6 * 37, "format": "ggml", "options": {"encoding": 2}},),
        up_weight_parts=(
            {"rows": 6 * 37, "format": "ggml", "options": {"encoding": up_encoding}},
        ),
    )
    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    entry, artifact = tmp_path / "fused.py", tmp_path / "fused.tbin"
    entry.write_text(
        source(
            "expert_swiglu", p, contract="reference", provider=provider, target=target
        ).render()
    )
    build_artifact(entry, artifact, provider=provider, target=target)
    signature, _ = read_artifact(artifact)
    case = validation("expert_swiglu", p, signature["arguments"])
    with tensor.Device(provider=provider) as device:
        buffers = {n: device.from_numpy(v) for n, v in case.arguments.items()}
        try:
            with device.load(artifact) as module:
                args = tuple(buffers[a["name"]] for a in module.manifest["arguments"])
                module.launch(*args)
                np.testing.assert_allclose(
                    buffers["out"].to_numpy(),
                    case.expected["out"],
                    rtol=case.rtol,
                    atol=case.atol,
                )
                for name, value in case.arguments.items():
                    if name != "out":
                        np.testing.assert_array_equal(buffers[name].to_numpy(), value)
                write(
                    device, buffers["counts"], np.zeros_like(case.arguments["counts"])
                )
                module.launch(*args)
                np.testing.assert_array_equal(
                    buffers["out"].to_numpy(), np.zeros_like(case.expected["out"])
                )
        finally:
            for buffer in buffers.values():
                buffer.release()
