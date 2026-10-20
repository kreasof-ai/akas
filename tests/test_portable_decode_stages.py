"""Shared router and vocabulary stages preserve their explicit operation contracts."""

import os
import numpy as np
import pytest


def argmax_graph(*, named=False, contract="reference"):
    from akas.graph import Graph

    graph = Graph(contract=contract)
    x = graph.buffer("x", (2 * 2051,), "float32", role="input")
    out = graph.buffer("out", (2,), "int32", role="output")
    enabled = graph.buffer("enabled", (2,), "int32", role="input")
    if named:
        graph("argmax", dict(r=2, c=2051), x=x, out=out, enabled=enabled, label="token")
    else:
        graph("argmax", dict(r=2, c=2051), x, out, enabled, label="token")
    graph.bind_requests(inputs={"enabled": "enabled"}, prediction="out")
    return graph


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("named", [False, True])
def test_parallel_argmax_policy_preserves_output_and_feedback(provider, named):
    from akas.compilation.policy import lower
    from akas.kernels.registry import graph_outputs

    graph = argmax_graph(named=named)
    request = graph.request.copy()
    commit = object()
    result, updated = lower(
        graph,
        commit,
        policy={"vocabulary": {"parallel_argmax": True, "block": 1024}},
        provider=provider,
    )
    assert result is graph and updated is commit
    assert graph.request == request
    partial, merge = graph.calls
    assert (partial.kind, merge.kind) == ("argmax_partial", "argmax_merge")
    assert graph.buffers[partial.bindings["values"]].shape == (2, 3)
    assert all(
        graph.buffers[n].role == "scratch"
        for n in graph_outputs(partial, contract="reference")
    )
    assert graph_outputs(merge, contract="reference") == ("out",)
    assert merge.bindings["enabled"] == "enabled" and merge.label == "token"
    # Applying the same policy again cannot create duplicate buffers/stages.
    lower(graph, policy={"vocabulary": {"parallel_argmax": True}}, provider=provider)
    assert len(graph.calls) == 2


@pytest.mark.parametrize(
    "settings",
    [
        {"parallel_argmax": 1},
        {"parallel_argmax": True, "block": 0},
        {"parallel_argmax": True, "block": True},
        {"unknown": True},
    ],
)
def test_parallel_argmax_policy_rejects_invalid_settings(settings):
    from akas.ops.vocabulary.definition import lower

    with pytest.raises(ValueError):
        lower(argmax_graph(), None, settings, provider="webgpu", target=None)


def test_parallel_argmax_policy_does_not_rewrite_native_stateful_contract():
    from akas.ops.vocabulary.definition import lower, search_policies

    graph = argmax_graph(contract="bf16")
    assert search_policies(graph, None, provider="cuda", target=None) == []
    with pytest.raises(ValueError, match="plain reference"):
        lower(graph, None, {"parallel_argmax": True}, provider="cuda", target=None)


def test_column_projection_default_is_limited_to_small_webgpu_f32_vectors():
    from akas.kernels.registry import select, candidates

    parameters = dict(r=1, k=2048, o=64)
    assert (
        select("linear", parameters, contract="reference", provider="webgpu").factory
        == "column_projection"
    )
    assert (
        select("linear", parameters, contract="reference", provider="cuda").factory
        == "floating_warp"
    )
    for changed in (
        dict(r=2),
        dict(o=257),
        dict(k=127),
        dict(weight_dtype="float16"),
        dict(ggml=True, type=2),
        dict(type=0),
    ):
        assert not any(
            i.factory == "column_projection"
            for i in candidates(
                "linear",
                {**parameters, **changed},
                contract="reference",
                provider="webgpu",
            )
        )


def compile_stage(
    tmp_path, provider, kind, parameters, *, preferred=None, schedule=None
):
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.selection import selected_source
    from akas.kernels.definition import definition
    from akas.compilation.emit import build_artifact

    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    entry, artifact = tmp_path / (kind + ".py"), tmp_path / (kind + ".tbin")
    spec = dict(kind=kind, parameters=parameters, contract="reference", phase="forward")
    capabilities = dict(provider=provider, target=target)
    implementation, axes = definition(spec, schedule, capabilities, preferred=preferred)
    entry.write_text(selected_source(spec, axes, capabilities, implementation).render())
    build_artifact(entry, artifact, provider=provider, target=target)
    return tensor, artifact, read_artifact(artifact)[0]


def gpu_enabled(provider):
    if os.environ.get("AKAS_LFM_MOE_" + provider.upper()) != "1":
        pytest.skip("opt-in portable decode GPU qualification")


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_parallel_argmax_gpu_tails_ties_invalid_and_inactive(tmp_path, provider):
    gpu_enabled(provider)
    from akas.memory.transfer import write

    r, c = 9, 2051
    p = dict(r=r, c=c, block=1024)
    tensor, partial, _ = compile_stage(tmp_path, provider, "argmax_partial", p)
    _, merge, _ = compile_stage(tmp_path, provider, "argmax_merge", p)
    rng = np.random.default_rng(17)
    x = rng.normal(size=(r, c)).astype("float32")
    x[0, [3, 1025, 2050]] = 100
    x[1] = -np.inf
    x[2] = np.nan
    x[3] = -np.inf
    x[3, [1, 2049]] = np.inf
    x[4] = -1
    x[4, [7, 2049]] = [0.0, -0.0]
    x[5, [0, 2050]] = [np.nan, 200.0]
    x[6] = -np.inf
    x[6, 2050] = -3
    x[7, 2050] = 300
    expected = np.array([3, c, c, 1, 7, 2050, 2050, 2050, -19], dtype="int32")
    enabled = np.ones(r, dtype="int32")
    enabled[-1] = 0
    with tensor.Device(provider=provider) as device:
        buffers = dict(
            x=device.from_numpy(x.reshape(-1)),
            enabled=device.from_numpy(enabled),
            out=device.from_numpy(np.full(r, -19, dtype="int32")),
            values=device.empty((r, 3), "float32"),
            indices=device.empty((r, 3), "int32"),
        )
        try:
            with device.load(partial) as a, device.load(merge) as b:
                for module in (a, b):
                    module.launch(
                        *(buffers[s["name"]] for s in module.manifest["arguments"])
                    )
                np.testing.assert_array_equal(buffers["out"].to_numpy(), expected)
                np.testing.assert_array_equal(buffers["x"].to_numpy(), x.reshape(-1))
                np.testing.assert_array_equal(buffers["enabled"].to_numpy(), enabled)
                write(device, buffers["enabled"], np.zeros(r, dtype="int32"))
                b.launch(*(buffers[s["name"]] for s in b.manifest["arguments"]))
                np.testing.assert_array_equal(buffers["out"].to_numpy(), expected)
        finally:
            for buffer in buffers.values():
                buffer.release()


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("threads", [32, 256])
def test_column_projection_gpu_f64_reference_tails_and_strided_output(
    tmp_path, provider, matrix, threads
):
    gpu_enabled(provider)
    p = dict(r=1, k=257, o=7, output_stride=15, output_offset=3, matrix_output=matrix)
    tensor, artifact, signature = compile_stage(
        tmp_path,
        provider,
        "linear",
        p,
        preferred=f"linear.{provider}.column_projection",
        schedule=dict(threads=threads),
    )
    rng = np.random.default_rng(19)
    x = rng.normal(size=257).astype("float32")
    w = rng.normal(size=7 * 257).astype("float32")
    shape = (1, 15) if matrix else (15,)
    sentinel = np.full(shape, 17.25, dtype="float32")
    expected = sentinel.copy().reshape(-1)
    expected[3:10] = (w.reshape(7, 257).astype("float64") @ x.astype("float64")).astype(
        "float32"
    )
    with tensor.Device(provider=provider) as device:
        buffers = dict(
            x=device.from_numpy(x),
            w=device.from_numpy(w),
            out=device.from_numpy(sentinel),
        )
        try:
            with device.load(artifact) as module:
                module.launch(
                    *(buffers[s["name"]] for s in module.manifest["arguments"])
                )
                result = buffers["out"].to_numpy().reshape(-1)
                np.testing.assert_allclose(
                    result[3:10], expected[3:10], rtol=2e-5, atol=2e-5
                )
                np.testing.assert_array_equal(
                    result[np.r_[0:3, 10:15]], expected[np.r_[0:3, 10:15]]
                )
                np.testing.assert_array_equal(buffers["x"].to_numpy(), x)
                np.testing.assert_array_equal(buffers["w"].to_numpy(), w)
        finally:
            for buffer in buffers.values():
                buffer.release()
