"""Finite native padding preserves valid KV and stays outside the live prefix."""

from contextlib import ExitStack
import os
import numpy as np
import pytest

from akas.graph import Graph
from akas.ops.attention.definition import lower


def graph(chunk=512):
    p = dict(r=chunk, chunk=chunk, h=4, kh=1, d=64, capacity=4096,
             kv_dtype="float16", unit_scales=True, provider="cuda", optimized=True)
    graph = Graph(contract="reference", phase="prefill")
    fields = {
        "q": ((chunk, 256), "float32"), "kc": ((1, 1, 4096, 64), "float16"),
        "vc": ((1, 1, 4096, 64), "float16"), "ks": ((1, 1, 4096, 1), "float32"),
        "vs": ((1, 1, 4096, 1), "float32"), "out": ((chunk * 256,), "float32"),
        "positions": ((chunk,), "int32"), "enabled": ((chunk,), "int32"),
    }
    buffers = {name: graph.buffer(name, shape, dtype,
                                 role="state" if name in ("kc", "vc", "ks", "vs") else "input")
               for name, (shape, dtype) in fields.items()}
    graph("attention", p, *buffers.values())
    return graph


def test_native_tail_lowering_reuses_kv_without_conversion_or_scale_changes():
    model = graph()
    lowered, _ = lower(model, None, dict(native_tail=True), provider="cuda", target="sm_89")
    assert [call.kind for call in lowered.calls] == ["attention_cache_extent", "attention_cache_pad", "attention"]
    assert lowered.calls[1].arguments[:2] == ("kc", "vc")
    assert lowered.calls[-1].arguments == ("q", "kc", "vc", "ks", "vs", "out", "positions", "enabled")
    assert lowered.calls[-1].parameters["staged_kv"] is True
    assert [spec.name for spec in lowered.buffers.values() if "tail" in spec.name] == ["pipeline.attention.tail.0.lengths"]
    assert lowered.buffers["kc"].dtype == "float16"
    small, _ = lower(graph(32), None, dict(native_tail=True), provider="cuda", target="sm_89")
    assert [call.kind for call in small.calls] == ["attention"]


@pytest.mark.parametrize("provider,target", [("webgpu", "webgpu-portable-v1"), ("cuda", "sm_90a")])
def test_native_tail_policy_requires_qualified_ordered_backend(provider, target):
    with pytest.raises(ValueError, match="sm_89"):
        lower(graph(), None, dict(native_tail=True), provider=provider, target=target)


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_native_tail_gpu_preserves_live_and_distant_future_bits(tmp_path, provider):
    if os.environ.get("AKAS_LFM_MOE_" + provider.upper()) != "1":
        pytest.skip("opt-in native tail qualification")
    import tensor
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import selected_source
    from akas.kernels.definition import definition

    parameters = dict(slots=6, heads=1, capacity=513, d=64)
    spec = dict(kind="attention_cache_pad", parameters=parameters, contract="reference", phase="prefill")
    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    capabilities = dict(provider=provider, target=target)
    implementation, schedule = definition(spec, {}, capabilities)
    source = tmp_path / "native-tail.py"
    source.write_text(selected_source(spec, schedule, capabilities, implementation).render())
    artifact = source.with_suffix(".tbin")
    build_artifact(source, artifact, target=target, provider=provider)
    lengths = np.array([0, 1, 255, 256, 257, 513], dtype="int32")
    keys = np.full((6, 1, 513, 64), np.nan, dtype="float16")
    values = keys.copy()
    expected_k, expected_v = keys.copy(), values.copy()
    for row, live in enumerate(lengths):
        keys[row, :, :live] = expected_k[row, :, :live] = .75
        values[row, :, :live] = expected_v[row, :, :live] = -.5
        end = min((int(live) + 255) // 256 * 256, 513)
        expected_k[row, :, live:end] = expected_v[row, :, live:end] = 0
    with tensor.Device(provider=provider) as device, ExitStack() as owned:
        module = device.load(artifact)
        owned.callback(module.release)
        buffers = {name: device.from_numpy(value) for name, value in
                   (("kc", keys), ("vc", values), ("lengths", lengths))}
        for buffer in buffers.values():
            owned.callback(buffer.release)
        module.launch(*(buffers[argument["name"]] for argument in module.manifest["arguments"]))
        np.testing.assert_array_equal(buffers["kc"].to_numpy().view("uint16"), expected_k.view("uint16"))
        np.testing.assert_array_equal(buffers["vc"].to_numpy().view("uint16"), expected_v.view("uint16"))
        np.testing.assert_array_equal(buffers["lengths"].to_numpy(), lengths)
