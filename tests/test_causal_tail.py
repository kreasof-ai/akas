"""Causal suffix lowering preserves full KV updates and final convolution state."""

import os

import numpy as np
import pytest

from akas.compilation.rewrites import causal_tail
from akas.graph import Graph
from akas.models.lfm2.config import Config
from akas.models.lfm2._composition import compose


def graph(monkeypatch, provider="cuda", chunk=16):
    config = Config(64, 128, 1, 1, ("attention", "conv", "conv"), 64, 3, 1e-6, 10000, 64)
    with monkeypatch.context() as patch:
        patch.setattr("akas.compilation.rewrites.optimize", lambda value: value)
        return compose(config, batch=2, chunk=chunk, capacity=64, provider=provider, optimized=True)


def test_causal_tail_retains_full_cache_and_crops_query_and_convolutions(monkeypatch):
    original = graph(monkeypatch)
    lowered = causal_tail(Graph.from_metadata(original.metadata()))
    assert [call for call in original.calls if call.kind == "cache_write"] == [
        call for call in lowered.calls if call.kind == "cache_write"]
    assert next(call for call in lowered.calls if call.kind == "tail_controls").parameters == dict(r=2, chunk=16, tail=8)
    assert [call.parameters["chunk"] for call in lowered.calls if call.kind == "conv"] == [8, 8]
    assert lowered.buffers["blk.0.q"].shape == (2 * 8 * 64,)
    assert lowered.buffers["blk.0.k"].shape == original.buffers["blk.0.k"].shape
    assert all(lowered.buffers[name] == spec for name, spec in original.buffers.items() if spec.role == "state")
    assert lowered.calls[-1] == original.calls[-1]  # Full token count advances positions.
    small = graph(monkeypatch, chunk=8)
    before = small.metadata()
    assert causal_tail(small).metadata() == before
    assert causal_tail(Graph.from_metadata(original.metadata()), tail=4).metadata() == original.metadata()


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_causal_tail_native_ragged_output_and_state_parity(monkeypatch, tmp_path, provider):
    if os.environ.get("AKAS_FORWARD_" + provider.upper()) != "1":
        pytest.skip("enable native suffix qualification")
    import tensor
    from akas.compilation.compile import produce
    from akas.execution.forward import Forward
    from akas.memory.transfer import write_many

    original = graph(monkeypatch, provider)
    lowered = causal_tail(Graph.from_metadata(original.metadata()))
    assert any(call.kind == "tail_controls" for call in lowered.calls)
    rng = np.random.default_rng(509)
    supplied = {}
    for name, spec in original.buffers.items():
        if spec.role in ("weight", "input", "output") and spec.alias_of is None:
            if spec.dtype == "int32":
                supplied[name] = np.zeros(spec.shape, dtype="int32")
            else:
                values = rng.normal(0, .025, spec.shape)
                if "norm" in name:
                    values = np.ones(spec.shape)
                supplied[name] = values.astype(spec.dtype)
    paths = [tmp_path / "full", tmp_path / "tail"]
    for definition, path in zip((original, lowered), paths):
        produce(definition, path, provider=provider, search=False)
    with tensor.Device(provider=provider) as device:
        executions, allocations = [], []
        try:
            for path in paths:
                buffers = {name: device.from_numpy(value) for name, value in supplied.items()}
                allocations.extend(buffers.values())
                executions.append(Forward(path, device, buffers, graphs=True))
            for lengths in ([16, 9], [3, 0]):
                tokens = rng.integers(0, 64, size=(32,), dtype="int32")
                for forward in executions:
                    write_many(device, [(forward.buffers["active"], np.asarray(lengths, dtype="int32")),
                                        (forward.buffers["prefill_tokens"], tokens)])
                    forward.run()
                for name, spec in original.buffers.items():
                    if spec.role == "state" or name == "logits":
                        np.testing.assert_allclose(executions[0].buffers[name].to_numpy(),
                                                   executions[1].buffers[name].to_numpy(), rtol=.002, atol=.0002,
                                                   err_msg=name)
                np.testing.assert_array_equal(executions[0].buffers["sampled"].to_numpy(),
                                              executions[1].buffers["sampled"].to_numpy())
        finally:
            for forward in reversed(executions):
                forward.close()
            for buffer in reversed(allocations):
                buffer.release()
        assert not device._buffers
