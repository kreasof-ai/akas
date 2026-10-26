"""Quantized checkpoints retain their encoded device allocation."""

from types import SimpleNamespace
import numpy as np
import pytest

from akas.graph import Graph
from akas.models.spark import Spark, SparkCheckpoint
from akas.models.spark.weights import PackedWeights, tensor_shapes
from support.spark import fixture, fp8_fixture


class Device:
    info = {"provider": "cuda"}

    def from_numpy(self, values, *, dtype=None):
        return SimpleNamespace(
            values=values.copy(), dtype=dtype or str(values.dtype), release=lambda: None
        )


def test_gguf_upload_never_decodes_or_adds_floating_weights():
    raw = np.arange(16 * 144, dtype="uint8")

    def no_decode(*args):
        raise AssertionError("quantized weights must not be expanded during upload")

    gguf = SimpleNamespace(
        tensors={"matrix": SimpleNamespace(type=12, shape=(16, 256))},
        packed=lambda name: raw,
        array=no_decode,
        discard=lambda name: None,
    )
    c = SparkCheckpoint.__new__(SparkCheckpoint)
    c.gguf = gguf
    buffers = c.upload(Device())
    assert set(buffers) == {"matrix"}
    assert buffers["matrix"].dtype == "uint8"
    np.testing.assert_array_equal(buffers["matrix"].values, raw)
    graph = Graph(contract="reference")
    weights = PackedWeights(graph, gguf)
    assert weights.parameters("matrix") == {"ggml": True, "type": 12}
    assert list(graph.buffers) == ["matrix"]
    with pytest.raises(TypeError, match="prefill_cache"):
        c.upload(Device(), prefill_cache=True)


def test_all_gguf_phases_retain_original_encodings(tmp_path):
    c = SparkCheckpoint(fixture(tmp_path), tokenizer=False)
    c.gguf = SimpleNamespace(
        tensors={
            name: SimpleNamespace(shape=shape, type=2 if len(shape) == 2 else 0)
            for name, shape in tensor_shapes(c.config).items()
        }
    )
    for chunk in (1, 32, 64):
        g = Spark.compose(
            c,
            rows=2,
            chunk=chunk,
            max_requests=2,
            max_context=128,
            prefill_chunk=64,
            rotary_cache=True,
            optimized_attention=True,
        ).graph
        for name, b in g.buffers.items():
            if b.role == "weight" and not name.startswith("rope."):
                assert name in c.gguf.tensors
                assert b.dtype in ("uint8", "float32")
        assert all(
            call.parameters.get("type") == 2
            for call in g.calls
            if call.kind == "linear"
        )
    with pytest.raises(TypeError, match="prefill_weight_cache"):
        Spark.compose(
            c,
            rows=1,
            chunk=1,
            max_requests=1,
            max_context=128,
            prefill_weight_cache=True,
        )


def test_fp8_upload_retains_bytes_and_channel_scales(tmp_path, monkeypatch):
    from akas.checkpoints.safetensors import read_tensor

    c = SparkCheckpoint(fp8_fixture(tmp_path), tokenizer=False)
    original = c.read

    def read_unquantized(name):
        assert c.names[name] not in c.scales
        return original(name)

    monkeypatch.setattr(c, "read", read_unquantized)
    buffers = c.upload(Device())
    graph = c.graph_weights(Graph(contract="reference"), provider="cuda")
    for name, hf in c.names.items():
        if hf not in c.scales:
            continue
        assert buffers[name].dtype == "uint8"
        assert buffers[name + ".scale"].dtype == "bfloat16"
        assert graph[name].dtype == "uint8"
        np.testing.assert_array_equal(buffers[name].values, read_tensor(c.tensors[hf]))


def test_fused_prefill_narrows_activations_and_keeps_weights_packed(tmp_path):
    from dataclasses import replace

    c = SparkCheckpoint(fixture(tmp_path), tokenizer=False)
    c.config = replace(c.config, width=512, ff=1024)
    c.gguf = SimpleNamespace(
        tensors={
            name: SimpleNamespace(shape=shape, type=12 if len(shape) == 2 else 0)
            for name, shape in tensor_shapes(c.config).items()
        }
    )
    g = Spark.compose(
        c,
        rows=1,
        chunk=128,
        max_requests=1,
        max_context=128,
        prefill_chunk=128,
        optimized_attention=True,
        fused_ffn=True,
        half_activations=True,
        weight_arena=True,
    ).graph
    fused = [call for call in g.calls if call.kind == "geglu_linear"]
    assert len(fused) == len(c.config.layers)
    for call in fused:
        assert (
            call.parameters["input_dtype"]
            == call.parameters["output_dtype"]
            == "float16"
        )
        assert call.parameters["type"] == 12
        for name in call.arguments[1:3]:
            assert g.buffers[name].dtype == "uint8" and g.buffers[name].role == "weight"
    assert all(not name.endswith((".gate", ".up")) for name in g.buffers)
