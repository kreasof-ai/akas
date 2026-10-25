"""Adaptive split scratch retains the dense softmax and masks unused parts."""

import numpy as np
import pytest

from akas.compilation.policy import lower
from akas.kernels.registry import validation
from akas.models.lfm2.model import LFM2
from support.diagnostic import fixture


def test_adaptive_decode_policy_is_explicit_and_keeps_merge_abi(tmp_path):
    checkpoint = LFM2.checkpoint_type(fixture(tmp_path / "dense.gguf"), tokenizer=False)
    graph = LFM2.compose(checkpoint, rows=2, chunk=1, max_requests=4,
                         max_context=257, provider="webgpu").graph
    graph, _ = lower(graph, provider="webgpu", target="webgpu-portable-v1",
                     policy=dict(attention=dict(splits=64, adaptive_splits=True)))
    calls = [c for c in graph.calls if c.kind == "attention_adaptive_partial"]
    assert len(calls) == 1
    assert graph.buffers[calls[0].arguments[5]].shape == (2 * 4 * 64 * 66,)
    assert any(c.kind == "attention_merge" and c.parameters["splits"] == 64 for c in graph.calls)
    with pytest.raises(ValueError, match="WebGPU"):
        lower(graph, provider="cuda", policy=dict(attention=dict(adaptive_splits=True)))


@pytest.mark.parametrize("minimum", [2, 128])
def test_split_ffn_only_changes_prefill_and_preserves_weight_layout(minimum):
    from akas.graph import Graph
    graph = Graph(contract="reference")
    for rows in (1, 32, 512):
        x = graph.buffer(f"x{rows}", (rows * 256,), "float32", role="input")
        w = graph.buffer(f"w{rows}", (9216,), "uint32", role="weight")
        up = graph.buffer(f"up{rows}", (9216,), "uint32", role="weight")
        out = graph.buffer(f"out{rows}", (rows * 256,), "float32", role="output")
        graph("ffn", dict(r=rows, k=256, o=256, ggml=True, type=2,
                          weight_layout="planar"), x, w, up, out, label=f"ffn{rows}")
    graph, _ = lower(graph, provider="webgpu", policy=dict(linear=dict(
        split_ffn=True, split_ffn_min_rows=minimum)))
    expected = ["ffn", "linear", "linear", "swiglu", "linear", "linear", "swiglu"]
    if minimum == 128:
        expected = ["ffn", "ffn", "linear", "linear", "swiglu"]
    assert [c.kind for c in graph.calls] == expected
    assert graph.calls[-1].arguments[-1] == "out512" and graph.calls[-1].label == "ffn512"
    assert all(c.parameters["weight_layout"] == "planar"
               for c in graph.calls if c.kind == "linear")


def test_weight_staging_keeps_small_phases_and_declares_exact_half_scratch():
    from akas.graph import Graph
    graph = Graph(contract="reference")
    for rows in (1, 32, 512):
        x = graph.buffer(f"x{rows}", (rows * 256,), "float32", role="input")
        w = graph.buffer(f"w{rows}", (9216,), "uint32", role="weight")
        up = graph.buffer(f"up{rows}", (9216,), "uint32", role="weight")
        out = graph.buffer(f"out{rows}", (rows * 256,), "float32", role="output")
        graph("ffn", dict(r=rows, k=256, o=256, ggml=True, type=2), x, w, up, out)
    graph, _ = lower(graph, provider="webgpu", policy=dict(linear=dict(stage_weights=True)))
    assert [c.kind for c in graph.calls] == ["ffn", "ffn", "dequantize_ggml", "dequantize_ggml", "ffn"]
    assert graph.calls[-1].parameters["type"] == 1
    assert graph.calls[0].parameters["type"] == graph.calls[1].parameters["type"] == 2
    for call in graph.calls[2:4]:
        assert call.parameters["output_dtype"] == "float16"
        assert graph.buffers[call.arguments[-1]].shape == (256,256)


@pytest.mark.parametrize("capacity", [1, 19, 257, 4097, 8193])
@pytest.mark.parametrize("granule", [128, 512])
def test_adaptive_partial_reference_merges_to_dense_attention(capacity, granule):
    p = dict(r=2, h=4, kh=1, d=64, capacity=capacity, splits=64,
             kv_dtype="float16", unit_scales=True, optimized=True, provider="webgpu", granule=granule)
    signature = [dict(name=n, shape=s, dtype=t) for n,s,t in (
        ("q", [2,256], "float32"), ("kc", [2,1,capacity,64], "float16"),
        ("vc", [2,1,capacity,64], "float16"), ("ks", [2,1,1,1], "float32"),
        ("vs", [2,1,1,1], "float32"), ("out", [2*4*64*66], "float32"),
        ("positions", [2], "int32"), ("enabled", [2], "int32"))]
    case = validation("attention_adaptive_partial", p, signature)
    parts = case.expected["out"].reshape(2,4,64,66)
    live = min(64, (capacity + granule - 1) // granule)
    assert np.all(parts[0,:,live:,:64] == 0)
    assert np.all(parts[0,:,live:,65] == 0)
    assert np.all(np.isneginf(parts[0,:,live:,64]))
    assert np.all(parts[1,:,:,65] == 0)
    for head in range(4):
        value = parts[0,head].astype("float64")
        factors = np.exp(value[:,64] - value[:,64].max())
        merged = (value[:,:64] * factors[:,None]).sum(axis=0) / (value[:,65] * factors).sum()
        keys = case.arguments["kc"][0,0].astype("float64")
        values = case.arguments["vc"][0,0].astype("float64")
        query = case.arguments["q"][0,head*64:(head+1)*64].astype("float64")
        score = keys @ query / 8
        probability = np.exp(score - score.max())
        np.testing.assert_allclose(merged, probability @ values / probability.sum(),
                                   rtol=2e-6, atol=2e-7)
