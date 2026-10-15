"""Explicit Q16 lowering preserves declared bytes and kernel arithmetic."""

import os
import numpy as np
import pytest


def test_q16_encoding_reconstruction_and_zero():
    from akas.quantization.activations.q16.validation import encode
    from akas.quantization.activations.q16.validation import decode

    values = np.random.default_rng(4).normal(size=(4,256)).astype("float32")
    values[0] = 0
    for fixed in (False,True):
        packed, scales, sums = encode(values, fixed_residual=fixed)
        q = packed.view("int8").reshape(2,4,8,32)
        np.testing.assert_array_equal(sums.reshape(2,4,8), q.astype("int32").sum(-1))
        reconstructed = decode(packed, scales, rows=4, depth=256)
        np.testing.assert_allclose(reconstructed, values.astype("float16").astype("float32"), atol=.0002,rtol=.0002)
        np.testing.assert_array_equal(reconstructed[0], 0)


def test_q16_lowering_reuses_source_quantization():
    from akas.graph import Graph
    from akas.compilation.quantization import q16_activations

    graph = Graph(contract="reference")
    x = graph.buffer("x",(4*256,),"float32",role="input")
    w = graph.buffer("w",(16*256//32*18//4,),"uint32",role="weight")
    for name in ("a","b"):
        out=graph.buffer(name,(4*16,),"float32",role="output")
        graph("linear",dict(r=4,k=256,o=16,type=2,ggml=True),x,w,out)
    q16_activations(graph)
    assert [call.kind for call in graph.calls] == ["quantize_q16","linear","linear"]
    assert graph.calls[1].arguments[:3] == graph.calls[2].arguments[:3]
    assert graph.calls[1].parameters["q16"]


@pytest.mark.skipif(os.environ.get("AKAS_FORWARD_WEBGPU") != "1", reason="native WebGPU")
def test_q16_common_default_search_qualifies_quantizer_and_projection(tmp_path):
    from akas.graph import Graph
    from akas.compilation.quantization import q16_activations
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings

    graph = Graph(contract="reference")
    x = graph.buffer("x",(4*256,),"float32",role="input")
    w = graph.buffer("w",(16*256//32*18//4,),"uint32",role="weight")
    out=graph.buffer("out",(4*16,),"float32",role="output")
    graph("linear",dict(r=4,k=256,o=16,type=2,ggml=True),x,w,out)
    q16_activations(graph)
    result=produce(graph,tmp_path,provider="webgpu",search_settings=SearchSettings(budget=2,samples=1,repeats=1))
    assert len(result["tuning"]["operations"]) == 2
