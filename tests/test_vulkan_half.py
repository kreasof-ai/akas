"""Declared short-half arithmetic has an independent error bound and RMS gate."""

import numpy as np
import pytest


def test_half_input_staging_is_declared_and_keeps_decode_abi():
    from akas.graph import Graph
    from akas.compilation.policy import lower

    graph = Graph(contract="reference")
    for rows in (1, 32):
        x = graph.buffer(f"x{rows}", (rows*128,), "float32", role="input")
        w = graph.buffer(f"w{rows}", (128*128//2,), "uint32", role="weight")
        out = graph.buffer(f"out{rows}", (rows*128,), "float32", role="output")
        graph("linear", dict(r=rows, k=128, o=128, ggml=True, type=1,
                             half_accum=rows>1), x, w, out)
    graph, _ = lower(graph, provider="webgpu", policy=dict(linear=dict(stage_inputs=True)))
    assert [c.kind for c in graph.calls] == ["linear", "linear_half_input", "linear"]
    assert graph.calls[0].arguments[0] == "x1"
    staged = graph.calls[-1]
    assert staged.parameters["input_dtype"] == "float16"
    assert graph.buffers[staged.arguments[0]].dtype == "float16"
    assert graph.calls[-2].arguments == ("x32", staged.arguments[0])


@pytest.mark.parametrize("kind", ["linear", "ffn"])
@pytest.mark.parametrize("encoding", [1, 2])
def test_short_half_gate_accepts_independent_fma_emulation_and_rejects_corruption(kind, encoding):
    from akas.kernels.registry import validation
    from akas.quantization.weights.ggml.validation import decode_rows

    r, k, o = 3, 128, 18
    size = k * o // 2 if encoding == 1 else k * o // 32 * 18 // 4
    signature = [dict(name="x", shape=[r*k], dtype="float32"),
                 dict(name="w", shape=[size], dtype="uint32")]
    if kind == "ffn":
        signature.append(dict(name="w2", shape=[size], dtype="uint32"))
    signature.append(dict(name="out", shape=[r*o], dtype="float32"))
    p = dict(r=r, k=k, o=o, ggml=True, type=encoding, half_accum=True)
    case = validation(kind, p, signature)
    x = case.arguments["x"].reshape(r, k).astype("float16").astype("float64")

    def product(name):
        weight = decode_rows(case.arguments[name], encoding, k, 0, o).astype("float16").astype("float64")
        result = np.zeros((r, o), "float32")
        for block in range(0, k, 16):
            even = np.zeros((r, o), "float16")
            odd = np.zeros((r, o), "float16")
            for step in range(8):
                index = block + step * 2
                even = (even.astype("float64") + x[:, index, None] * weight[None, :, index]).astype("float16")
                odd = (odd.astype("float64") + x[:, index+1, None] * weight[None, :, index+1]).astype("float16")
            result += even.astype("float32") + odd.astype("float32")
        return result

    actual = product("w")
    if kind == "ffn":
        actual = actual / (1 + np.exp(-actual)) * product("w2")
    actual = actual.reshape(-1)
    check = case.comparators["out"]
    check(actual, case.expected["out"])
    broken = actual.copy()
    broken[0] += 10
    with pytest.raises(AssertionError):
        check(broken, case.expected["out"])
