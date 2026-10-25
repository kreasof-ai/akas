"""Staged-half optimizations must preserve the existing operand rounding."""

import os

import numpy as np
import pytest


def test_packed_half_rounding_boundaries(tmp_path):
    if os.environ.get("AKAS_LFM_MOE_WEBGPU") != "1":
        pytest.skip("opt-in WebGPU qualification")
    import tensor
    from akas.compilation.emit import build_artifact

    ties = np.array([2**-25, 2**-24, 2**-14, 1 + 2**-11, 1 + 3 * 2**-11,
                     0.0, 65504.0], dtype="float32")
    positive = np.concatenate([np.nextafter(ties, -np.inf), ties,
                               np.nextafter(ties, np.inf)])
    values = np.concatenate([positive, -positive])
    count = values.size
    source = tmp_path / "rounding.py"
    source.write_text(f'''
import tilelang.language as T
from akas.kernels.kernel import Tensor, kernel, output
from akas.quantization._shared.rounding import round_half, round_half_branch
rounded = round_half()
branched = round_half_branch()
@kernel
def compute(x: Tensor(({count},), "float32")) -> Tensor(({2 * count},), "float32"):
    out = output(({2 * count},), "float32")
    with T.Kernel(1, threads=64):
        i = T.get_thread_binding()
        if i < {count}:
            bits = T.call_extern("uint32", "pack2x16float",
                T.call_extern("float32x2", "vec2<f32>", rounded(x[i]), T.float32(0)))
            out[i] = T.call_extern("float32", "tensor_unpack_f16", bits)
            fast = T.call_extern("uint32", "pack2x16float",
                T.call_extern("float32x2", "vec2<f32>", branched(x[i]), T.float32(0)))
            out[i + {count}] = T.call_extern("float32", "tensor_unpack_f16", fast)
    return out
def tensor_export():
    return {{"kernel": compute}}
''')
    artifact = source.with_suffix(".tbin")
    build_artifact(source, artifact, provider="webgpu", target="webgpu-portable-v1")
    with tensor.Device(provider="webgpu") as device:
        x, out = device.from_numpy(values), device.empty((2 * count,), "float32")
        with device.load(artifact) as executable:
            executable.launch(x, out)
            expected = values.astype("float16").astype("float32")
            actual = out.to_numpy().reshape(2, count)
            np.testing.assert_array_equal(actual[0], expected)
            np.testing.assert_array_equal(actual[1], expected)
        out.release()
        x.release()
