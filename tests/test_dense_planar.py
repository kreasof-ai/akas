"""Dense Q4 layout changes preserve bytes, declarations and dot references."""

from contextlib import ExitStack
import os
from types import SimpleNamespace

import numpy as np
import pytest

from support.diagnostic import fixture
from akas.models.lfm2.model import LFM2


@pytest.mark.parametrize("chunk", [1, 32])
def test_dense_planar_declaration_preserves_allocations(tmp_path, chunk):
    checkpoint = LFM2.checkpoint_type(fixture(tmp_path / "dense.gguf"), tokenizer=False)
    options = dict(rows=1, chunk=chunk, max_requests=1, max_context=64, provider="webgpu")
    raw = LFM2.compose(checkpoint, **options).graph
    planar = LFM2.compose(checkpoint, dense_weight_layout="planar", **options).graph
    assert {n:(b.shape,b.dtype) for n,b in raw.buffers.items() if b.role == "weight"} == {
        n:(b.shape,b.dtype) for n,b in planar.buffers.items() if b.role == "weight"}
    changed = [c for c in planar.calls if c.parameters.get("weight_layout") == "planar"]
    assert changed and all(c.parameters["type"] == 2 for c in changed)
    assert all("token_embd.weight" not in c.arguments for c in changed)
    assert checkpoint.program_upload_options({}, model_options=dict(dense_weight_layout="planar")) == {
        "dense_weight_layout": "planar"}
    with pytest.raises(ValueError, match="WebGPU"):
        LFM2.compose(checkpoint, **{**options, "provider": "cuda"}, dense_weight_layout="planar")


def test_dense_planar_upload_is_reversible_without_expansion(tmp_path):
    checkpoint = LFM2.checkpoint_type(fixture(tmp_path / "dense.gguf"), tokenizer=False)
    original = {n:checkpoint.gguf.packed(n).copy().view("uint8").reshape(-1)
                for n in checkpoint.gguf.tensors}

    class Device:
        info = dict(provider="webgpu")

        def prepare_plan(self, calls):
            return SimpleNamespace(launch=lambda:None, close=lambda:None)

        def synchronize(self):
            pass

        def from_numpy(self, values):
            return SimpleNamespace(values=values.copy(), release=lambda:None)

    from akas.models.lfm2.weights import dense_planar
    from akas.quantization.weights.ggml.kernel import repack_host
    uploaded = checkpoint.upload(Device(), dense_weight_layout="planar")
    for name, buffer in uploaded.items():
        raw = buffer.values.view("uint8").reshape(-1)
        if dense_planar(name, checkpoint.gguf.tensors[name], "planar"):
            raw = repack_host(raw, source="planar", target="raw")
        np.testing.assert_array_equal(raw[:len(original[name])], original[name])
        assert len(raw) == (len(original[name]) + 3) // 4 * 4


@pytest.mark.parametrize("kind,rows,preferred", [
    ("linear", 1, None), ("ffn", 1, None), ("linear_add", 1, None),
    ("linear", 17, "linear.webgpu.packed:blocked-prefill"),
    ("ffn", 17, "linear.webgpu.packed:branch-prefill"),
])
def test_dense_planar_device_reference(tmp_path, kind, rows, preferred):
    if os.environ.get("AKAS_LFM_MOE_WEBGPU") != "1":
        pytest.skip("opt-in planar projection qualification")
    import tensor
    from akas.compilation.selection import source
    from akas.compilation.emit import build_artifact
    from akas.kernels.registry import validation
    from tensor.artifacts.format import read_artifact

    p = dict(r=rows, k=256, o=258, type=2, ggml=True, optimized=True, weight_layout="planar")
    entry = tmp_path / "projection.py"
    entry.write_text(source(kind, p, contract="reference", provider="webgpu",
        target="webgpu-portable-v1", preferred=preferred).render())
    artifact = entry.with_suffix(".tbin")
    build_artifact(entry, artifact, provider="webgpu", target="webgpu-portable-v1")
    meta, _ = read_artifact(artifact)
    case = validation(kind, p, meta["arguments"])
    with tensor.Device(provider="webgpu") as device, ExitStack() as stack:
        buffers = {n:device.from_numpy(a) for n,a in case.arguments.items()}
        for buffer in buffers.values():
            stack.callback(buffer.release)
        executable = stack.enter_context(device.load(artifact))
        executable.launch(*(buffers[a["name"]] for a in meta["arguments"]))
        np.testing.assert_allclose(buffers["out"].to_numpy(), case.expected["out"],
                                   rtol=case.rtol, atol=case.atol)


@pytest.mark.parametrize("encoding,layout", [(2, "raw"), (2, "planar"), (14, "raw")])
def test_staged_weight_device_reference(tmp_path, encoding, layout):
    if os.environ.get("AKAS_LFM_MOE_WEBGPU") != "1":
        pytest.skip("opt-in exact weight staging qualification")
    import tensor
    from akas.compilation.selection import source
    from akas.compilation.emit import build_artifact
    from akas.kernels.registry import validation
    from tensor.artifacts.format import read_artifact

    p = dict(r=258, k=256, type=encoding, output_dtype="float16", weight_layout=layout)
    entry = tmp_path / "stage.py"
    entry.write_text(source("dequantize_ggml", p, contract="reference", provider="webgpu",
                           target="webgpu-portable-v1").render())
    artifact = entry.with_suffix(".tbin")
    build_artifact(entry, artifact, provider="webgpu", target="webgpu-portable-v1")
    meta, _ = read_artifact(artifact)
    case = validation("dequantize_ggml", p, meta["arguments"])
    from akas.quantization.weights.ggml.kernel import decode_host, repack_host
    raw = case.arguments["w"].view("uint8").reshape(-1).copy()
    if layout == "planar":
        raw = repack_host(raw, source="planar", target="raw")
    size, offset = (18, 0) if encoding == 2 else (210, 208)
    scales = np.resize(np.array([2**-24, -2**-24, 2**-14, -2**-14,
                                  .125, -.125, .001], "float16"), len(raw) // size)
    raw.reshape(-1, size)[:, offset:offset+2] = scales.view("uint8").reshape(-1, 2)
    case.arguments["w"].view("uint8").reshape(-1)[:] = (
        repack_host(raw) if layout == "planar" else raw)
    case.expected["w"][:] = case.arguments["w"]
    case.expected["out"][:] = decode_host(raw, encoding).astype("float16").reshape(258, 256)
    with tensor.Device(provider="webgpu") as device, ExitStack() as stack:
        buffers = {n:device.from_numpy(a) for n,a in case.arguments.items()}
        for buffer in buffers.values():
            stack.callback(buffer.release)
        executable = stack.enter_context(device.load(artifact))
        executable.launch(*(buffers[a["name"]] for a in meta["arguments"]))
        np.testing.assert_array_equal(buffers["out"].to_numpy(), case.expected["out"])
