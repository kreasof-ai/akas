"""Q6_K decode must cover every coefficient with 16-lane ownership."""

import os

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("AKAS_LFM3_WEBGPU") != "1", reason="opt-in Vulkan Q6 execution")
@pytest.mark.parametrize("kind", ["linear", "ffn"])
@pytest.mark.parametrize("vector", [False, True])
def test_q6_sixteen_lanes_match_independent_reference(tmp_path, kind, vector):
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import selected_source
    from akas.kernels.definition import definition
    from akas.kernels.registry import validation
    from tensor.artifacts.format import read_artifact
    from tools.context_sweep import vulkan_device

    parameters = dict(r=1, k=512, o=35, ggml=True, type=14)
    spec = dict(kind=kind, parameters=parameters, contract="reference", phase="decode")
    capabilities = dict(provider="webgpu", target="webgpu-portable-v1")
    requested = (dict(implementation="linear.webgpu.packed:vector-decode", decode_layout=2)
                 if vector else dict(implementation="linear.webgpu.packed:quantized-decode",
                                     gemv_lanes=16, gemv_threads=128, sg=True))
    implementation, schedule = definition(spec, requested, capabilities)
    source, artifact = tmp_path / "q6.py", tmp_path / "q6.tbin"
    source.write_text(selected_source(spec, schedule, capabilities, implementation).render())
    build_artifact(source, artifact, **capabilities)
    manifest, _ = read_artifact(artifact)
    case = validation(kind, parameters, manifest["arguments"])
    with vulkan_device() as device:
        with device.load(artifact) as kernel:
            buffers = {name: device.from_numpy(value) for name, value in case.arguments.items()}
            try:
                kernel.launch(*(buffers[item["name"]] for item in manifest["arguments"]))
                for name, expected in case.expected.items():
                    np.testing.assert_allclose(buffers[name].to_numpy(), expected, rtol=case.rtol, atol=case.atol)
            finally:
                for buffer in buffers.values():
                    buffer.release()
