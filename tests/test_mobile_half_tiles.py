"""Small register tiles preserve packed projection and gated FFN arithmetic."""

import os

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("AKAS_LFM3_WEBGPU") != "1", reason="opt-in Vulkan projection execution")
@pytest.mark.parametrize("layout,encoding", [(6, 2), (7, 2)])
@pytest.mark.parametrize("kind", ["linear", "ffn"])
def test_mobile_half_tiles_against_independent_reference(tmp_path, layout, encoding, kind):
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import selected_source
    from akas.kernels.definition import definition
    from akas.kernels.registry import validation
    from tensor.artifacts.format import read_artifact
    from tools.context_sweep import vulkan_device

    parameters = dict(r=35, k=128, o=34, ggml=True, type=encoding, half_accum=True)
    spec = dict(kind=kind, parameters=parameters, contract="reference", phase="prefill")
    capabilities = dict(provider="webgpu", target="webgpu-portable-v1")
    requested = dict(implementation="linear.webgpu.short_half", half_layout=layout, half_k=64)
    implementation, schedule = definition(spec, requested, capabilities)
    source, artifact = tmp_path / "projection.py", tmp_path / "projection.tbin"
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
                    actual = buffers[name].to_numpy()
                    if name in case.comparators:
                        case.comparators[name](actual, expected)
                    else:
                        np.testing.assert_allclose(actual, expected, rtol=case.rtol, atol=case.atol)
            finally:
                for buffer in buffers.values():
                    buffer.release()
