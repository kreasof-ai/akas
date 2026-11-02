"""Circular prefill softmax must also span subgroups smaller than 32 lanes."""

import os
import hashlib
import json
import re
import zipfile

import numpy as np
import pytest


@pytest.mark.skipif(
    os.environ.get("AKAS_LFM3_WEBGPU") != "1",
    reason="opt-in Vulkan execution of the small-subgroup attention fallback",
)
@pytest.mark.parametrize("window", [0, 16])
@pytest.mark.parametrize("kind", ["window_attention", "window_grouped"])
def test_window_small_subgroups_match_independent_reference(tmp_path, window, kind):
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import selected_source
    from akas.kernels.definition import definition
    from akas.kernels.registry import validation
    from tensor.artifacts.format import read_artifact
    from tools.context_sweep import vulkan_device

    prefill = kind == "window_attention"
    parameters = dict(
        r=32 if prefill else 1, h=8, kh=2, d=128, capacity=64,
        chunk=32 if prefill else 1, window=window, splits=1,
        query_rows=4 if prefill else 1, key_rows=32,
    )
    spec = dict(kind=kind, parameters=parameters,
                contract="reference", phase="prefill")
    capabilities = dict(provider="webgpu", target="webgpu-portable-v1")
    implementation, schedule = definition(spec, {}, capabilities)
    source, artifact = tmp_path / "attention.py", tmp_path / "attention.tbin"
    source.write_text(selected_source(spec, schedule, capabilities, implementation).render())
    build_artifact(source, artifact, **capabilities)
    manifest, files = read_artifact(artifact)
    # Exercise the Mali path on desktop CI GPUs as well. Only the test shader
    # substitutes subgroup size; production shaders use the actual GPU builtin.
    shader, replacements = re.subn(
        r"\btensorSubgroupSize\b(?!\s*:)", "16u", files["kernel.wgsl"].decode()
    )
    assert replacements > 0
    files["kernel.wgsl"] = shader.encode()
    manifest["files"]["kernel.wgsl"] = hashlib.sha256(files["kernel.wgsl"]).hexdigest()
    with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in files.items():
            archive.writestr(name, data)
    case = validation(kind, parameters, manifest["arguments"])
    with vulkan_device() as device:
        with device.load(artifact) as kernel:
            buffers = {name: device.from_numpy(value) for name, value in case.arguments.items()}
            try:
                kernel.launch(*(buffers[item["name"]] for item in manifest["arguments"]))
                for name, expected in case.expected.items():
                    np.testing.assert_allclose(buffers[name].to_numpy(), expected,
                                               rtol=case.rtol, atol=case.atol)
            finally:
                for buffer in buffers.values():
                    buffer.release()
