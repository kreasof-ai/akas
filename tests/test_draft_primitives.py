"""Draft data movement remains functional, exact and shared across providers."""

import os
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize(
    "kind,p",
    [
        ("select_token", dict(r=2, chunk=3, c=5, index=1)),
        ("concatenate", dict(r=2, left=3, right=5)),
        ("replace_token", dict(r=2, chunk=5, index=4)),
        ("block_limits", dict(r=2, chunk=5)),
        ("sigmoid_bias", dict(r=7)),
    ],
)
def test_draft_primitive_gpu_matches_independent_numpy_and_preserves_sources(
    tmp_path, provider, kind, p
):
    if os.environ.get("AKAS_LFM_MOE_" + provider.upper()) != "1":
        pytest.skip("opt-in draft primitive GPU qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import selected_source
    from akas.kernels.definition import definition
    from akas.kernels.registry import validation

    spec = dict(kind=kind, parameters=p, contract="reference", phase="proposal")
    caps = dict(
        provider=provider,
        target="sm_89" if provider == "cuda" else "webgpu-portable-v1",
    )
    implementation, schedule = definition(spec, {}, caps)
    source = tmp_path / (kind + ".py")
    source.write_text(selected_source(spec, schedule, caps, implementation).render())
    artifact = source.with_suffix(".tbin")
    build_artifact(source, artifact, provider=provider, target=caps["target"])
    signature, _ = read_artifact(artifact)
    case = validation(kind, p, signature["arguments"])
    with tensor.Device(provider=provider) as device:
        buffers = {
            name: device.from_numpy(value) for name, value in case.arguments.items()
        }
        with device.load(artifact) as module:
            module.launch(*(buffers[a["name"]] for a in module.manifest["arguments"]))
        for name, expected in case.expected.items():
            actual = buffers[name].to_numpy()
            if name != "out" or expected.dtype.kind in "iu":
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(
                    actual, expected, rtol=case.rtol, atol=case.atol
                )
