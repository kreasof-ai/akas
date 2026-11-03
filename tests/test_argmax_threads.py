"""Wider argmax blocks preserve stable ties and disabled rows."""

import os
import numpy as np
import pytest


@pytest.mark.skipif(
    os.getenv("AKAS_LFM_FP8_CUDA") != "1", reason="native CUDA qualification"
)
@pytest.mark.parametrize("threads", [128, 1024])
def test_argmax_threads_preserve_ties_and_disabled_rows(tmp_path, threads):
    import tensor
    from akas.kernels.definition import definition
    from akas.compilation.selection import selected_source

    spec = dict(kind="argmax", parameters=dict(r=3, c=513), contract="reference")
    caps = dict(provider="cuda", target="sm_89")
    impl, schedule = definition(
        spec, dict(implementation="vocabulary.reference", threads=threads), caps
    )
    source, artifact = tmp_path / "kernel.py", tmp_path / "kernel.tbin"
    source.write_text(selected_source(spec, schedule, caps, impl).render())
    tensor.build(source, artifact, target="sm_89")
    values = np.random.default_rng(31).normal(size=(3, 513)).astype("float32")
    values[0, 1] = values[0, 510] = 10
    with tensor.Device() as device, device.load(artifact) as executable:
        x = device.from_numpy(values.reshape(-1))
        out = device.from_numpy(np.array([-7, -7, -7], dtype="int32"))
        enabled = device.from_numpy(np.array([1, 0, 1], dtype="int32"))
        executable.launch(x, out, enabled)
        np.testing.assert_array_equal(out.to_numpy(), [1, -7, values[2].argmax()])
