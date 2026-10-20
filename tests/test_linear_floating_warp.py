"""CUDA F32 projection keeps the reference geometry, tails, and output views."""

from contextlib import ExitStack
import os
import numpy as np
import pytest


@pytest.mark.parametrize(
    "rows,depth,columns,matrix", [(1, 257, 7, False), (33, 256, 64, True)]
)
def test_floating_warp_projection(tmp_path, rows, depth, columns, matrix):
    if os.environ.get("AKAS_LFM_MOE_CUDA") != "1":
        pytest.skip("opt-in CUDA floating projection qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.kernels.registry import validation

    p = dict(
        r=rows,
        k=depth,
        o=columns,
        output_stride=columns + 3,
        output_offset=2,
        matrix_output=matrix,
    )
    entry, artifact = tmp_path / "kernel.py", tmp_path / "kernel.tbin"
    entry.write_text(
        source(
            "linear",
            p,
            contract="reference",
            provider="cuda",
            target="sm_89",
            preferred="linear.cuda.floating_warp",
        ).render()
    )
    build_artifact(entry, artifact, provider="cuda", target="sm_89")
    meta, _ = read_artifact(artifact)
    case = validation("linear", p, meta["arguments"])
    with tensor.Device(provider="cuda") as device, ExitStack() as stack:
        buffers = {name: device.from_numpy(a) for name, a in case.arguments.items()}
        for buffer in buffers.values():
            stack.callback(buffer.release)
        module = stack.enter_context(device.load(artifact))
        module.launch(*(buffers[a["name"]] for a in meta["arguments"]))
        for name, buffer in buffers.items():
            np.testing.assert_allclose(
                buffer.to_numpy(), case.expected[name], rtol=case.rtol, atol=case.atol
            )
