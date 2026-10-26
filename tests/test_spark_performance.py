"""Independent qualification of Spark's faster projections and attention."""

from contextlib import ExitStack
import os

import numpy as np
import pytest


@pytest.mark.skipif(
    os.getenv("AKAS_SPARK_CUDA") != "1", reason="opt-in CUDA qualification"
)
@pytest.mark.parametrize(
    "kind,parameters",
    [
        (
            "quantized_linear",
            dict(
                r=32,
                k=10240,
                o=2560,
                weight_format="fp8",
                weight_options={"scale_layout": "channel"},
            ),
        ),
        *[
            (
                "quantized_linear",
                dict(
                    r=r,
                    k=64,
                    o=143,
                    weight_format="fp8",
                    weight_options={"scale_layout": "channel"},
                ),
            )
            for r in (1, 133)
        ],
        (
            "linear",
            dict(r=1, k=768, o=137, ggml=True, type=8, optimized=True, vector_q8=True),
        ),
        *[
            (
                "linear",
                dict(
                    r=32,
                    k=512,
                    o=16 if q == 30 else 137,
                    ggml=True,
                    type=q,
                    optimized=True,
                    short_packed=True,
                ),
            )
            for q in (8, 12, 14, 30)
        ],
        *[
            (
                "window_attention",
                dict(
                    r=256,
                    h=16,
                    kh=4,
                    d=256,
                    capacity=1536 if window else 8192,
                    window=window,
                    splits=1,
                    chunk=128,
                    query_rows=64,
                    key_rows=32,
                    contiguous=not window,
                ),
            )
            for window in (0, 512)
        ],
        (
            "rotary_table",
            dict(r=7, h=4, d=256, rotary=64, theta=5000000.0, capacity=131072),
        ),
        (
            "window_grouped",
            dict(r=2, h=16, kh=4, d=256, capacity=1536, window=512, splits=4, chunk=1),
        ),
        (
            "window_grouped",
            dict(r=2, h=16, kh=4, d=256, capacity=8192, window=0, splits=32, chunk=1),
        ),
    ],
)
def test_cuda_numerical_reference(tmp_path, kind, parameters):
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.kernels.registry import validation

    entry, artifact = tmp_path / "kernel.py", tmp_path / "kernel.tbin"
    entry.write_text(
        source(
            kind, parameters, contract="reference", provider="cuda", target="sm_89"
        ).render()
    )
    build_artifact(entry, artifact, provider="cuda", target="sm_89")
    meta, _ = read_artifact(artifact)
    case = validation(kind, parameters, meta["arguments"])
    with tensor.Device() as device, ExitStack() as stack:
        dtypes = {a["name"]: a["dtype"] for a in meta["arguments"]}
        buffers = {
            name: device.from_numpy(
                a, **({"dtype": "bfloat16"} if dtypes[name] == "bfloat16" else {})
            )
            for name, a in case.arguments.items()
        }
        for buffer in buffers.values():
            stack.callback(buffer.release)
        module = stack.enter_context(device.load(artifact))
        module.launch(*(buffers[a["name"]] for a in meta["arguments"]))
        np.testing.assert_allclose(
            buffers["out"].to_numpy(),
            case.expected["out"],
            rtol=case.rtol,
            atol=case.atol,
        )
