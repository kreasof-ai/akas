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
                "linear",
                dict(
                    r=133,
                    k=512,
                    o=137,
                    ggml=True,
                    type=q,
                    optimized=True,
                    tiled_packed=True,
                ),
            )
            for q in (8, 12, 14, 30)
        ],
        *[
            (
                "geglu_linear",
                dict(r=133, k=512, o=137, ggml=True, type=q),
            )
            for q in (8, 12, 14, 30)
        ],
        *[
            (
                kind,
                dict(
                    r=133,
                    k=512,
                    o=137,
                    ggml=True,
                    type=q,
                    input_dtype="float16",
                    **extra,
                ),
            )
            for q in (8, 12, 14, 30)
            for kind, extra in (
                ("linear", dict(optimized=True, tiled_packed=True)),
                ("geglu_linear", dict(output_dtype="float16")),
            )
        ],
        (
            "rms",
            dict(
                r=133,
                c=2560,
                eps=1e-6,
                output_dtype="float16",
                algorithm="fragment",
                flat=True,
            ),
        ),
        ("head_gate_half", dict(r=133, h=16, d=256)),
        (
            "add_rms",
            dict(
                r=133,
                c=2560,
                eps=1e-6,
                output_dtype="float16",
                residual_dtype="float32",
                algorithm="fragment",
                flat=True,
            ),
        ),
        *[
            (
                "qkv_rotary_cache",
                dict(
                    r=256,
                    h=16,
                    kh=4,
                    d=256,
                    chunk=128,
                    capacity=1536 if window else 131072,
                    window=window,
                    context=131072,
                    stride=1024,
                    blocks=128,
                    rotary=rot,
                    theta=theta,
                ),
            )
            for window in (0, 512)
            for rot, theta in ((64, 5000000.0), (256, 10000.0))
        ],
        *[
            (
                "window_attention",
                dict(
                    r=256,
                    h=16,
                    kh=4,
                    d=256,
                    chunk=128,
                    splits=1,
                    capacity=1536 if window else 8192,
                    window=window,
                    query_rows=64,
                    key_rows=32,
                    contiguous=not window,
                    query_dtype="float16",
                ),
            )
            for window in (0, 512)
        ],
        *[
            (
                "rotary_compact",
                dict(
                    r=8,
                    h=4,
                    d=256,
                    rotary=rot,
                    theta=theta,
                    capacity=cap,
                    stride=1024,
                    blocks=(cap + 1023) // 1024,
                ),
            )
            for rot, theta in ((64, 5000000.0), (256, 10000.0))
            for cap in (131072, 1048576)
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
        for name, expected in case.expected.items():
            if expected is not case.arguments[name]:
                if name in case.comparators:
                    case.comparators[name](buffers[name].to_numpy(), expected)
                else:
                    np.testing.assert_allclose(
                        buffers[name].to_numpy(),
                        expected,
                        rtol=case.rtol,
                        atol=case.atol,
                    )


@pytest.mark.skipif(
    os.getenv("AKAS_SPARK_CUDA") != "1", reason="opt-in compact sliding cache"
)
@pytest.mark.parametrize(
    "kind,parameters",
    [
        (
            "qkv_rotary_cache",
            dict(
                r=256,
                h=8,
                kh=2,
                d=64,
                capacity=128,
                chunk=128,
                window=32,
                rotary=16,
                theta=5000000.0,
                context=131072,
                stride=1024,
                blocks=128,
                current_chunk=True,
            ),
        ),
        (
            "window_attention_current",
            dict(
                r=256,
                h=8,
                kh=2,
                d=64,
                capacity=32,
                chunk=128,
                window=32,
                splits=1,
                query_rows=64,
                key_rows=32,
                query_dtype="float16",
            ),
        ),
        (
            "window_attention_current",
            dict(
                r=256,
                h=16,
                kh=4,
                d=256,
                capacity=512,
                chunk=128,
                window=512,
                splits=1,
                query_rows=64,
                key_rows=32,
                query_dtype="float16",
            ),
        ),
        (
            "window_cache_tail",
            dict(r=256, h=8, kh=2, d=64, capacity=32, chunk=128, window=32),
        ),
        (
            "window_attention",
            dict(
                r=256,
                h=8,
                kh=2,
                d=64,
                capacity=160,
                chunk=128,
                window=32,
                splits=1,
                query_rows=64,
                key_rows=32,
                query_dtype="float16",
            ),
        ),
        (
            "window_attention",
            dict(
                r=256,
                h=16,
                kh=4,
                d=256,
                capacity=8192,
                chunk=128,
                window=0,
                splits=1,
                query_rows=64,
                key_rows=32,
                query_dtype="float16",
                contiguous=True,
            ),
        ),
    ],
)
def test_compact_sliding_operators(tmp_path, kind, parameters):
    test_cuda_numerical_reference(tmp_path, kind, parameters)
