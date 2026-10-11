"""Packed WebGPU inference kernels preserve fused arithmetic and masking."""

import os
import numpy as np
import pytest
import tensor

NATIVE = pytest.mark.skipif(
    os.environ.get("TENSOR_WEBGPU") != "1", reason="requires a native WebGPU adapter"
)


@NATIVE
@pytest.mark.parametrize("force_fallback", [False, True])
@pytest.mark.parametrize("shared_input", [False, True])
@pytest.mark.parametrize(
    "depth,unroll,accumulators", [(256, 2, 2), (1024, 8, 8), (2560, 20, 4)]
)
def test_streamed_gemv_fused_tail_and_f32_activation_contract(
    tmp_path, force_fallback, shared_input, depth, unroll, accumulators
):
    from akas.compilation.source import export_kernel as source

    p = dict(
        r=1,
        k=depth,
        o=7,
        type=1,
        sg=True,
        decode_schedule="streamed",
        lanes=32,
        threads=64,
        micro_rows=2,
        dot_width=4,
        unroll=unroll,
        accumulators=accumulators,
        k_layout="striped",
        shared_input=shared_input,
    )
    text = source(
        "akas.compilation.factories.packed", "packed_webgpu_factory", "ffn", p
    )
    if force_fallback:
        from benchmarks.lfm2.dsl_probes import subgroup_source

        text = subgroup_source("ffn", p, 0)
    path = tmp_path / "gemv.py"
    artifact = path.with_suffix(".tbin")
    path.write_text(text)
    tensor.build(path, artifact, provider="webgpu")
    rng = np.random.default_rng(172)
    weights = [rng.normal(size=(7, depth)).astype(np.float16) for _ in range(2)]
    with tensor.Device(provider="webgpu") as device:
        kernel = device.load(artifact)
        ws = [device.from_numpy(w.ravel()) for w in weights]
        for scale in (1, 0.01, 1e-5):
            x = (rng.normal(size=depth) * scale).astype(np.float32)
            g, u = [w.astype(np.float64) @ x.astype(np.float64) for w in weights]
            expected = g / (1 + np.exp(-g)) * u
            out = device.from_numpy(np.full(7, np.nan, np.float32))
            inp = device.from_numpy(x)
            kernel.launch(inp, *ws, out)
            actual = out.to_numpy()
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=1e-7)
            inp.release()
            out.release()


@NATIVE
@pytest.mark.parametrize("force_fallback", [False, True])
def test_fused_attention_active_tail_and_qk_fallback(tmp_path, force_fallback):
    from akas.compilation.source import export_kernel as source

    h, kh, d, cap = 4, 2, 64, 96
    text = source(
        "akas.compilation.factories.packed",
        "packed_webgpu_factory",
        "attention",
        dict(
            r=1,
            h=h,
            kh=kh,
            d=d,
            cap=cap,
            sg=True,
            attention_schedule="partitioned_values",
            channels=32,
            value_parts=16,
            fused_scores=True,
        ),
    )
    if force_fallback:
        from benchmarks.lfm2.dsl_probes import subgroup_source

        text = subgroup_source(
            "attention",
            dict(
                r=1,
                h=h,
                kh=kh,
                d=d,
                cap=cap,
                sg=True,
                attention_schedule="partitioned_values",
                channels=32,
                value_parts=16,
                fused_scores=True,
            ),
            0,
        )
    path = tmp_path / "attention.py"
    path.write_text(text)
    artifact = tmp_path / "attention.tbin"
    tensor.build(path, artifact, provider="webgpu")
    with tensor.Device(provider="webgpu") as device:
        kernel = device.load(artifact)
        rng = np.random.default_rng(314)
        q = rng.normal(size=(h, d)).astype(np.float32)
        keys = rng.normal(size=(cap, kh, d)).astype(np.float16)
        values = rng.normal(size=(cap, kh, d)).astype(np.float16)
        qb = device.from_numpy(q.ravel())
        kb = device.from_numpy(keys.ravel())
        vb = device.from_numpy(values.ravel())
        out = device.full(h * d, np.nan)
        control = device.zeros(2, "int32")
        for position in (0, 31, 95):
            k = keys.copy()
            v = values.copy()
            k[position + 1 :] = np.nan
            v[position + 1 :] = np.nan
            device.write(kb, k.ravel())
            device.write(vb, v.ravel())
            device.write(control, np.array([position, 1], np.int32))
            device.write(out, np.full(h * d, np.nan, np.float32))
            kernel.launch(qb, kb, vb, out, control)
            actual = out.to_numpy().reshape(h, d)
            k = (
                keys[: position + 1]
                .astype(np.float64)[:, np.arange(h) // 2, :]
                .transpose(1, 0, 2)
            )
            v = (
                values[: position + 1]
                .astype(np.float64)[:, np.arange(h) // 2, :]
                .transpose(1, 0, 2)
            )
            scores = (k * q.astype(np.float64)[:, None, :]).sum(axis=2) * d**-0.5
            prob = np.exp(scores - scores.max(axis=1, keepdims=True))
            prob /= prob.sum(axis=1, keepdims=True)
            expected = (prob[:, :, None] * v).sum(axis=1)
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=3e-7)
