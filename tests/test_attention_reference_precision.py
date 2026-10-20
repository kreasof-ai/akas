"""Large independent attention references keep bounded F32 working arrays."""

import numpy as np
import pytest


@pytest.mark.parametrize("precision", [None, "float16"])
def test_f32_attention_oracle_agrees_with_f64(monkeypatch, precision):
    import akas.ops.attention.validation as reference

    p = dict(
        r=8,
        h=4,
        kh=1,
        d=64,
        capacity=127,
        chunk=4,
        kv_dtype="float16",
        provider="cuda",
        optimized=True,
        unit_scales=True,
        key_rows=32,
    )
    if precision:
        p["prefill_precision"] = precision
    signature = [
        dict(name=n, shape=s, dtype=d)
        for n, s, d in [
            ("q", (8, 256), "float32"),
            ("kc", (2, 1, 127, 64), "float16"),
            ("vc", (2, 1, 127, 64), "float16"),
            ("ks", (2, 1, 127, 1), "float32"),
            ("vs", (2, 1, 127, 1), "float32"),
            ("out", (8 * 256,), "float32"),
            ("positions", (8,), "int32"),
            ("enabled", (8,), "int32"),
        ]
    ]
    monkeypatch.setattr(reference, "reference_dtype", lambda *args: "float64")
    wide = reference.canonical_validation("attention", p, signature)
    monkeypatch.setattr(reference, "reference_dtype", lambda *args: "float32")
    bounded = reference.canonical_validation("attention", p, signature)
    np.testing.assert_allclose(
        bounded.expected["out"], wide.expected["out"], rtol=0.0002, atol=0.000002
    )
    assert (bounded.rtol, bounded.atol) == (wide.rtol, wide.atol)
    for name in bounded.arguments:
        np.testing.assert_array_equal(bounded.arguments[name], wide.arguments[name])


def test_reference_precision_changes_only_for_large_prefill():
    from akas.ops.attention.validation import reference_dtype

    assert reference_dtype(4096, 32, 131072, 4096) == "float32"
    assert reference_dtype(512, 32, 131072, 512) == "float32"
    assert reference_dtype(8, 4, 127, 4) == "float64"
    assert reference_dtype(16, 32, 131072, 1) == "float64"
