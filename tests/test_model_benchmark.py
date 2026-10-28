"""Matched benchmark acceptance uses both phase medians and every shape."""

import pytest
from tools.model_benchmark import phase_parity


def report(prefill, decode, prefix=32):
    return dict(
        results=[
            dict(
                case=dict(prompt=prefix),
                status="ok",
                samples=[
                    dict(prefill_tokens_per_second=p, decode_tokens_per_second=d)
                    for p, d in zip(prefill, decode)
                ],
            )
        ]
    )


def test_phase_parity_does_not_hide_decode_regression_or_use_outlier():
    a = report([110, 111, 1000], [80, 81, 1000])
    b = report([100, 100, 100], [90, 90, 90])
    gate = phase_parity(a, b)
    assert not gate["passed"]
    assert gate["rows"][0]["prefill"]["parity_or_better"]
    assert not gate["rows"][0]["decode"]["parity_or_better"]


def test_phase_parity_requires_matching_shapes_and_accepts_exact_parity():
    a = report([100], [90])
    assert phase_parity(a, a)["passed"]
    with pytest.raises(ValueError, match="identical"):
        phase_parity(a, report([100], [90], 512))


def test_decode_priority_keeps_prefill_visible_but_gates_decode():
    a, b = report([50], [110]), report([100], [100])
    result = phase_parity(a, b, ("decode",))
    assert result["passed"]
    assert not result["rows"][0]["prefill"]["parity_or_better"]
    with pytest.raises(ValueError):
        phase_parity(a, b, ())
