from types import SimpleNamespace

import pytest

from tools.spark_mfu import estimate, prefill_flops


def test_prefill_counts_window_pairs_and_only_one_vocabulary_projection():
    c = SimpleNamespace(
        width=2,
        ff=3,
        heads=2,
        kv_heads=1,
        head_dim=4,
        vocab=7,
        window=3,
        layers=("sliding_attention", "full_attention", "sliding_attention"),
    )
    # Per-layer projection parameters: 32 QKV + 4 gate + 16 output + 18 FFN.
    # Six causal queries see 21 full pairs or 1+2+3+3+3+3=15 window pairs.
    assert prefill_flops(c, 6) == 4180
    report = dict(
        profiles=[
            dict(
                runtime="akas", measurements=[dict(batch=1, prefix=6, prefill_tok_s=6)]
            )
        ]
    )
    assert estimate(report, c, peak_tflops=1)["rows"][0][
        "mfu_percent"
    ] == pytest.approx(4180e-10)
    with pytest.raises(ValueError):
        estimate(report, c, peak_tflops=float("nan"))
    with pytest.raises(ValueError):
        prefill_flops(c, 0)
