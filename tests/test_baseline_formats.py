"""Baseline weight identity is independent of model execution/cache dtype."""

import pytest
from tools.baselines.__main__ import select_weight_format


def test_native_fp8_selects_original_bytes_without_bf16_alias():
    model = {"weights": {"fp8": {"path": "model"}}}
    assert select_weight_format(model, "vllm") == "fp8"
    assert select_weight_format(model, "sglang", "fp8") == "fp8"
    with pytest.raises(ValueError, match="does not declare"):
        select_weight_format(model, "vllm", "bf16")


def test_existing_defaults_and_incompatible_native_encodings():
    model = {"weights": {"bf16": {}, "q4_k_m": {}}}
    assert select_weight_format(model, "vllm") == "bf16"
    assert select_weight_format(model, "llama") == "q4_k_m"
    with pytest.raises(ValueError, match="safetensors"):
        select_weight_format(model, "sglang", "q4_k_m")
