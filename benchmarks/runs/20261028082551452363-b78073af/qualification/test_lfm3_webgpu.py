"""LFM3 Vulkan storage and independent equations, including circular KV reuse."""

import os
import numpy as np
import pytest
from akas.models.lfm3 import LFM3, LFM3Checkpoint
from support.lfm3 import gguf_fixture, oracle


def test_webgpu_gguf_declares_word_storage(tmp_path):
    checkpoint = LFM3Checkpoint(gguf_fixture(tmp_path / "model.gguf"), tokenizer=False)
    graph = LFM3.compose(
        checkpoint,
        rows=1,
        chunk=32,
        max_requests=1,
        max_context=128,
        prefill_chunk=32,
        provider="webgpu",
    ).graph
    weight = graph.buffers["token_embd.weight"]
    assert weight.dtype == "uint32"
    assert weight.shape[0] * 4 == checkpoint.gguf.tensors[weight.name].nbytes
    for call in graph.calls:
        if call.kind == "window_attention":
            assert call.parameters["query_rows"] == 1
            assert call.parameters["key_rows"] == 32


@pytest.mark.skipif(
    not os.getenv("AKAS_LFM3_WEBGPU"), reason="opt-in hardware Vulkan qualification"
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("arithmetic", [None, "mixed"])
def test_webgpu_model_matches_independent_equations(tmp_path, head_dim, arithmetic):
    from pathlib import Path
    from akas.compilation import compile_checkpoint
    from akas.compilation.tuning import SearchSettings
    from tools.context_sweep import vulkan_device

    path = gguf_fixture(tmp_path / "model.gguf", width=4 * head_dim, head_dim=head_dim)
    checkpoint = LFM3Checkpoint(path, tokenizer=False)
    cache = os.getenv("AKAS_LFM3_WEBGPU_CACHE")
    bundle = (
        Path(cache) / f"{head_dim}-{arithmetic or 'float'}"
        if cache
        else tmp_path / "program"
    )
    rng = np.random.default_rng(18)
    seqs = [rng.integers(0, 64, n).tolist() for n in (81, 67)]

    def compare(actual, seq):
        expected = oracle(checkpoint, seq)
        np.testing.assert_allclose(actual, expected, rtol=0.03, atol=0.003)
        assert actual.argmax() == expected.argmax()

    with vulkan_device() as device:
        compile_checkpoint(
            LFM3,
            path,
            bundle,
            checkpoint_options={"tokenizer": False},
            geometries=[(2, 1), (2, 32)],
            max_requests=4,
            max_context=128,
            model_options={"prefill_chunk": 32, "activation_quantization": arithmetic},
            provider="webgpu",
            device=device,
            search_settings=SearchSettings(budget=1, samples=1, repeats=1),
        )
        with LFM3(
            path, bundle, device, checkpoint_options={"tokenizer": False}, graphs=True
        ) as model:
            slots = [model.admit_at(3), model.admit_at(1)]
            predictions = model.prefill(slots, seqs, chunk=32)
            actual = (
                model.batches.select(2, 32).feature("logits").to_numpy().reshape(2, 64)
            )
            for row, seq in enumerate(seqs):
                compare(actual[row], seq)
            model.decode(slots[::-1], predictions[::-1])
            actual = (
                model.batches.select(2, 1).feature("logits").to_numpy().reshape(2, 64)
            )
            for row, index in enumerate((1, 0)):
                compare(actual[row], seqs[index] + [predictions[index]])
            model.release(slots[0])
            reused = model.admit_at(3)
            model.prefill([reused, slots[1]], [[3, 5, 7], [9]], chunk=32)
            actual = (
                model.batches.select(2, 32).feature("logits").to_numpy().reshape(2, 64)
            )
            compare(actual[0], [3, 5, 7])
