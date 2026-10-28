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
            assert call.parameters["query_rows"] == 4
            assert call.parameters["key_rows"] == 32


def test_mixed_arithmetic_keeps_batched_decode_floating(tmp_path):
    checkpoint = LFM3Checkpoint(gguf_fixture(tmp_path / "model.gguf"), tokenizer=False)
    graph = LFM3.compose(
        checkpoint, rows=2, chunk=1, max_requests=2, max_context=128,
        provider="webgpu", activation_quantization="mixed",
    ).graph
    assert not any(call.kind == "quantize_q16" for call in graph.calls)
    for call in graph.calls:
        if call.kind in ("linear", "ffn"):
            assert not call.parameters.get("q16")
            assert not call.parameters.get("half_accum")


@pytest.mark.skipif(
    not os.getenv("AKAS_LFM3_WEBGPU"), reason="opt-in hardware Vulkan qualification"
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("arithmetic,chunk", [(None, 32), ("mixed", 32), ("mixed", 64)])
def test_webgpu_model_matches_independent_equations(tmp_path, head_dim, arithmetic, chunk):
    from pathlib import Path
    from akas.compilation import compile_checkpoint
    from akas.compilation.tuning import SearchSettings
    from tools.context_sweep import vulkan_device
    from tensor.compiler.search import ScheduleProfile

    path = gguf_fixture(tmp_path / "model.gguf", width=4 * head_dim, head_dim=head_dim)
    checkpoint = LFM3Checkpoint(path, tokenizer=False)
    cache = os.getenv("AKAS_LFM3_WEBGPU_CACHE")
    bundle = (
        Path(cache) / f"{head_dim}-{arithmetic or 'float'}-{chunk}"
        if cache
        else tmp_path / "program"
    )
    rng = np.random.default_rng(18)
    seqs = [rng.integers(0, 64, n).tolist() for n in (81, 67)]

    def compare(actual, seq):
        expected = oracle(checkpoint, seq)
        if arithmetic == "mixed":
            # Short F16 sums and two-component activation quantization can
            # move near-zero logits across zero. Bound the full vector and
            # peak error against the independent floating equations instead.
            assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.01
            assert np.max(np.abs(actual - expected)) / np.max(np.abs(expected)) < 0.01
        else:
            np.testing.assert_allclose(actual, expected, rtol=0.03, atol=0.003)
        assert actual.argmax() == expected.argmax()

    with vulkan_device() as device:
        compile_checkpoint(
            LFM3,
            path,
            bundle,
            checkpoint_options={"tokenizer": False},
            geometries=[(2, 1), (2, chunk)],
            max_requests=4,
            max_context=128,
            model_options={
                "prefill_chunk": chunk,
                "activation_quantization": arithmetic,
                "fused_sandwich": True,
                "q4_layout": "planar",
                "decode_splits": 32 if arithmetic is None else 128,
                "sliding_splits": 8,
                "ring_power_of_two": True,
                "fused_rotary_norm": True,
            },
            provider="webgpu",
            device=device,
            schedules=ScheduleProfile(
                dict(
                    schema="tensor.schedule-profile.v1",
                    provider="webgpu",
                    target="webgpu-portable-v1",
                    entries=[
                        dict(
                            operation="window_grouped",
                            parameters={},
                            schedule=dict(
                                implementation="attention.webgpu.window_blocked",
                            key_unroll=8 if head_dim == 128 and arithmetic is None else 4,
                            ),
                        ),
                        dict(
                            operation="window_merge",
                            parameters={},
                            schedule=dict(
                                implementation="attention.webgpu.window_merge_subgroup"
                            ),
                        ),
                        dict(
                            operation="rms",
                            parameters={"c": 128},
                            schedule=dict(
                                implementation="normalization.webgpu.parallel-subgroup",
                                sg=True,
                            ),
                        ),
                        dict(
                            operation="post_add_rms",
                            parameters={},
                            schedule=dict(
                                implementation="normalization.webgpu.post_add_rms",
                                threads=min(512, 4 * head_dim),
                            ),
                        ),
                    ],
                )
            ),
            search_settings=SearchSettings(budget=1, samples=1, repeats=1),
        )
        with LFM3(
            path, bundle, device, checkpoint_options={"tokenizer": False}, graphs=True
        ) as model:
            slots = [model.admit_at(3), model.admit_at(1)]
            predictions = model.prefill(slots, seqs, chunk=chunk)
            actual = (
                model.batches.select(2, chunk).feature("logits").to_numpy().reshape(2, 64)
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
            model.prefill([reused, slots[1]], [[3, 5, 7], [9]], chunk=chunk)
            actual = (
                model.batches.select(2, chunk).feature("logits").to_numpy().reshape(2, 64)
            )
            compare(actual[0], [3, 5, 7])
