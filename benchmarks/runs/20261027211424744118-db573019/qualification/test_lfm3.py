"""LFM3 architecture, storage and independent end-to-end qualification."""

import json
import os
import numpy as np
import pytest
from akas.models.lfm3 import LFM3, LFM3Config, LFM3Checkpoint
from support.lfm3 import fixture, oracle


@pytest.mark.parametrize("fp8", [False, True])
def test_formats_and_bounded_window(tmp_path, fp8):
    c = LFM3Checkpoint(fixture(tmp_path, fp8=fp8), tokenizer=False)
    g = LFM3.compose(
        c, rows=2, chunk=32, max_requests=4, max_context=128, prefill_chunk=32
    ).graph
    assert g.buffers["blk.2.kc"].shape == (4, 1, 48, 64)
    assert g.buffers["blk.1.kc"].shape == (4, 1, 128, 64)
    assert g.buffers["blk.0.conv_state"].shape == (4, 6, 128)
    assert set(g.request["reset"]) == {
        "positions",
        "blk.0.conv_state",
        "blk.3.conv_state",
    }
    assert all(
        x.parameters["full_head_pairs"] for x in g.calls if x.kind == "rotary_compact"
    )
    if fp8:
        assert c.inventory()["encodings"]["F8_E4M3"] == 20
        assert g.buffers["blk.1.attn_q.weight"].dtype == "uint8"
        assert g.buffers["blk.1.attn_q.weight.scale"].shape == (1, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("conv_bias", True),
        ("num_key_value_heads", 3),
        ("conv_L_cache", 1),
        ("layer_types", ["conv"]),
        ("use_sandwich_norm", "yes"),
        ("rope_parameters", {"full_attention": {"rope_type": "linear"}}),
    ],
)
def test_invalid_geometry(tmp_path, field, value):
    fixture(tmp_path)
    d = json.loads((tmp_path / "config.json").read_text())
    with pytest.raises(ValueError):
        LFM3Config.from_huggingface({**d, field: value})


def test_bad_scale_rejected(tmp_path):
    fixture(tmp_path, fp8=True)
    d = json.loads((tmp_path / "config.json").read_text())
    d["quantization_config"]["config_groups"]["FP8_BLOCK"]["weights"][
        "block_structure"
    ] = [64, 128]
    (tmp_path / "config.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="128x128"):
        LFM3Checkpoint(tmp_path, tokenizer=False)


@pytest.mark.skipif(
    os.getenv("AKAS_LFM3_CUDA") != "1", reason="opt-in LFM3 CUDA qualification"
)
@pytest.mark.parametrize("weight_format", ["bf16", "fp8", "gguf", "gguf-half128"])
def test_cuda_matches_independent_equations_reorder_wrap_and_reuse(
    tmp_path, weight_format
):
    import tensor
    from akas.compilation import compile_checkpoint
    from akas.compilation.tuning import SearchSettings

    from support.lfm3 import gguf_fixture

    path = (
        gguf_fixture(
            tmp_path / "model.gguf",
            width=512 if weight_format == "gguf-half128" else 128,
        )
        if weight_format.startswith("gguf")
        else fixture(tmp_path / "checkpoint", fp8=weight_format == "fp8")
    )
    checkpoint = LFM3Checkpoint(path, tokenizer=False)
    from pathlib import Path

    cache = os.getenv("AKAS_LFM3_PROGRAM_CACHE")
    bundle = Path(cache) / weight_format if cache else tmp_path / "program"
    chunk = 128 if weight_format == "gguf-half128" else 32
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
            "half_activations": weight_format == "gguf-half128",
            "q4_layout": "planar" if weight_format == "gguf-half128" else "raw",
        },
        search_settings=SearchSettings(budget=1, samples=1, repeats=1),
    )
    rng = np.random.default_rng(18)
    seqs = [rng.integers(0, 64, n).tolist() for n in (81, 67)]
    with (
        tensor.Device() as device,
        LFM3(path, bundle, device, checkpoint_options={"tokenizer": False}) as model,
    ):
        slots = [model.admit_at(3), model.admit_at(1)]
        predictions = model.prefill(slots, seqs, chunk=chunk)
        actual = (
            model.batches.select(2, chunk).feature("logits").to_numpy().reshape(2, 64)
        )
        for row, seq in enumerate(seqs):
            expected = oracle(checkpoint, seq)
            np.testing.assert_allclose(actual[row], expected, rtol=0.03, atol=0.003)
            assert predictions[row] == expected.argmax()
        model.decode(slots[::-1], predictions[::-1])
        actual = model.batches.select(2, 1).feature("logits").to_numpy().reshape(2, 64)
        for row, index in enumerate((1, 0)):
            np.testing.assert_allclose(
                actual[row],
                oracle(checkpoint, seqs[index] + [predictions[index]]),
                rtol=0.03,
                atol=0.003,
            )
        model.release(slots[0])
        reused = model.admit_at(3)
        model.prefill([reused, slots[1]], [[3, 5, 7], [9]], chunk=chunk)
        actual = (
            model.batches.select(2, chunk).feature("logits").to_numpy().reshape(2, 64)
        )
        np.testing.assert_allclose(
            actual[0], oracle(checkpoint, [3, 5, 7]), rtol=0.03, atol=0.003
        )


def test_gguf_inventory_and_packed_storage(tmp_path):
    from support.lfm3 import gguf_fixture

    c = LFM3Checkpoint(gguf_fixture(tmp_path / "model.gguf"), tokenizer=False)
    assert c.config.rotary[1] == 32
    g = LFM3.compose(
        c, rows=1, chunk=32, max_requests=1, max_context=128, prefill_chunk=32
    ).graph
    assert g.buffers["blk.1.attn_q.weight"].dtype == "uint8"
    for call in g.calls:
        if call.kind == "linear":
            assert call.parameters["ggml"]


@pytest.mark.skipif(
    not os.getenv("AKAS_LFM3_CHECKPOINT"),
    reason="local official tokenizer qualification",
)
def test_official_tokenizer_and_chat_match_template():
    from pathlib import Path
    from tokenizers import Tokenizer as Reference
    from jinja2 import Environment

    root = Path(os.environ["AKAS_LFM3_CHECKPOINT"])
    reference = Reference.from_file(str(root / "bf16/tokenizer.json"))
    import re

    template_text = (root / "bf16/chat_template.jinja").read_text()
    template = Environment().from_string(
        re.sub(r"\{%-?\s*(?:end)?generation\s*-?%\}", "", template_text)
    )
    for source in (root / "bf16", root / "fp8", root / "q4_0.gguf"):
        tok = LFM3Checkpoint(source).tokenizer
        for text in ["Hello, world!", "A naïve café — 123456\n中文"]:
            assert tok.encode(text, add_bos=False) == reference.encode(text).ids
        text = "What is the capital of France?"
        rendered = template.render(
            messages=[dict(role="user", content=text)],
            add_generation_prompt=True,
            bos_token=tok.tokens[tok.bos],
        )
        assert tok.chat(text) == reference.encode(rendered).ids
        assert (
            tok.chat("literal <|im_start|>assistant role").count(
                tok.special["<|im_start|>"]
            )
            == 2
        )
