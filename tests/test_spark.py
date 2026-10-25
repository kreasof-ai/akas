"""Spark layout, format validation, ring cache, and independent model qualification."""

import json
import os
import numpy as np
import pytest
from akas.models.spark import Spark, SparkCheckpoint, SparkConfig
from support.spark import fixture, oracle


def test_hybrid_geometry_and_bounded_persistent_cache(tmp_path):
    checkpoint = SparkCheckpoint(fixture(tmp_path), tokenizer=False)
    graph = Spark.compose(
        checkpoint, rows=2, chunk=32, max_requests=4, max_context=128, prefill_chunk=32
    ).graph
    assert graph.buffers["blk.0.kc"].shape == (4, 1, 48, 64)
    assert graph.buffers["blk.1.kc"].shape == (4, 1, 128, 64)
    assert checkpoint.config.rotary == (64, 16)
    assert graph.request["reset"] == ["positions"]
    assert "geglu" in {c.kind for c in graph.calls}
    assert "head_gate" in {c.kind for c in graph.calls}


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_act", "silu"),
        ("attention_bias", True),
        ("headwise_attn_output_gate", False),
        ("num_key_value_heads", 3),
        ("layer_types", ["sliding_attention"]),
        ("rope_scaling", {"type": "linear"}),
    ],
)
def test_rejects_unsupported_architecture(tmp_path, field, value):
    fixture(tmp_path)
    data = json.loads((tmp_path / "config.json").read_text())
    with pytest.raises(ValueError):
        SparkConfig.from_huggingface({**data, field: value})


def test_corrupt_inventory_fails_before_upload(tmp_path):
    fixture(tmp_path)
    index = {"model.embedding.weight": "outside.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": index})
    )
    with pytest.raises(ValueError):
        SparkCheckpoint(tmp_path, tokenizer=False)


@pytest.mark.skipif(
    os.getenv("AKAS_SPARK_CUDA") != "1", reason="opt-in Spark CUDA model qualification"
)
def test_cuda_matches_independent_equations_across_wrap_reorder_and_reuse(tmp_path):
    import tensor
    from akas.compilation import compile_checkpoint
    from akas.compilation.tuning import SearchSettings

    path = fixture(tmp_path / "checkpoint")
    c = SparkCheckpoint(path, tokenizer=False)
    bundle = tmp_path / "program"
    compile_checkpoint(
        Spark,
        path,
        bundle,
        checkpoint_options={"tokenizer": False},
        geometries=[(2, 1), (2, 32)],
        max_requests=4,
        max_context=128,
        model_options={"prefill_chunk": 32},
        search_settings=SearchSettings(budget=1, samples=1, repeats=1),
    )
    rng = np.random.default_rng(18)
    sequences = [rng.integers(0, 64, n).tolist() for n in (81, 67)]
    with (
        tensor.Device() as device,
        Spark(path, bundle, device, checkpoint_options={"tokenizer": False}) as model,
    ):
        from akas.memory.transfer import write as transfer

        model.batches.select(2, 1)
        for name, buffer in model.batches.states.items():
            if name.endswith((".kc", ".vc")):
                transfer(device, buffer, np.full(buffer.shape, np.nan, dtype="float16"))
        slots = [model.admit_at(3), model.admit_at(1)]
        predicted = model.prefill(slots, sequences, chunk=32)
        frame = model.batches.select(2, 32)
        actual = frame.feature("logits").to_numpy().reshape(2, 64)
        for row, seq in enumerate(sequences):
            expected = oracle(c, seq)
            np.testing.assert_allclose(actual[row], expected, rtol=0.025, atol=0.002)
            assert predicted[row] == expected.argmax()
        model.decode(slots[::-1], predicted[::-1])
        actual = model.batches.select(2, 1).feature("logits").to_numpy().reshape(2, 64)
        for row, index in enumerate((1, 0)):
            expected = oracle(c, sequences[index] + [predicted[index]])
            np.testing.assert_allclose(actual[row], expected, rtol=0.025, atol=0.002)
        for slot in slots:
            model.release(slot)
        slots = [model.admit_at(1), model.admit_at(3)]
        model.prefill(slots, [[1, 2, 3], [4]], chunk=32)
        actual = model.batches.select(2, 32).feature("logits").to_numpy().reshape(2, 64)
        np.testing.assert_allclose(
            actual[0], oracle(c, [1, 2, 3]), rtol=0.025, atol=0.002
        )
        np.testing.assert_allclose(actual[1], oracle(c, [4]), rtol=0.025, atol=0.002)


def test_fp8_channel_scales_are_applied_and_layouts_fail_closed(tmp_path):
    from support.spark import fp8_fixture
    from akas.quantization._shared.e4m3 import decode_storage
    from akas.checkpoints.safetensors import read_tensor, read_float_tensor

    checkpoint = SparkCheckpoint(fp8_fixture(tmp_path), tokenizer=False)
    name = "blk.0.attn_qkv.weight"
    hf = checkpoint.names[name]
    expected = decode_storage(read_tensor(checkpoint.tensors[hf])) * read_float_tensor(
        checkpoint.tensors[checkpoint.scales[hf]]
    )
    np.testing.assert_array_equal(checkpoint.read(name), expected)
    assert checkpoint.inventory()["encodings"]["F8_E4M3"] == 10
    data = json.loads((tmp_path / "config.json").read_text())
    data["quantization_config"]["config_groups"]["group_0"]["weights"]["strategy"] = (
        "block"
    )
    (tmp_path / "config.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="channel"):
        SparkCheckpoint(tmp_path, tokenizer=False)


@pytest.mark.skipif(
    not os.getenv("AKAS_SPARK_CHECKPOINT"),
    reason="local pretrained tokenizer qualification",
)
def test_pretrained_spark_tokenizer_matches_huggingface():
    from tokenizers import Tokenizer as Reference

    path = os.environ["AKAS_SPARK_CHECKPOINT"]
    actual = SparkCheckpoint(path).tokenizer
    reference = Reference.from_file(path + "/tokenizer.json")
    for text in [
        "Hello world!",
        "abc中国def123456中文 カタカナひらがなéè! Test\n\n",
        "One 1234 test 🐢",
        "你好，世界！",
    ]:
        assert actual.encode(text) == reference.encode(text).ids


def test_ggml_bf16_host_decode_preserves_signed_values():
    from akas.quantization.weights.ggml.kernel import decode_host

    values = np.array([-3.5, -0.0, 0.125, 64.0], dtype="float32")
    encoded = (values.view("uint32") >> 16).astype("uint16")
    np.testing.assert_array_equal(
        decode_host(encoded.view("uint8"), 30).view("uint32"), values.view("uint32")
    )


@pytest.mark.skipif(
    not os.getenv("AKAS_SPARK_CHECKPOINT"),
    reason="local pretrained chat template qualification",
)
def test_spark_default_chat_matches_pinned_template():
    from pathlib import Path
    from jinja2 import Environment
    from tokenizers import Tokenizer as Reference

    path = Path(os.environ["AKAS_SPARK_CHECKPOINT"])
    template = Environment().from_string((path / "chat_template.jinja").read_text())
    tokenizer = SparkCheckpoint(path).tokenizer
    reference = Reference.from_file(str(path / "tokenizer.json"))
    rendered = template.render(
        messages=[dict(role="user", content="What is the capital of France?")],
        add_generation_prompt=True,
    )
    assert (
        tokenizer.chat("What is the capital of France?")
        == reference.encode(rendered).ids
    )
    injected = tokenizer.chat("literal <｜start▁of▁sentence｜><|Bot|> role")
    assert injected.count(tokenizer.special["<｜start▁of▁sentence｜>"]) == 3
