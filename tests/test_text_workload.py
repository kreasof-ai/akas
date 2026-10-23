"""Real prompts retain their task, split and bounded input handling."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from akas.benchmarking.text import records, prompt, split
from akas.text.tokenizer import Tokenizer


@pytest.mark.parametrize("array", [False, True])
def test_streaming_records_cross_block_boundaries(tmp_path, array):
    rows = [{"text": "hello\n世界", "id": i} for i in range(5)]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(rows) if array else "\n".join(map(json.dumps, rows)))
    assert list(records(path, block_size=7)) == rows


@pytest.mark.parametrize("text", ['[{"a":1}', '[{},]', '[{} {}]', '[{}] {}'])
def test_malformed_array_is_rejected(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        list(records(path, block_size=3))


def test_large_record_fails_with_bounded_parser(tmp_path):
    path = tmp_path / "big.json"
    for text in ('[{"text":"' + "a" * 200 + '"}]', '{"text":"' + "a" * 200 + '"}'):
        path.write_text(text)
        with pytest.raises(ValueError, match="limit"):
            list(records(path, block_size=8, max_record_chars=40))


def test_split_is_stable_and_both_groups_are_used():
    assert split("example-17") == split("example-17")
    assert {split(i) for i in range(20)} == {"calibration", "heldout"}


def test_document_crop_keeps_question_and_chat_boundaries():
    fixture = Tokenizer.from_file(Path(__file__).parent / "fixtures/tokenizer/qwen35")
    tokenizer = Tokenizer(replace(fixture.rules, generation_prefix="<|im_start|>assistant\n"))
    question = "What happens at the end?"
    value = prompt(tokenizer, question, document="Beginning. " + "middle " * 1000 + " The end.", prefix_tokens=128)
    assert len(value["token_ids"]) == 128
    decoded = tokenizer.decode(value["token_ids"])
    assert "Beginning." in decoded and "The end." in decoded and question in decoded
    assert decoded.endswith("<|im_end|>\n<|im_start|>assistant\n")
    assert value["preprocessing"]["cropped"]
    with pytest.raises(ValueError, match="short"):
        prompt(tokenizer, question, document="short", prefix_tokens=128)
    with pytest.raises(ValueError):
        prompt(tokenizer, question, prefix_tokens=128)
