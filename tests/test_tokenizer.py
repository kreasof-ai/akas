"""Exact IDs for serialized BPE rules, added tokens and shared format behavior."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from akas.text.formats import Rules
from akas.text.formats import TokenizerError
from akas.text.tokenizer import Tokenizer

FIXTURE = Path(__file__).parent / "fixtures/tokenizer/qwen35"


@pytest.mark.parametrize(
    "case", json.loads((FIXTURE / "cases.json").read_text())["cases"]
)
def test_qwen_matches_frozen_reference_ids_and_decoding(case):
    tokenizer = Tokenizer.from_pretrained(FIXTURE)
    assert tokenizer.encode(case["text"], parse_special=True) == case["ids"]
    assert tokenizer.encode(case["text"]) == case["literal_ids"]
    assert tokenizer.decode(case["ids"]) == case["decoded"]
    assert tokenizer.decode(case["ids"], skip_special=True) == case["skip_special"]
    assert tokenizer.bos is None and tokenizer.eos == 248046


def test_same_bpe_uses_format_declared_merge_policy():
    merged = Tokenizer(
        Rules(
            vocab={"h": 0, "i": 1, "hi": 2},
            merges=(),
            pattern=r"\w+",
            ignore_merges=True,
        )
    )
    ranked = Tokenizer(replace(merged.rules, ignore_merges=False))
    assert merged.encode("hi") == [2]
    assert ranked.encode("hi") == [0, 1]
    assert merged.decode([2]) == ranked.decode([0, 1]) == "hi"
    assert Tokenizer(replace(ranked.rules, merges=(("h", "i"),))).encode("hi") == [2]


def test_nfc_and_non_special_added_tokens_are_independent_of_special_controls():
    tokenizer = Tokenizer.from_pretrained(FIXTURE)
    assert tokenizer.encode("e\u0301") == tokenizer.encode("é")
    assert tokenizer.encode("<think>") == [tokenizer.ids["<think>"]]
    assert tokenizer.decode([tokenizer.ids["<think>"]], skip_special=True) == "<think>"
    assert tokenizer.encode("<|im_start|>") != [tokenizer.special["<|im_start|>"]]
    assert tokenizer.encode("<|im_start|>", parse_special=True) == [
        tokenizer.special["<|im_start|>"]
    ]


def test_bos_override_missing_bos_and_unmatched_regex_spans():
    tokenizer = Tokenizer(
        Rules(
            vocab={"a": 0, "!": 1, "<bos>": 2},
            merges=(),
            pattern=r"a+",
            special={"<bos>": 2},
            added={"<bos>": 2},
            bos=2,
            add_bos=True,
        )
    )
    assert tokenizer.encode("a!") == [2, 0, 1]
    assert tokenizer.encode("a!", add_bos=False) == [0, 1]
    with pytest.raises(TokenizerError, match="no BOS"):
        Tokenizer.from_pretrained(FIXTURE).encode("a", add_bos=True)


@pytest.mark.parametrize("ids", [[-1], [248320], [True], [1.0]])
def test_decode_rejects_invalid_ids(ids):
    with pytest.raises(ValueError, match="token ID"):
        Tokenizer.from_pretrained(FIXTURE).decode(ids)


@pytest.mark.parametrize(
    "field,value",
    [
        ("normalizer", {"type": "Lowercase"}),
        ("decoder", {"type": "WordPiece"}),
        ("post_processor", {"type": "TemplateProcessing"}),
        ("pre_tokenizer", {"type": "Whitespace"}),
    ],
)
def test_unsupported_serialized_rules_are_rejected(tmp_path, field, value):
    data = json.loads((FIXTURE / "tokenizer.json").read_text())
    data[field] = value
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(data))
    with pytest.raises(TokenizerError):
        Tokenizer.from_file(path)


def test_invalid_added_token_boundaries_are_rejected(tmp_path):
    data = json.loads((FIXTURE / "tokenizer.json").read_text())
    data["added_tokens"][0]["lstrip"] = True
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(data))
    with pytest.raises(TokenizerError, match="boundary"):
        Tokenizer.from_file(path)


def test_empty_special_token_set_and_gpt2_gguf_rules():
    tokenizer = Tokenizer.from_gguf(
        {
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "gpt2",
            "tokenizer.ggml.tokens": ["a", "b", "ab"],
            "tokenizer.ggml.token_type": [1, 1, 1],
            "tokenizer.ggml.merges": ["a b"],
        }
    )
    assert tokenizer.encode("ab") == [2]
    assert tokenizer.encode("ab", parse_special=True) == [2]
    assert tokenizer.decode([2]) == "ab"
