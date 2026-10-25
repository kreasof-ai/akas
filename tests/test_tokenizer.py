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


def test_sequence_decoder_and_bos_template_with_identity_normalized_added_token(
    tmp_path,
):
    data = {
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "<bos>": 2, "python": 3, "Ġ": 4},
            "merges": [],
        },
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "use_regex": True,
        },
        "decoder": {"type": "Sequence", "decoders": [{"type": "ByteLevel"}]},
        "post_processor": {
            "type": "Sequence",
            "processors": [
                {"type": "ByteLevel"},
                {
                    "type": "TemplateProcessing",
                    "single": [
                        {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}},
                    ],
                    "special_tokens": {"<bos>": {"ids": [2], "tokens": ["<bos>"]}},
                },
            ],
        },
        "added_tokens": [
            {"id": 2, "content": "<bos>", "special": True, "normalized": False},
            {"id": 3, "content": "python", "special": False, "normalized": True},
        ],
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(data))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<bos>", "add_bos_token": False})
    )
    tokenizer = Tokenizer.from_pretrained(tmp_path)
    assert tokenizer.encode("python ab") == [2, 3, 4, 0, 1]
    assert tokenizer.encode("python ab", add_bos=False) == [3, 4, 0, 1]
    assert tokenizer.decode([2, 3, 4, 0, 1]) == "<bos>python ab"
    data["post_processor"]["processors"][1]["special_tokens"]["<bos>"]["ids"] = [99]
    (tmp_path / "tokenizer.json").write_text(json.dumps(data))
    with pytest.raises(TokenizerError, match="BOS"):
        Tokenizer.from_pretrained(tmp_path)


@pytest.mark.parametrize("newline", ["\\n", "\n"])
def test_standalone_chat_template_enables_chat_with_config_precedence(
    tmp_path, newline
):
    import shutil

    shutil.copyfile(FIXTURE / "tokenizer.json", tmp_path / "tokenizer.json")
    template = "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
    (tmp_path / "chat_template.jinja").write_text(template.replace("\\n", newline))
    tokenizer = Tokenizer.from_pretrained(tmp_path)
    assert tokenizer.generation_prefix == "<|im_start|>assistant\n"
    assert tokenizer.chat("hello") == tokenizer.encode(
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
        add_bos=False,
        parse_special=True,
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "unsupported"})
    )
    with pytest.raises(TokenizerError, match="generation prefix"):
        Tokenizer.from_pretrained(tmp_path).chat("hello")


def test_new_gguf_profile_needs_no_parser_or_model_branch():
    metadata = {
        "tokenizer.ggml.model": "custom-byte-bpe",
        "tokenizer.ggml.pre": "new-pretokenizer",
        "tokenizer.ggml.tokens": ["a", "b", "ab"],
        "tokenizer.ggml.token_type": [1, 1, 1],
        "tokenizer.ggml.merges": ["a b"],
    }
    profile = dict(model="custom-byte-bpe", pattern=r"\w+")
    with pytest.raises(TokenizerError, match="unsupported"):
        Tokenizer.from_gguf(metadata)
    tokenizer = Tokenizer.from_gguf(metadata, profile=profile)
    assert tokenizer.encode("ab") == [2]
    assert tokenizer.decode([2]) == "ab"


@pytest.mark.parametrize("newline", ["\\n", "\n"])
def test_chat_template_with_thinking_branch_uses_non_thinking_default(
    tmp_path, newline
):
    import shutil

    shutil.copyfile(FIXTURE / "tokenizer.json", tmp_path / "tokenizer.json")
    # Generation suffix from Qwen/Qwen3.5-0.8B's published template.
    template = r"""
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}""".replace("\\n", newline)
    path = tmp_path / "chat_template.jinja"
    path.write_text(template)
    tokenizer = Tokenizer.from_pretrained(tmp_path)
    assert tokenizer.chat("hello") == tokenizer.encode(
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_bos=False,
        parse_special=True,
    )
    # A different default is not silently approximated by this bounded formatter.
    path.write_text(template.replace("</think>", "</other>"))
    with pytest.raises(TokenizerError, match="generation prefix"):
        Tokenizer.from_pretrained(tmp_path).chat("hello")


def test_qwen_gguf_profile_normalizes_text_and_keeps_user_tokens_literal():
    from akas.text.tokenizer import ENCODE_BYTE

    tokens = list(ENCODE_BYTE.values()) + ["<|im_start|>", "<|im_end|>", "<think>"]
    tokenizer = Tokenizer.from_gguf(
        {
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "qwen35",
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.token_type": [1] * 256 + [3, 3, 4],
            "tokenizer.ggml.merges": [],
        }
    )
    assert tokenizer.encode("e\u0301") == tokenizer.encode("é")
    assert tokenizer.encode("<think>") == [258]
    assert "<think>" not in tokenizer.special
    assert tokenizer.decode([258], skip_special=True) == "<think>"
    assert tokenizer.decode([256, 257], skip_special=True) == ""


def test_ordered_isolated_splits_preserve_individual_digit_boundaries():
    from akas.text.tokenizer import ENCODE_BYTE

    vocab = {char: i for i, char in enumerate(ENCODE_BYTE.values())}
    vocab["12"] = len(vocab)
    rules = Rules(
        vocab=vocab,
        merges=(("1", "2"),),
        pattern=r".+",
        split_patterns=(r"\p{N}{1,3}", r"\p{N}"),
    )
    tokenizer = Tokenizer(rules)
    assert tokenizer.encode("a123") == [vocab[c] for c in "a123"]
