"""Read tokenizer rules from GGUF metadata or a local tokenizer.json."""

from dataclasses import dataclass, field
import json
from pathlib import Path


class TokenizerError(ValueError):
    pass


def profiles():
    """Serialized format presets; no model execution policy lives here."""
    return json.loads(Path(__file__).with_name("profiles.json").read_text())


@dataclass(frozen=True)
class Rules:
    vocab: dict[str, int]
    merges: tuple[tuple[str, str], ...]
    pattern: str
    ignore_merges: bool = False
    normalization: str | None = None
    add_prefix_space: bool = False
    added: dict[str, int] = field(default_factory=dict)
    special: dict[str, int] = field(default_factory=dict)
    bos: int | None = None
    eos: int | None = None
    add_bos: bool = False
    chat_template: str | None = None
    generation_prefix: str | None = None
    split_patterns: tuple[str, ...] = ()


def _merges(values):
    result = []
    for value in values:
        pair = value.split(" ") if isinstance(value, str) else value
        if len(pair) != 2 or any(not isinstance(part, str) for part in pair):
            raise TokenizerError("invalid BPE merge pair")
        result.append(tuple(pair))
    return tuple(result)


def from_gguf(metadata, *, profile=None):
    pre = metadata.get("tokenizer.ggml.pre")
    profile = profiles()["gguf"].get(pre) if profile is None else profile
    if profile is None or metadata.get("tokenizer.ggml.model") != profile["model"]:
        raise TokenizerError(f"unsupported GGUF byte-level BPE profile: {pre!r}")
    pattern, ignore = profile["pattern"], profile.get("ignore_merges", False)
    tokens = metadata["tokenizer.ggml.tokens"]
    types = metadata["tokenizer.ggml.token_type"]
    if len(set(tokens)) != len(tokens) or len(types) != len(tokens):
        raise TokenizerError("duplicate vocabulary token or invalid token types")
    special = {token: i for i, token in enumerate(tokens) if types[i] == 3}
    added = {token: i for i, token in enumerate(tokens) if types[i] in (3, 4)}
    return Rules(
        vocab={token: i for i, token in enumerate(tokens)},
        merges=_merges(metadata["tokenizer.ggml.merges"]),
        pattern=pattern,
        split_patterns=tuple(profile.get('split_patterns', ())),
        ignore_merges=ignore,
        normalization=profile.get("normalization"),
        added=added,
        special=special,
        bos=metadata.get("tokenizer.ggml.bos_token_id"),
        eos=metadata.get("tokenizer.ggml.eos_token_id"),
        add_bos=metadata.get(
            "tokenizer.ggml.add_bos_token", profile.get("add_bos", False)
        ),
        chat_template=metadata.get("tokenizer.chat_template"),
        generation_prefix=(
            profile.get("generation_prefix")
            if "tokenizer.chat_template" not in metadata
            else None
        ),
    )


def from_file(path):
    path = Path(path)
    if path.is_dir():
        path /= "tokenizer.json"
    data = json.loads(path.read_text())
    config_path = path.with_name("tokenizer_config.json")
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    template_path = path.with_name("chat_template.jinja")
    chat_template = config.get("chat_template")
    if chat_template is None and template_path.is_file():
        chat_template = template_path.read_text()
    model = data.get("model", {})
    if model.get("type") != "BPE" or any(
        model.get(key)
        for key in (
            "dropout",
            "continuing_subword_prefix",
            "end_of_word_suffix",
            "byte_fallback",
        )
    ):
        raise TokenizerError(
            "requires deterministic byte-level BPE without fallback or affixes"
        )
    normalizer = data.get("normalizer")
    if normalizer == {'type': 'Sequence', 'normalizers': []}:
        normalizer = None
    normalization = normalizer.get("type") if isinstance(normalizer, dict) else None
    if normalizer is not None and normalization not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise TokenizerError("unsupported tokenizer normalizer")
    pre = data.get("pre_tokenizer", {})
    stages = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else [pre]
    split_patterns = ()
    if (
        len(stages) == 1
        and stages[0].get("type") == "ByteLevel"
        and stages[0].get("use_regex", True)
    ):
        byte = stages[0]
        pattern = profiles()["byte_level"]["pattern"]
    elif (
        len(stages) >= 2
        and all(s.get('type') in ('Split', 'Digits') for s in stages[:-1])
        and stages[-1].get("type") == "ByteLevel"
    ):
        byte = stages[-1]
        patterns = []
        for split in stages[:-1]:
            if split['type'] == 'Digits' and split.get('individual_digits') is True:
                patterns.append(r'\p{N}')
            elif (split['type'] == 'Split' and split.get('behavior') == 'Isolated'
                  and not split.get('invert') and isinstance(split.get('pattern', {}).get('Regex'), str)):
                patterns.append(split['pattern']['Regex'])
            else:
                raise TokenizerError('unsupported byte-level split rules')
        pattern = patterns[0]
        split_patterns = tuple(patterns)
        if (byte.get("use_regex", True)
            or byte.get("add_prefix_space", False)
        ):
            raise TokenizerError("unsupported byte-level split rules")
    else:
        raise TokenizerError("unsupported byte-level pre-tokenizer")
    if model.get('unk_token') is not None:
        from akas.text.tokenizer import ENCODE_BYTE
        utf8_bytes = {char for byte, char in ENCODE_BYTE.items()
                      if byte < 245 and byte not in (192, 193)}
        if not utf8_bytes <= model['vocab'].keys():
            raise TokenizerError('unknown-token fallback requires a complete byte alphabet')
    decoder = data.get("decoder") or {}
    if decoder.get("type") == "Sequence":
        decoders = decoder.get("decoders", [])
        if len(decoders) != 1:
            raise TokenizerError("requires one byte-level decoder")
        decoder = decoders[0]
    if decoder.get("type") != "ByteLevel":
        raise TokenizerError("requires a byte-level decoder")
    post = data.get("post_processor")
    processors = (
        post.get("processors", [])
        if post and post.get("type") == "Sequence"
        else [post]
        if post
        else []
    )
    template_bos = None
    template_id = None
    templates = 0
    for processor in processors:
        if processor.get("type") == "ByteLevel":
            continue  # offset-only transformation; token IDs are unchanged
        if processor.get("type") != "TemplateProcessing":
            raise TokenizerError("unsupported tokenizer post-processor")
        templates += 1
        single = processor.get("single", [])
        sequence = {"Sequence": {"id": "A", "type_id": 0}}
        if single == [sequence]:
            continue
        if len(single) != 2 or single[1] != sequence or "SpecialToken" not in single[0]:
            raise TokenizerError("unsupported single-sequence token template")
        prefix = single[0]["SpecialToken"]
        template_bos = prefix.get("id")
        entry = processor.get("special_tokens", {}).get(template_bos, {})
        if (
            prefix.get("type_id") != 0
            or entry.get("tokens") != [template_bos]
            or len(entry.get("ids", [])) != 1
        ):
            raise TokenizerError("unsupported BOS token template")
        template_id = entry["ids"][0]
    if templates > 1:
        raise TokenizerError("multiple tokenizer templates are unsupported")
    added, special = {}, {}
    vocab = dict(model["vocab"])
    for token in data.get("added_tokens", []):
        if any(token.get(key) for key in ("single_word", "lstrip", "rstrip")) or (
            token.get("normalized") and normalization is not None
        ):
            raise TokenizerError(
                "unsupported added-token boundary or normalization rules"
            )
        text, index = token["content"], token["id"]
        if text in vocab and vocab[text] != index:
            raise TokenizerError("added-token ID conflicts with vocabulary")
        vocab[text] = index
        added[text] = index
        if token.get("special"):
            special[text] = index

    def token_id(name):
        value = config.get(name)
        text = value.get("content") if isinstance(value, dict) else value
        if text is not None and text not in vocab:
            raise TokenizerError(f"{name} is absent from vocabulary")
        return vocab.get(text)

    bos = token_id("bos_token")
    if template_bos is not None and (
        vocab.get(template_bos) != bos or template_id != bos
    ):
        raise TokenizerError("BOS template disagrees with tokenizer configuration")
    return Rules(
        vocab=vocab,
        merges=_merges(model["merges"]),
        pattern=pattern,
        split_patterns=split_patterns,
        ignore_merges=model.get("ignore_merges", False),
        normalization=normalization,
        add_prefix_space=byte.get("add_prefix_space", False),
        added=added,
        special=special,
        bos=bos,
        eos=token_id("eos_token"),
        add_bos=template_bos is not None or config.get("add_bos_token", False),
        chat_template=chat_template,
    )
