"""Bounded single-user ChatML formatting, separate from tokenization rules."""

import regex
from akas.text.formats import TokenizerError


def generation_prefix(template, default=None):
    if template is None:
        return default
    if isinstance(template, str):
        match = regex.search(
            r"""\{%-?\s*if add_generation_prompt\s*-?%\}\s*\{\{-?\s*(["'])(<\|im_start\|>assistant(?:\\n|\n)(?:<think>)?)\1\s*-?\}\}\s*\{%-?\s*endif\s*-?%\}\s*\Z""",
            template,
        )
        if match:
            return match[2].replace("\\n", "\n")
    return None


def single_user(tokenizer, text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if (
        tokenizer.generation_prefix is None
        or not {"<|im_start|>", "<|im_end|>"} <= tokenizer.special.keys()
    ):
        raise TokenizerError("unsupported chat generation prefix")
    # Encode user text literally so embedded control strings cannot create roles.
    prefix = ([tokenizer.bos] if tokenizer.bos is not None else []) + [
        tokenizer.special["<|im_start|>"]
    ]
    return (
        prefix
        + tokenizer.encode("user\n" + text, add_bos=False)
        + tokenizer.encode(
            "<|im_end|>\n" + tokenizer.generation_prefix,
            add_bos=False,
            parse_special=True,
        )
    )
