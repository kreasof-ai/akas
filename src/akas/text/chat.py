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
        # Some templates append a thinking-mode branch after the assistant role.
        # The single-user API uses that template's explicit non-thinking default.
        match = regex.search(
            r"""(?x)
            \{%-?\s*if\s+add_generation_prompt\s*-?%\}\s*
            \{\{-?\s*(?P<role_quote>["'])(?P<role><\|im_start\|>assistant(?:\\n|\n))(?P=role_quote)\s*-?\}\}\s*
            \{%-?\s*if\s+enable_thinking\s+is\s+defined\s+and\s+enable_thinking\s+is\s+true\s*-?%\}\s*
            \{\{-?\s*(?P<think_quote>["'])<think>(?:\\n|\n)(?P=think_quote)\s*-?\}\}\s*
            \{%-?\s*else\s*-?%\}\s*
            \{\{-?\s*(?P<default_quote>["'])(?P<default><think>(?:\\n|\n){2}</think>(?:\\n|\n){2})(?P=default_quote)\s*-?\}\}\s*
            \{%-?\s*endif\s*-?%\}\s*\{%-?\s*endif\s*-?%\}\s*\Z
            """,
            template,
        )
        if match:
            return (match["role"] + match["default"]).replace("\\n", "\n")
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
