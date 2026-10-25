"""Shared byte-level BPE driven by GGUF or tokenizer.json rules.

Both formats use the same split, merge and decode implementation. Format loaders
reject rules this runtime cannot reproduce rather than approximating token IDs.
"""

from __future__ import annotations
from collections.abc import Mapping
import functools
from numbers import Integral
import unicodedata
import regex

from akas.text import chat
from akas.text.formats import Rules
from akas.text.formats import TokenizerError
from akas.text.formats import from_file
from akas.text.formats import from_gguf


def _byte_alphabet():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = values.copy()
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            characters.append(256 + extra)
            extra += 1
    encode = dict(zip(values, map(chr, characters)))
    return encode, {char: byte for byte, char in encode.items()}


ENCODE_BYTE, DECODE_BYTE = _byte_alphabet()


class Tokenizer:
    def __init__(self, source):
        rules = (
            source
            if isinstance(source, Rules)
            else from_gguf(source)
            if isinstance(source, Mapping)
            else from_file(source)
        )
        self.rules = rules
        self.ids = dict(rules.vocab)
        if (
            not self.ids
            or any(type(index) is not int or index < 0 for index in self.ids.values())
            or len(set(self.ids.values())) != len(self.ids)
        ):
            raise TokenizerError("invalid or duplicate vocabulary token ID")
        self.tokens = [None] * (max(self.ids.values()) + 1)
        for token, index in self.ids.items():
            self.tokens[index] = token
        self.ranks = {pair: rank for rank, pair in enumerate(rules.merges)}
        self.added = rules.added
        self.special = rules.special
        self.bos, self.eos = rules.bos, rules.eos
        for index in (self.bos, self.eos):
            if index is not None and (
                type(index) is not int
                or not 0 <= index < len(self.tokens)
                or self.tokens[index] is None
            ):
                raise TokenizerError("invalid BOS/EOS token ID")
        self.pattern = regex.compile(rules.pattern)
        self.split_patterns = tuple(regex.compile(p) for p in rules.split_patterns)
        self.special_pattern = self._added_pattern(self.added)
        self.literal_pattern = self._added_pattern(
            {
                text: index
                for text, index in self.added.items()
                if text not in self.special
            }
        )
        self._piece = functools.lru_cache(maxsize=8192)(self._piece)
        self.generation_prefix = chat.generation_prefix(
            rules.chat_template, rules.generation_prefix
        )

    @classmethod
    def from_gguf(cls, metadata, *, profile=None):
        return cls(from_gguf(metadata, profile=profile))

    @classmethod
    def from_file(cls, path):
        return cls(from_file(path))

    @classmethod
    def from_pretrained(cls, directory):
        """Load local tokenizer files; no downloads or Transformers imports."""
        return cls.from_file(directory)

    @staticmethod
    def _added_pattern(tokens):
        return (
            regex.compile(
                "|".join(
                    regex.escape(t) for t in sorted(tokens, key=lambda text: -len(text))
                )
            )
            if tokens
            else None
        )

    def _piece(self, text):
        encoded = "".join(ENCODE_BYTE[byte] for byte in text.encode("utf-8"))
        if self.rules.ignore_merges and encoded in self.ids:
            return (self.ids[encoded],)
        pieces = list(encoded)
        while len(pieces) > 1:
            best = min(
                range(len(pieces) - 1),
                key=lambda i: self.ranks.get((pieces[i], pieces[i + 1]), float("inf")),
            )
            if (pieces[best], pieces[best + 1]) not in self.ranks:
                break
            pieces[best : best + 2] = [pieces[best] + pieces[best + 1]]
        try:
            return tuple(self.ids[piece] for piece in pieces)
        except KeyError as error:
            raise TokenizerError("missing byte-level vocabulary token") from error

    def _ordinary(self, text):
        if self.rules.normalization:
            text = unicodedata.normalize(self.rules.normalization, text)
        if self.rules.add_prefix_space and text and not text.startswith(" "):
            text = " " + text
        # Each isolated split preserves boundaries made by preceding stages.
        pieces = [text]
        for pattern in self.split_patterns or (self.pattern,):
            divided = []
            for piece in pieces:
                offset = 0
                for match in pattern.finditer(piece):
                    if match.start() > offset:
                        divided.append(piece[offset:match.start()])
                    divided.append(match.group())
                    offset = match.end()
                if offset < len(piece):
                    divided.append(piece[offset:])
            pieces = divided
        for piece in pieces:
            yield from self._piece(piece)

    def encode(self, text, *, add_bos=None, parse_special=False):
        """Return IDs; special control strings stay literal unless requested.

        Non-special added tokens retain their file-declared recognition rules.
        BOS defaults to the source format's setting and can be overridden.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if add_bos is None:
            add_bos = self.rules.add_bos
        if add_bos and self.bos is None:
            raise TokenizerError("tokenizer has no BOS token")
        result = [self.bos] if add_bos else []
        pattern = self.special_pattern if parse_special else self.literal_pattern
        offset = 0
        for match in pattern.finditer(text) if pattern else ():
            result.extend(self._ordinary(text[offset : match.start()]))
            result.append(self.added[match.group()])
            offset = match.end()
        result.extend(self._ordinary(text[offset:]))
        return result

    def decode(self, ids, *, skip_special=False):
        output = bytearray()
        for index in ids:
            if (
                not isinstance(index, Integral)
                or isinstance(index, bool)
                or not 0 <= index < len(self.tokens)
                or self.tokens[index] is None
            ):
                raise ValueError("invalid token ID")
            text = self.tokens[index]
            if text in self.added:
                if not (skip_special and text in self.special):
                    output.extend(text.encode("utf-8"))
            else:
                try:
                    output.extend(DECODE_BYTE[char] for char in text)
                except KeyError as error:
                    raise TokenizerError(
                        "non-byte token is absent from added tokens"
                    ) from error
        return output.decode("utf-8", errors="replace")

    def chat(self, text):
        return chat.single_user(self, text)
