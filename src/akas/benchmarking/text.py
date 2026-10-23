"""Bounded dataset reading and reproducible real-text prompt preparation."""

import hashlib
import json

from akas.benchmarking.workload import prompt_digest


def records(path, *, block_size=65536, max_record_chars=16 * 1024 * 1024):
    """Read JSONL or a JSON array without materializing the whole dataset."""
    if block_size < 1 or max_record_chars < 1:
        raise ValueError("positive parser limits required")
    with open(path, encoding="utf-8") as stream:
        first = stream.read(1)
        while first and first.isspace():
            first = stream.read(1)
        if first != "[":
            line = first + stream.readline(max_record_chars + 1)
            while line:
                if len(line) > max_record_chars:
                    raise ValueError("dataset record exceeds parser limit")
                if line.strip():
                    yield json.loads(line)
                line = stream.readline(max_record_chars + 1)
            return
        decoder, pending, ended, need_value = json.JSONDecoder(), "", False, True
        while True:
            pending = pending.lstrip()
            if pending.startswith("]"):
                if need_value and ended:
                    raise ValueError("trailing comma in dataset array")
                if pending[1:].strip() or any(block.strip() for block in
                    iter(lambda: stream.read(block_size), "")):
                    raise ValueError("data after dataset array")
                return
            if pending and not need_value:
                if not pending.startswith(","):
                    raise ValueError("missing comma in dataset array")
                pending, need_value = pending[1:].lstrip(), True
            if pending and need_value:
                try:
                    value, offset = decoder.raw_decode(pending)
                except json.JSONDecodeError:
                    pass
                else:
                    if not isinstance(value, dict):
                        raise ValueError("dataset array entries must be objects")
                    yield value
                    pending, need_value, ended = pending[offset:], False, True
                    continue
            more = stream.read(block_size)
            if not more:
                raise ValueError("incomplete dataset array")
            pending += more
            if len(pending) > max_record_chars:
                raise ValueError("dataset record exceeds parser limit")


def split(example_id):
    """Assign an example before calibration; crops keep their parent's split."""
    return ("calibration", "heldout")[
        int(hashlib.sha256(str(example_id).encode()).hexdigest()[:8], 16) % 2]


def prompt(tokenizer, text, *, document=None, prefix_tokens=None):
    """Keep the question and chat template intact; crop only document tokens.

    Long documents retain their beginning and end. Token IDs are spliced inside
    the literal user body, preserving the requested length without repeated text
    or synthetic filler. A cropped task is a throughput/acceptance probe, not an
    unmodified dataset accuracy result.
    """
    if document is None:
        if prefix_tokens is not None:
            raise ValueError("exact-length text prompts require a document")
        ids = tokenizer.chat(text)
        preprocessing = dict(format="single-user-chat", cropped=False)
    else:
        if type(prefix_tokens) is not int or prefix_tokens < 1:
            raise ValueError("positive document prompt length required")
        empty = tokenizer.chat("")
        body = tokenizer.encode("user\n", add_bos=False)
        # single_user() puts BOS and the user-role token before the literal body.
        start = (1 if tokenizer.bos is not None else 0) + 1
        left, right = empty[:start] + body, empty[start + len(body):]
        instruction = tokenizer.encode("\n\n" + text, add_bos=False)
        available = prefix_tokens - len(left) - len(right) - len(instruction)
        doc = tokenizer.encode(document, add_bos=False)
        if available < 1 or len(doc) < available:
            raise ValueError("document is too short or question exceeds token budget")
        leading = (available + 1) // 2
        trailing = available - leading
        ids = left + doc[:leading] + (doc[-trailing:] if trailing else []) + instruction + right
        preprocessing = dict(format="single-user-chat", cropped=len(doc) > available,
                             crop="document-token-head-tail", document_tokens=len(doc),
                             retained_document_tokens=available,
                             leading_document_tokens=leading, trailing_document_tokens=trailing)
    return dict(token_ids=ids, prefix_tokens=len(ids), prompt_sha256=prompt_digest([ids]),
                preprocessing=preprocessing)
