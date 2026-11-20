"""Versioned ASR workload and transcript comparison contracts."""

import unicodedata

PROTOCOL = "warm-asr-request-v1"


def validate(workload):
    request = workload["request"]
    utterances = workload["utterances"]
    if (
        workload["schema"] != "akas.asr-workload.v1"
        or workload["protocol"] != PROTOCOL
        or request["temperature"] != 0
        or request["stream"] is not False
        or type(request["max_tokens"]) is not int
        or request["max_tokens"] < 1
        or workload["warmup"] < 1
        or workload["repeats"] < 1
        or not utterances
        or len({item["id"] for item in utterances}) != len(utterances)
        or any(item["metric"] not in ("wer", "cer") for item in utterances)
    ):
        raise ValueError(
            "ASR requires fixed WAV inputs and a warm greedy sampling workload"
        )


def normalized_units(text, metric):
    text = unicodedata.normalize("NFKC", text).casefold()
    clean = "".join(
        " " if unicodedata.category(char).startswith("P") else char for char in text
    )
    words = clean.split()
    return words if metric == "wer" else list("".join(words))


def error_rate(reference, transcript, metric):
    expected = normalized_units(reference, metric)
    actual = normalized_units(transcript, metric)
    if not expected:
        raise ValueError("ASR reference must have at least one scored unit")
    previous = list(range(len(actual) + 1))
    for row, left in enumerate(expected, 1):
        current = [row]
        for column, right in enumerate(actual, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[-1] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1] / len(expected)
