"""Prepare a pinned conversation, code, math and long-document token workload."""

import argparse
from pathlib import Path
import urllib.request

from akas.benchmarking.history import digest, write
from akas.benchmarking.text import prompt, records, split
from akas.models.lfm2.model import LFM2MoE


SOURCES = {
    "mtbench": dict(
        revision="587d5cfa1609a43d192cedb8441cac3c17db105d",
        url="https://raw.githubusercontent.com/lm-sys/FastChat/587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge/data/mt_bench/question.jsonl",
        sha256="119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7",
        filename="mtbench.jsonl"),
    "gsm8k-train": dict(
        revision="3101c7d5072418e28b9008a6636bde82a006892c",
        url="https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl",
        sha256="17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
        filename="gsm8k-train.jsonl"),
    "gsm8k-test": dict(
        revision="3101c7d5072418e28b9008a6636bde82a006892c",
        url="https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
        sha256="3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        filename="gsm8k-test.jsonl"),
    "longbench-v2": dict(
        revision="2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
        url="https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json",
        sha256="15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2",
        filename="longbench-v2.json"),
}


def fetch(directory, source):
    path = directory / source["filename"]
    if not path.exists():
        temporary = path.with_suffix(".part")
        with urllib.request.urlopen(source["url"], timeout=120) as response, temporary.open("wb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
        if digest(temporary) != source["sha256"]:
            temporary.unlink()
            raise ValueError("dataset download does not match pinned checksum")
        temporary.replace(path)
    if digest(path) != source["sha256"]:
        raise ValueError("cached dataset does not match pinned checksum")
    return path


def prepare(tokenizer, directory, *, per_category=1, long_prefixes=(8192, 130944)):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: fetch(directory, source) for name, source in SOURCES.items()}
    cases, counts = [], {}

    def add(source, example_id, group, category, value):
        cases.append(dict(id=f"{source}:{example_id}:{value['prefix_tokens']}",
                          source=source, example_id=str(example_id), split=group,
                          category=category, **value))

    for row in records(paths["mtbench"]):
        category, example_id = row["category"], row["question_id"]
        if category not in ("writing", "coding", "reasoning"):
            continue
        group = split(f"mtbench:{example_id}")
        key = (group, category)
        if counts.get(key, 0) < per_category:
            add("mtbench", example_id, group, category, prompt(tokenizer, row["turns"][0]))
            counts[key] = counts.get(key, 0) + 1
    for source, group in (("gsm8k-train", "calibration"), ("gsm8k-test", "heldout")):
        for index, row in enumerate(records(paths[source])):
            if index >= 2 * per_category:
                break
            add(source, index, group, "math", prompt(tokenizer, row["question"]))
    wanted = {(group, size) for group in ("calibration", "heldout") for size in long_prefixes}
    for row in records(paths["longbench-v2"]):
        group = split(row["_id"])
        # Pick one real document per split, long enough for every requested crop.
        if not any(key[0] == group for key in wanted) or len(row["context"]) < 4 * max(long_prefixes):
            continue
        question = row["question"] + "\n" + "\n".join(
            f"{letter}. {row['choice_' + letter]}" for letter in "ABCD")
        question += "\nExplain your reasoning, then give the best answer."
        values = []
        try:
            for size in long_prefixes:
                values.append(prompt(tokenizer, question, document=row["context"], prefix_tokens=size))
        except ValueError:
            continue
        for value in values:
            add("longbench-v2", row["_id"], group, "long-document", value)
            wanted.discard((group, value["prefix_tokens"]))
        if not wanted:
            break
    if wanted or any(counts.get((group, category), 0) != per_category
                     for group in ("calibration", "heldout")
                     for category in ("writing", "coding", "reasoning")):
        raise ValueError("pinned datasets cannot fill requested workload")
    return dict(schema="akas.text-workload.v1", sources=SOURCES, cases=cases,
                output_tokens=128, ignore_eos=True,
                scope="Runtime acceptance and throughput; no training or dataset accuracy claim. "
                      "First chat turn only; original GSM8K train/test split; deterministic "
                      "example split for MT-Bench and LongBench; document-only token crops.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = LFM2MoE.checkpoint_type(args.checkpoint)
    workload = prepare(checkpoint.tokenizer, args.cache)
    workload["tokenizer_checkpoint_sha256"] = digest(args.checkpoint)
    write(args.output, workload)
    print(f"Prepared {len(workload['cases'])} prompts: {args.output}")


if __name__ == "__main__":
    main()
