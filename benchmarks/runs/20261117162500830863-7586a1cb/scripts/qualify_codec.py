"""Compare the compact inference codec against the original assets on identical code frames."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from zonos3.tokenizer import matched_codec


ROOT = Path(__file__).resolve().parent
CODEC = "opus_dual_3n_600k_sched600k_r4096_step600000"
ORIGINAL = ROOT / "cache2/zonos3-mini/autoencoder/generator.pt"
COMPACT = ROOT / "compact-codec/keep2_plain.pt"
OUT = ROOT / "results2/codec-compact-qualification.json"


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes()).hexdigest()


def main() -> None:
    report = {"codec": CODEC, "decoder": "keep2_plain", "checks": []}
    original_path = os.environ.pop("ZONOS3_COMPACT_CODEC", None)
    if original_path:
        raise RuntimeError("clear ZONOS3_COMPACT_CODEC for qualification")
    original = matched_codec.load_model(CODEC, "cpu", decoder="keep2_plain")
    compact_path = os.environ["ZONOS3_COMPACT_CODEC"] = str(COMPACT)
    compact = matched_codec.load_model(CODEC, "cpu", decoder="keep2_plain")
    original_state, compact_state = original.state_dict(), compact.state_dict()
    assert original_state.keys() == compact_state.keys()
    for name in original_state:
        assert torch.equal(original_state[name], compact_state[name]), name
    del original_state, compact_state
    original.to("cuda").eval()
    compact.to("cuda").eval()
    for frames in (1, 7, 32):
        generator = torch.Generator().manual_seed(1000 + frames)
        codes = torch.randint(0, 4096, (1, frames, 10), generator=generator, dtype=torch.int64)
        reference_audio = matched_codec.decode_model(original, codes, matched_codec.entry(CODEC))
        compact_audio = matched_codec.decode_model(compact, codes, matched_codec.entry(CODEC))
        reference = reference_audio.numpy()
        actual = compact_audio.numpy()
        bitwise = bool(np.array_equal(reference, actual))
        rms = float(np.sqrt(np.mean((reference.astype(np.float64) - actual.astype(np.float64)) ** 2)))
        report["checks"].append(
            {
                "frames": frames,
                "bitwise": bitwise,
                "rms": rms,
                "reference_sha256": digest(reference),
                "compact_sha256": digest(actual),
            }
        )
        assert bitwise and rms == 0.0
    report["original_checkpoint_bytes"] = ORIGINAL.stat().st_size
    report["compact_checkpoint_bytes"] = COMPACT.stat().st_size
    report["compact_checkpoint_sha256"] = hashlib.sha256(COMPACT.read_bytes()).hexdigest()
    report["all_bitwise"] = all(check["bitwise"] for check in report["checks"])
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
