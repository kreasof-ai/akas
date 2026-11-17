"""Pack the already-initialized keep2_plain codec without its unused training generator."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from zonos3.tokenizer import matched_codec


ROOT = Path(__file__).resolve().parent
CODEC = "opus_dual_3n_600k_sched600k_r4096_step600000"
OUT = ROOT / "compact-codec" / "keep2_plain.pt"


def main() -> None:
    started = time.monotonic()
    model = matched_codec.load_model(CODEC, "cpu", decoder="keep2_plain")
    state = model.state_dict()
    config = matched_codec.decoder_spec(CODEC, "keep2_plain")
    config_path = matched_codec.hf_join(config["campaign"], config["config"])
    payload = {
        "format": "zonos3-istft-codec-v1",
        "codec": CODEC,
        "decoder": "keep2_plain",
        "config": json.loads(Path(matched_codec.resolve_file(config_path)).read_text()),
        "step": matched_codec.decoder_info(CODEC, "keep2_plain")["step"],
        "state_dict": state,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "path": str(OUT),
                "bytes": OUT.stat().st_size,
                "sha256": digest,
                "tensor_count": len(state),
                "seconds": time.monotonic() - started,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
