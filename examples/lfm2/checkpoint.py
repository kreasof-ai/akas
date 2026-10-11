"""Export the diagnostic GGUF's same values to a local safetensors fixture.

The generated checkpoint is synthetic. It demonstrates loading parity without
claiming pretrained quality or downloading weights as part of normal tests.
"""

import argparse
import json
from pathlib import Path
import struct
from akas.models.lfm2.weights import ReferenceCheckpoint, huggingface_names


def export(gguf, directory, *, dtype="F32"):
    if dtype not in ("F32", "F16", "BF16"):
        raise ValueError("fixture dtype must be F32/F16/BF16")
    checkpoint = ReferenceCheckpoint(gguf, tokenizer=False)
    c = checkpoint.config
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(
        model_type="lfm2",
        hidden_size=c.width,
        intermediate_size=c.ff,
        num_attention_heads=c.heads,
        num_key_value_heads=c.kv_heads,
        num_hidden_layers=len(c.layers),
        full_attn_idxs=[i for i, k in enumerate(c.layers) if k == "attention"],
        vocab_size=c.vocab,
        conv_L_cache=c.conv,
        norm_eps=c.epsilon,
        rope_theta=c.theta,
        max_position_embeddings=c.max_context,
    )
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    header, payload = {}, bytearray()
    for name, hf in huggingface_names(
        c, output="output.weight" in checkpoint.names
    ).items():
        values = checkpoint.read(name)
        if name.endswith("conv.weight"):
            values = values[:, None, :]
        if dtype == "BF16":
            bits = values.view("uint32")
            values = ((bits + 32767 + ((bits >> 16) & 1)) >> 16).astype("uint16")
        else:
            values = values.astype("float32" if dtype == "F32" else "float16")
        start = len(payload)
        payload.extend(values.tobytes())
        header[hf] = dict(
            dtype=dtype, shape=list(values.shape), data_offsets=[start, len(payload)]
        )
    encoded = json.dumps(header).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("F32", "F16", "BF16"), default="F32")
    args = parser.parse_args()
    export(args.gguf, args.out, dtype=args.dtype)
