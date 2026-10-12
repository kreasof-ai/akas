"""Run a composed LFM decoder with batched text prompts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--prompt", action="append", required=True, help="repeat once per compiled slot"
    )
    parser.add_argument("--provider", choices=("cuda", "webgpu"), default="cuda")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument(
        "--reference", action="store_true", help="load a serial reference bundle"
    )
    parser.add_argument("--max-buffer-size", type=int)
    args = parser.parse_args()
    import tensor
    from akas.models.lfm2.model import LFM2Batched, LFM2Reference

    definition = LFM2Reference if args.reference else LFM2Batched

    with (
        tensor.Device(
            provider=args.provider, max_buffer_size=args.max_buffer_size
        ) as device,
        definition(args.checkpoint, args.bundle, device, graphs=args.graphs) as model,
    ):
        results = model.generate(
            args.prompt, max_tokens=args.max_tokens, chat=not args.raw
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
