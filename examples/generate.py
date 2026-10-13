"""Use the shared engine to schedule one length-bounded LFM2 request."""

import argparse
import tensor
from akas import Engine
from akas.models.lfm2.model import LFM2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    with tensor.Device() as device:
        with Engine.from_checkpoint(
            LFM2, args.model, args.bundle, device
        ) as engine:
            tokenizer = engine.model.checkpoint.tokenizer
            request = engine.submit(
                tokenizer.encode(args.prompt), max_tokens=args.max_tokens
            )
            while engine.step():
                engine.events.clear()
            print(tokenizer.decode(request.output))


if __name__ == "__main__":
    main()
