"""Translate optimize CLI arguments into optimization intent."""

import json
from pathlib import Path

from akas.optimization.context import Context, duration


def arguments(parser):
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="local safetensors directory or GGUF checkpoint",
    )
    parser.add_argument(
        "--workload",
        default="standard-v1",
        help="packaged workload ID or a workload JSON file",
    )
    parser.add_argument(
        "--budget",
        required=True,
        help="additional wall-clock allowance, e.g. 30m or 4h; includes compilation",
    )
    parser.add_argument("--backend", choices=("auto", "cuda", "webgpu"), default="auto")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("auto", "ar", "mtp"),
        default="auto",
        help="required generation mode, or evaluate supported modes",
    )
    parser.add_argument(
        "--kv-cache",
        choices=("float16", "bfloat16", "fp8"),
        help="required cache precision; defaults to the model declaration",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--definition", help="optional module:class for a custom model")
    parser.add_argument("--memory-gib", type=float, help="optional GPU memory ceiling")
    parser.add_argument("--ttft-ms", type=float, help="optional maximum serving TTFT")
    parser.add_argument(
        "--tpot-ms",
        type=float,
        help="optional maximum serving time per subsequent output token",
    )
    parser.add_argument(
        "--shape-timeout",
        help="maximum duration per measurement repetition, including diagnostics, e.g. 2m",
    )
    parser.add_argument("--history", type=Path, default=Path("benchmarks"))
    parser.set_defaults(run=command)


def command(args):
    from akas.optimization.supervisor import optimize

    if args.shape_timeout is not None:
        import os

        os.environ["AKAS_SHAPE_TIMEOUT"] = str(duration(args.shape_timeout))
    context = Context.from_options(args)
    result = optimize(
        context, args.out, budget=duration(args.budget), history=args.history
    )
    print(json.dumps(result, indent=2))
    if not result.get("deployment"):
        raise SystemExit(1)
