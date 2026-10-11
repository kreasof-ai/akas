"""Inspect checkpoints, generate locally, or serve an Engine over HTTP."""

import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
from importlib import import_module
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="akas", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="inspect model architecture and packed tensor types"
    )
    inspect.add_argument("--model", required=True, type=Path)
    inspect.set_defaults(run=_inspect)
    generate = commands.add_parser(
        "generate", help="generate with a precompiled CUDA or WebGPU bundle"
    )
    generate.add_argument("--model", required=True, type=Path)
    generate.add_argument("--bundle", required=True, type=Path)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-tokens", default=128, type=int)
    generate.add_argument(
        "--context", type=int, help="default: 512 for WebGPU, 8448 for CUDA"
    )
    generate.add_argument(
        "--raw", action="store_true", help="use a plain completion prompt"
    )
    generate.add_argument(
        "--out", type=Path, help="save generated token IDs and text as JSON"
    )
    generate.add_argument("--provider", choices=("cuda", "webgpu"), default="cuda")
    generate.add_argument("--device", type=int, default=0)
    generate.set_defaults(run=_generate)
    serve = commands.add_parser("serve", help="serve an Engine supplied by a factory")
    serve.add_argument(
        "--factory",
        required=True,
        type=_factory,
        help="module:callable yielding Service",
    )
    serve.add_argument(
        "--config", type=_settings, default={}, help="JSON object passed to the factory"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8103)
    serve.set_defaults(run=_serve)
    from akas.compilation import PROFILES

    compile_command = commands.add_parser(
        "compile", help="compile an execution profile for a model definition"
    )
    compile_command.add_argument(
        "--definition",
        type=_factory,
        required=True,
        help="module:class declaring checkpoint_type",
    )
    compile_command.add_argument("--checkpoint", required=True, type=Path)
    compile_command.add_argument("--out", required=True, type=Path)
    compile_command.add_argument("--profile", choices=PROFILES, required=True)
    compile_command.add_argument(
        "--config", type=_settings, default={}, help="JSON execution profile options"
    )
    compile_command.add_argument(
        "--checkpoint-options",
        type=_settings,
        default={},
        help="JSON options for the checkpoint reader",
    )
    compile_command.set_defaults(run=_compile)
    from akas.benchmarking.cli import arguments
    arguments(commands.add_parser("bench", help="reproducible native prefill/decode performance matrix"), _factory)
    args = parser.parse_args()
    args.run(args)


def _compile(args):
    from akas.compilation import compile_checkpoint

    compile_checkpoint(
        args.definition,
        args.checkpoint,
        args.out,
        profile=args.profile,
        checkpoint_options=args.checkpoint_options,
        **args.config,
    )


def _inspect(args):
    from akas.checkpoints.gguf import GGUF
    from akas.models.lfm2.config import Config

    gguf = GGUF(args.model)
    print(
        json.dumps(
            {
                "architecture": asdict(Config.from_gguf(gguf)),
                "tensors": len(gguf.tensors),
                "encodings": dict(Counter(t.encoding for t in gguf.tensors.values())),
            },
            indent=2,
        )
    )


def _generate(args):
    import tensor
    from akas.models.lfm2.model import LFM2

    context = (
        args.context
        if args.context is not None
        else (512 if args.provider == "webgpu" else 8448)
    )
    with (
        tensor.Device(args.device, provider=args.provider) as device,
        LFM2(args.model, args.bundle, device, context=context) as model,
    ):
        result = model.generate(
            args.prompt, max_tokens=args.max_tokens, chat=not args.raw
        )
    print(result["text"])
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def _factory(value):
    try:
        module, name = value.split(":", 1)
        factory = getattr(import_module(module), name)
        if not callable(factory):
            raise ValueError("factory must be callable")
        return factory
    except (ValueError, ImportError, AttributeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _settings(value):
    try:
        settings = json.loads(value)
        if not isinstance(settings, dict):
            raise ValueError("config must be a JSON object")
        return settings
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _serve(args):
    from aiohttp import web
    from akas.serving import create_app

    web.run_app(
        create_app(partial(args.factory, **args.config)),
        host=args.host,
        port=args.port,
        access_log=None,
        handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
