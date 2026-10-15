"""Inspect checkpoints, generate locally, or serve an Engine over HTTP."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from akas.benchmarking.cases import matrix
from akas.benchmarking.workload import STANDARD
from akas.benchmarking.runner import run
from akas.benchmarking.report import compare, scaling_regressions, table, write_report
from dataclasses import asdict
from functools import partial
from importlib import import_module


def main():
    parser = argparse.ArgumentParser(prog="akas", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="inspect model architecture and packed tensor types"
    )
    inspect.add_argument("--model", required=True, type=Path)
    inspect.add_argument("--definition", required=True, type=_factory)
    inspect.add_argument("--checkpoint-options", type=_settings, default={})
    inspect.set_defaults(run=_inspect)
    generate = commands.add_parser(
        "generate", help="generate with a precompiled CUDA or WebGPU bundle"
    )
    generate.add_argument("--model", required=True, type=Path)
    generate.add_argument("--definition", required=True, type=_factory)
    generate.add_argument("--config", type=_settings, default={})
    generate.add_argument("--max-buffer-size", type=int)
    generate.add_argument("--bundle", required=True, type=Path)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-tokens", default=128, type=int)
    generate.add_argument("--context", type=int, help="override constructor context")
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
    serving_source = serve.add_mutually_exclusive_group(required=True)
    serving_source.add_argument(
        "--factory",
        type=_factory,
        help="module:callable yielding Service",
    )
    serving_source.add_argument(
        "--deployment", type=Path, help="qualified deployment produced by akas optimize"
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
    compile_command.add_argument("--profile", choices=PROFILES, default="program")
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
    _benchmark_arguments(
        commands.add_parser("bench", help="native prefill/decode performance matrix")
    )
    from akas.benchmarking.history import arguments as history_arguments

    history_arguments(
        commands.add_parser(
            "history", help="find benchmark runs and export qualified schedules"
        )
    )
    from akas.optimization.context import arguments as optimization_arguments

    optimization_arguments(
        commands.add_parser(
            "optimize",
            help="compile, qualify and optimize serving from a model and workload",
        )
    )
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
    checkpoint = args.definition.checkpoint_type(args.model, **args.checkpoint_options)
    config = asdict(checkpoint.config)
    inventory = checkpoint.inventory() if hasattr(checkpoint, "inventory") else None
    print(json.dumps(dict(config=config, inventory=inventory), indent=2))


def _generate(args):
    import tensor

    options = dict(args.config)
    if args.context is not None:
        options["context"] = args.context
    with (
        tensor.Device(
            args.device, provider=args.provider, max_buffer_size=args.max_buffer_size
        ) as device,
        args.definition(args.model, args.bundle, device, **options) as model,
    ):
        result = model.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            chat=not args.raw,
        )
    rows = result if isinstance(result, list) else [result]
    print("\n".join(row["text"] for row in rows))
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

    if getattr(args, "deployment", None) is not None:
        if args.config:
            raise ValueError(
                "a deployment already contains its selected inference settings"
            )
        from akas.optimization.deployment import service

        factory = partial(service, args.deployment)
    else:
        factory = partial(args.factory, **args.config)
    web.run_app(
        create_app(factory),
        host=args.host,
        port=args.port,
        access_log=None,
        handler_cancellation=True,
    )


def _benchmark_arguments(parser):
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("benchmarks"),
        help="durable run archive (default: benchmarks)",
    )
    parser.add_argument(
        "--no-history",
        dest="history",
        action="store_const",
        const=None,
        help="write only the requested report",
    )
    parser.add_argument(
        "--change", help="kernel, algorithm or configuration change being measured"
    )
    parser.add_argument(
        "--change-kind",
        choices=(
            "kernel",
            "algorithm",
            "schedule",
            "configuration",
            "baseline",
            "validation",
        ),
        default="configuration",
    )
    parser.add_argument("--parent-run", help="history run this measurement builds on")
    parser.add_argument(
        "--definition",
        type=_factory,
        required=True,
        help="module:class implementing the session protocol",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-options",
        type=_settings,
        default={},
        help="JSON options for the model checkpoint reader",
    )
    parser.add_argument(
        "--bundles",
        type=Path,
        required=True,
        help="bundle bank, one directory per batch",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="build missing/stale bundles before measurement",
    )
    parser.add_argument("--provider", choices=("cuda", "webgpu"), default="cuda")
    parser.add_argument("--target")
    parser.add_argument(
        "--schedule-profile",
        type=Path,
        help="target-bound Tensor schedule profile used with --compile",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--max-buffer-size",
        type=int,
        help="WebGPU buffer/binding limit to request from the adapter, in bytes",
    )
    parser.add_argument("--kv-dtype")
    parser.add_argument("--batches", nargs="+", type=int, default=STANDARD["batches"])
    parser.add_argument(
        "--prompts", nargs="+", type=int, default=STANDARD["prefix_tokens"]
    )
    parser.add_argument("--depths", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--generated",
        type=int,
        default=STANDARD["output_tokens"],
        help="total output tokens per request, including the first prefill prediction",
    )
    parser.add_argument(
        "--prefill-chunk",
        type=int,
        help="tokens per slot in each compiled prefill pass",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--mixed", action="store_true", help="stagger prompt lengths across slots"
    )
    parser.add_argument(
        "--check",
        type=_factory,
        help="module:callable(session, case) returning correctness evidence",
    )
    parser.add_argument(
        "--graphs", action="store_true", help="capture the token-step graph"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--max-scaling-regression",
        type=float,
        help="stop if a larger batch loses this fraction of aggregate pp/tg throughput",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.1,
        help="fail on throughput loss beyond this fraction against baseline",
    )
    parser.set_defaults(run=_benchmark)


def checkpoint_digest(path):
    path = Path(path)
    if path.is_file():
        from akas.benchmarking.history import digest

        return digest(path)
    files = (
        [path]
        if path.is_file()
        else sorted([*path.glob("*.safetensors"), *path.glob("*.json")])
    )
    if not files:
        raise ValueError("checkpoint has no local weights/configuration")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _benchmark(args):
    from akas.benchmarking.history import Recorder

    recorder, latest, programs = None, None, set()
    if args.history is not None:
        recorder = Recorder(
            args.history,
            dict(
                runtime="akas",
                definition=args.definition.__module__ + ":" + args.definition.__name__,
                checkpoint=str(args.checkpoint.resolve()),
                provider=args.provider,
                target=args.target,
                device=dict(ordinal=args.device),
                protocol="native-token-step-v2",
            ),
            change=args.change,
            parent=args.parent_run,
            kind=args.change_kind,
        )

    def start(metadata):
        if recorder is not None:
            metadata["run_id"] = recorder.id
            recorder.record["metadata"] = dict(metadata)

    def measured(report):
        nonlocal latest
        latest = report
        if recorder is not None:
            recorder.update(report)

    try:
        _benchmark_run(args, start=start, measured=measured, program=programs.add)
    finally:
        if recorder is not None:
            error, error_value, _ = sys.exc_info()
            for directory in sorted(programs):
                if (directory / "program.json").is_file():
                    try:
                        recorder.program(directory)
                    except (OSError, ValueError, KeyError) as archive_error:
                        recorder.record["metadata"].setdefault(
                            "history_errors", []
                        ).append(str(archive_error))
            outcome = (
                "interrupted"
                if error is KeyboardInterrupt
                else "failed"
                if error
                else None
            )
            report = latest or dict(
                schema="akas.benchmark.v1",
                metadata=recorder.record["metadata"],
                results=[
                    dict(case=case.metadata(), status="error", error=str(error_value))
                    for case in matrix(
                        batches=args.batches,
                        prompts=args.prompts,
                        depths=args.depths,
                        generated=args.generated,
                        mixed=args.mixed,
                    )
                ],
            )
            if recorder.record["metadata"].get("history_errors"):
                report["metadata"]["history_errors"] = recorder.record["metadata"][
                    "history_errors"
                ]
                outcome = "failed"
            recorder.update(report, final=True, status=outcome)


def _benchmark_run(args, *, start, measured, program):
    import tensor

    cases = matrix(
        batches=args.batches,
        prompts=args.prompts,
        depths=args.depths,
        generated=args.generated,
        mixed=args.mixed,
    )
    if not 0 <= args.max_regression < 1 or args.repeats < 1 or args.warmup < 0:
        raise ValueError("invalid repeat/warmup/regression settings")
    if args.max_scaling_regression is not None:
        scaling_regressions([], args.max_scaling_regression)
    capacity = max(case.capacity for case in cases)
    profile = getattr(args.definition, "compilation_profile", "reference")
    args.kv_dtype = args.kv_dtype or getattr(args.definition, "model_defaults", {}).get(
        "kv_dtype", "float16"
    )
    chunk = args.prefill_chunk or (
        min(capacity, 128 if args.provider == "cuda" else 32)
        if profile in ("floating", "program")
        else 1
    )
    schedules = None
    if args.schedule_profile:
        if not args.compile:
            raise ValueError("schedule profile requires --compile")
        from tensor.compiler.search import ScheduleProfile

        schedules = ScheduleProfile(json.loads(args.schedule_profile.read_text()))
    definition = args.definition.__module__ + ":" + args.definition.__name__
    device_options = dict(max_buffer_size=args.max_buffer_size)
    with tensor.Device(args.device, provider=args.provider, **device_options) as probe:
        detected = dict(probe.info)
        args.target = args.target or detected["arch"]
    metadata = dict(
        device=detected,
        schedule_profile_sha256=schedules.sha256 if schedules else None,
        correctness="callback" if args.check else "not checked",
        definition=definition,
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=checkpoint_digest(args.checkpoint),
        provider=args.provider,
        target=args.target,
        kv_dtype=args.kv_dtype,
        graphs=args.graphs,
        revision=revision(),
        python=platform.python_version(),
        seed=args.seed,
        timestamp=time.time(),
        protocol="native-token-step-v2",
        input="seeded synthetic token IDs; greedy device sampling",
        timing="wall clock, synchronize before/after; compile/load/reset excluded; prefill includes control uploads",
        ttft="prefill to first greedy token, including logits/head/sampling; no network",
        prefill="chunked" if chunk > 1 else "serial token-step reference",
        prefill_chunk=chunk,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    results = []
    start(metadata)

    def publish_report():
        report = write_report(args.out, metadata, results)
        measured(report)
        return report

    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    program_compiled = False
    for batch in dict.fromkeys(args.batches):
        selected = [case for case in cases if case.batch == batch]
        bundle = args.bundles / (
            f"{args.provider}-{args.kv_dtype}-program"
            if profile == "program"
            else f"{args.provider}-{args.kv_dtype}-b{batch}"
        )
        try:
            program(bundle)
            if args.compile and (profile != "program" or not program_compiled):
                from akas.compilation import compile_checkpoint

                options = (
                    dict(
                        geometries=list(
                            dict.fromkeys(
                                (rows, tokens)
                                for rows in args.batches
                                for tokens in (1, chunk)
                            )
                        ),
                        max_requests=max(args.batches),
                        max_context=capacity,
                        model_options={"kv_dtype": args.kv_dtype},
                        schedules=schedules,
                    )
                    if profile == "program"
                    else dict(
                        batch=batch,
                        capacity=capacity,
                        kv_dtype=args.kv_dtype,
                        schedules=schedules,
                        prefill_chunk=chunk,
                    )
                )
                compile_checkpoint(
                    args.definition,
                    args.checkpoint,
                    bundle,
                    profile=profile,
                    checkpoint_options=args.checkpoint_options
                    if profile == "program"
                    else {"tokenizer": False, **args.checkpoint_options},
                    provider=args.provider,
                    target=args.target,
                    **options,
                )
                program_compiled = profile == "program"
            with tensor.Device(
                args.device, provider=args.provider, **device_options
            ) as device:
                metadata["device"] = device.info
                from akas.benchmarking.session import Session

                constructor_options = dict(graphs=args.graphs)
                if profile == "program":
                    constructor_options["checkpoint_options"] = args.checkpoint_options
                else:
                    constructor_options["tokenizer"] = False
                with args.definition(
                    args.checkpoint, bundle, device, **constructor_options
                ) as model:
                    context = (
                        Session(model, batch=batch, chunk=chunk)
                        if profile == "program"
                        else nullcontext(model)
                    )
                    with context as session:
                        actual_chunk = getattr(session, "prefill_chunk", chunk)
                        metadata["prefill_chunk"] = actual_chunk
                        metadata["prefill"] = (
                            "chunked"
                            if actual_chunk > 1
                            else "serial token-step reference"
                        )
                        if session.kv_dtype != args.kv_dtype:
                            raise ValueError(
                                "bundle KV dtype differs from benchmark request"
                            )
                        for case in selected:
                            print(
                                f"running batch={batch} prompt={case.prompt} depth={case.depth} kv={args.kv_dtype}",
                                flush=True,
                            )
                            try:
                                evidence = (
                                    args.check(session, case) if args.check else None
                                )
                                if args.check and (
                                    not isinstance(evidence, dict)
                                    or evidence.get("correct") is not True
                                ):
                                    raise ValueError("correctness check failed")
                                row = run(
                                    session,
                                    case,
                                    repeats=args.repeats,
                                    warmup=args.warmup,
                                    seed=args.seed,
                                )
                                if evidence is not None:
                                    row["correctness"] = evidence
                            except Exception as error:
                                row = dict(
                                    case=case.metadata(),
                                    status="error",
                                    error=f"{type(error).__name__}: {error}",
                                )
                            results.append(row)
                            publish_report()
                            print(table([row]), flush=True)
                            if args.max_scaling_regression is not None:
                                losses = scaling_regressions(
                                    results, args.max_scaling_regression
                                )
                                if losses:
                                    print(
                                        json.dumps(dict(scaling_regressions=losses)),
                                        flush=True,
                                    )
                                    raise SystemExit(2)
        except Exception as error:
            completed = {json.dumps(row["case"], sort_keys=True) for row in results}
            for case in selected:
                if json.dumps(case.metadata(), sort_keys=True) not in completed:
                    results.append(
                        dict(
                            case=case.metadata(),
                            status="error",
                            error=f"{type(error).__name__}: {error}",
                        )
                    )
            publish_report()
    report = publish_report()
    print(table(results))
    if any(row["status"] != "ok" for row in results):
        raise SystemExit(1)
    if baseline:
        changes = compare(report, baseline)
        if not changes or len(changes) != len(results):
            raise ValueError("baseline must cover every measured case")
        print(json.dumps(dict(comparison=changes), indent=2))
        if any(
            ratio < 1 - args.max_regression
            for row in changes
            for ratio in row["ratios"].values()
        ):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
