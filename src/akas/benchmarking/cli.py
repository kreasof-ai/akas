"""Native benchmark command: compile once per batch, then reuse across lengths."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from akas.benchmarking.cases import matrix
from akas.benchmarking.runner import run
from akas.benchmarking.report import compare, table, write_report


def arguments(parser, factory):
    parser.add_argument(
        "--definition",
        type=factory,
        required=True,
        help="module:class implementing the session protocol",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--kv-dtype", default="float16")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--prompts", nargs="+", type=int, default=[32, 2048])
    parser.add_argument("--depths", nargs="+", type=int, default=[0])
    parser.add_argument("--generated", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--mixed", action="store_true", help="stagger prompt lengths across slots"
    )
    parser.add_argument(
        "--check",
        type=factory,
        help="module:callable(session, case) returning correctness evidence",
    )
    parser.add_argument(
        "--graphs", action="store_true", help="capture the token-step graph"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.1,
        help="fail on throughput loss beyond this fraction against baseline",
    )
    parser.set_defaults(run=main)


def checkpoint_digest(path):
    path = Path(path)
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


def main(args):
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
    capacity = max(case.capacity for case in cases)
    schedules = None
    if args.schedule_profile:
        if not args.compile:
            raise ValueError("schedule profile requires --compile")
        from tensor.compiler.search import ScheduleProfile

        schedules = ScheduleProfile(json.loads(args.schedule_profile.read_text()))
    definition = args.definition.__module__ + ":" + args.definition.__name__
    with tensor.Device(args.device, provider=args.provider) as probe:
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
        protocol="native-token-step-v1",
        input="seeded synthetic token IDs; greedy device sampling",
        timing="wall clock, synchronize before/after; compile/load/reset excluded; prefill includes control uploads",
        ttft="prefill to first greedy token, including logits/head/sampling; no network",
        prefill="serial token-step reference",
        repeats=args.repeats,
        warmup=args.warmup,
    )
    results = []
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    for batch in dict.fromkeys(args.batches):
        selected = [case for case in cases if case.batch == batch]
        bundle = args.bundles / f"{args.provider}-{args.kv_dtype}-b{batch}"
        try:
            if args.compile:
                from akas.compilation import compile_checkpoint

                compile_checkpoint(
                    args.definition,
                    args.checkpoint,
                    bundle,
                    profile="reference",
                    checkpoint_options={"tokenizer": False},
                    provider=args.provider,
                    target=args.target,
                    batch=batch,
                    capacity=capacity,
                    kv_dtype=args.kv_dtype,
                    schedules=schedules,
                )
            with tensor.Device(args.device, provider=args.provider) as device:
                metadata["device"] = device.info
                with args.definition(
                    args.checkpoint, bundle, device, graphs=args.graphs, tokenizer=False
                ) as session:
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
                            evidence = args.check(session, case) if args.check else None
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
                        write_report(args.out, metadata, results)
                        print(table([row]), flush=True)
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
            write_report(args.out, metadata, results)
    report = write_report(args.out, metadata, results)
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
