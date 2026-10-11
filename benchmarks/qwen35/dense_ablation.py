"""Matched native AR/MTP matrix with serving, state checks and GPU diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

from benchmarks.llm_serving.runner import atomic_json
from benchmarks.qwen35.dense_scaling import CONCURRENCIES, REVISION, replay


def configurations():
    result = {}
    for fused in (True, False):
        suffix = "fused" if fused else "unfused"
        result[f"ar-{suffix}"] = dict(
            fused_head=fused, window=None, single_graph=False, deferred=False
        )
        for window in (2, 4):
            for mode in ("host", "graph", "deferred"):
                result[f"mtp-w{window}-{mode}-{suffix}"] = dict(
                    fused_head=fused,
                    window=window,
                    single_graph=mode != "host",
                    deferred=mode == "deferred",
                )
    return result


def create_engine(args, config, device):
    from akas.models.qwen35.model import DenseEngine
    from akas.execution.bindings.bf16.speculative import SpeculativeEngine

    target = DenseEngine(
        args.checkpoint, args.bundle, device, fused_head=config["fused_head"]
    )
    engine = target
    if config["window"]:
        engine = SpeculativeEngine(
            target,
            window=config["window"],
            single_graph=config["single_graph"],
            deferred=config["deferred"],
        )
    return target, engine


def qualify(args, config):
    from tensor.providers.cuda.device import Device
    from benchmarks.qwen35.dense_speculation import execute
    from akas.models.qwen35.model import DenseEngine

    references = json.loads(Path(args.references).read_text())
    records = []
    with Device() as device:
        target, engine = create_engine(args, config, device)
        # AR also checks fused versus unfused output and physical states.
        control = (
            target
            if config["window"]
            else DenseEngine(
                args.checkpoint,
                args.bundle,
                device,
                fused_head=not config["fused_head"],
            )
        )
        try:
            for count in CONCURRENCIES:
                specifications = [
                    (
                        references[i % len(references)]["request"]["prompt_token_ids"],
                        24 - i % 7,
                    )
                    for i in range(count)
                ]
                expected, _, _, expected_states = execute(control, specifications)
                actual, _, _, actual_states = execute(engine, specifications)
                row = dict(
                    concurrency=count,
                    tokens=sum(map(len, actual)),
                    greedy_identical=actual == expected,
                    bitwise_terminal_states=actual_states == expected_states,
                    token_sha256=hashlib.sha256(
                        json.dumps(actual).encode()
                    ).hexdigest(),
                )
                records.append(row)
                atomic_json(
                    Path(args.out) / f"{args.profile}-qualification.json",
                    dict(configuration=config, points=records, status="running"),
                )
                if not row["greedy_identical"] or not row["bitwise_terminal_states"]:
                    raise AssertionError(
                        f"qualification failed: {args.profile} C{count}: {row}"
                    )
                print("qualified", args.profile, count, flush=True)
        finally:
            if control is not target:
                control.close()
            engine.close()
    atomic_json(
        Path(args.out) / f"{args.profile}-qualification.json",
        dict(
            configuration=config,
            points=records,
            status="passed",
            independent_model_quality_qualified=False,
        ),
    )


def measured_calls(device, calls, metadata, resources):
    """Fixed-input diagnostic replays, with event nodes around each launch.

    These mutate scratch/request state without advancing host positions. They
    cannot be used as serving throughput or as output/state qualification.
    """
    from tensor.providers.cuda.graph import CudaGraph

    recorder = device.driver.lib.cuEventRecordWithFlags
    recorder.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint]
    recorder.restype = ct.c_int
    handles = []

    def event():
        handle = ct.c_void_p()
        device.driver.call("cuEventCreate", ct.byref(handle), 0)
        handles.append(handle)
        return handle

    rows = [
        (kernel, call, event(), event(), label)
        for (kernel, call), label in zip(calls, metadata)
    ]

    def submit():
        for kernel, call, start, end, _ in rows:
            device.driver.call("cuEventRecordWithFlags", start, device.stream, 1)
            device._launch(kernel, call)
            device.driver.call("cuEventRecordWithFlags", end, device.stream, 1)

    graph = None
    samples = []
    try:
        graph = CudaGraph(device, submit, resources=resources)
        for _ in range(5):
            graph.launch()
            device.synchronize()
            groups, stages = {}, {}
            for _, _, start, end, label in rows:
                elapsed = ct.c_float()
                device.driver.call("cuEventElapsedTime", ct.byref(elapsed), start, end)
                stage, kind = label
                groups[kind] = groups.get(kind, 0.0) + elapsed.value
                stages[stage] = stages.get(stage, 0.0) + elapsed.value
            samples.append(
                dict(groups=groups, stages=stages, total_ms=sum(groups.values()))
            )
        # Keep a real measured sample, rather than independently summing medians.
        return sorted(samples, key=lambda r: r["total_ms"])[len(samples) // 2]
    finally:
        if graph:
            graph.close()
        for handle in handles:
            device.driver.call("cuEventDestroy_v2", handle)


def profile_gpu(args, config):
    from tensor.providers.cuda.device import Device
    from akas.scheduling.scheduler import Request

    references = json.loads(Path(args.references).read_text())
    manifest_rows = {}
    for path in Path(args.bundle).glob("*.json"):
        manifest_rows.update(json.loads(path.read_text())["kernels"])
    result = []
    with Device() as device:
        target, engine = create_engine(args, config, device)
        try:
            for count in CONCURRENCIES:
                requests = [
                    Request(
                        references[i % len(references)]["request"]["prompt_token_ids"],
                        256,
                    )
                    for i in range(count)
                ]
                for request in requests:
                    request.slot = engine.admit()
                # Follow the same chunked prefill path as the serving scheduler.
                for offset in range(0, max(len(r.prompt) for r in requests), 32):
                    active = [r for r in requests if offset < len(r.prompt)]
                    for begin in range(0, len(active), 32):
                        batch_requests = active[begin : begin + 32]
                        sequences = [
                            r.prompt[offset : offset + 32] for r in batch_requests
                        ]
                        if config["window"]:
                            tokens = engine.prefill_batch(batch_requests, sequences, 32)
                        else:
                            tokens = target.batch(len(batch_requests), 32).run(
                                [r.slot for r in batch_requests], sequences
                            )
                        for request, sequence, token in zip(
                            batch_requests, sequences, tokens
                        ):
                            request.offset += len(sequence)
                            if request.offset == len(request.prompt):
                                request.output.append(token)
                # Real rounds provide host-visible latency and draft acceptance.
                # Load/capture decode work before collecting wall-time samples.
                if config["window"]:
                    warm_outputs = engine.decode_many(requests)
                else:
                    warm_outputs = [
                        [t]
                        for t in target.decode(
                            [r.slot for r in requests], [r.output[-1] for r in requests]
                        )
                    ]
                for request, tokens in zip(requests, warm_outputs):
                    request.output.extend(tokens)
                actual_samples = []
                before = dict(getattr(engine, "stats", {}))
                for _ in range(8):
                    device.synchronize()
                    started = time.perf_counter()
                    if config["window"]:
                        outputs = engine.decode_many(requests)
                    else:
                        outputs = [
                            [t]
                            for t in target.decode(
                                [r.slot for r in requests],
                                [r.output[-1] for r in requests],
                            )
                        ]
                    actual_samples.append(
                        dict(
                            seconds=time.perf_counter() - started,
                            emitted=sum(map(len, outputs)),
                        )
                    )
                    for request, tokens in zip(requests, outputs):
                        request.output.extend(tokens)
                delta = {
                    key: value - before[key]
                    for key, value in getattr(engine, "stats", {}).items()
                }
                # A composed diagnostic graph describes GPU work in host mode too;
                # actual_samples above retains its real host/copy overhead.
                if config["window"]:
                    batch = engine.round(count)
                    batch.run(requests)
                    calls, resources = batch.calls, batch.graph.resources
                    maps = [batch.kernels]
                    for part in (batch.verifier, batch.repair, batch.draft):
                        if part:
                            maps.append(part.kernels)
                    if hasattr(target, "deferred"):
                        maps.append(target.deferred.profile(batch.verifier)["kernels"])
                    verifier_start = batch.verifier.calls[0][1]
                else:
                    batch = target.batch(count)
                    batch.run(
                        [r.slot for r in requests], [[r.output[-1]] for r in requests]
                    )
                    calls, resources, maps = (
                        batch.calls,
                        batch.graph.resources,
                        [batch.kernels],
                    )
                    verifier_start = None
                by_object = {
                    id(kernel): manifest_rows[key]
                    for bank in maps
                    for key, kernel in bank.items()
                }
                labels = []
                stage = "control" if config["window"] else "target"
                for kernel, call in calls:
                    info = by_object[id(kernel)]
                    kind, parameters = info["kind"], info["parameters"]
                    if call is verifier_start:
                        stage = "verify"
                    if kind == "spec_draft":
                        stage = "draft"
                    elif kind == "spec_accept":
                        stage = "commit"
                    elif kind == "spec_repair":
                        stage = "repair"
                    elif kind == "spec_result":
                        stage = "control"
                    if kind in ("split_linear", "head_linear"):
                        kind += f"-k{parameters['k']}-o{parameters['o']}"
                    labels.append((stage, kind))
                measured = measured_calls(device, calls, labels, resources)
                row = dict(
                    concurrency=count,
                    diagnostic=True,
                    fixed_input_replay=True,
                    samples=5,
                    real_decode_rounds=actual_samples,
                    speculation=delta,
                    **measured,
                )
                result.append(row)
                atomic_json(
                    Path(args.out) / f"{args.profile}-gpu-profile.json",
                    dict(configuration=config, points=result),
                )
                print("profiled", args.profile, count, measured["total_ms"], flush=True)
                for request in requests:
                    engine.release(request.slot)
        finally:
            engine.close()


async def prewarm(args, config):
    import aiohttp
    from benchmarks.llm_serving.adapters import Adapter
    from benchmarks.llm_serving.runner import request_once
    from benchmarks.qwen35.dense_scaling import server

    evidence = []
    base = server("tensor", args.port, speculative=bool(config["window"]))
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=300)
    ) as session:
        for count in CONCURRENCIES:
            workload = json.loads((Path(args.workloads) / f"c{count}.json").read_text())
            origin = time.perf_counter()
            rows = await asyncio.gather(
                *(
                    request_once(
                        session,
                        Adapter("tensor", "qwen08"),
                        base,
                        dict(request, output_tokens=64),
                        origin=origin,
                        offered=origin,
                        seed=20261014 + i,
                        phase="concurrency-prewarm",
                        timeout=300,
                    )
                    for i, request in enumerate(workload["requests"][:count])
                )
            )
            if any(
                r["status"] != "completed" or r["output_tokens"] != 64 for r in rows
            ):
                raise RuntimeError(f"prewarm failed: {args.profile} C{count}")
            evidence.append(dict(concurrency=count, requests=rows))
    atomic_json(Path(args.out) / f"{args.profile}-prewarm.json", evidence)


def get_info(port):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/server_info", timeout=5
    ) as response:
        return json.load(response)


def sweep(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"PYTHONPATH": ".:src:src"}
    failures = []
    matrix = configurations()
    atomic_json(
        root / "configurations.json",
        dict(model_revision=REVISION, configurations=matrix),
    )
    invariance = root / "batch-invariance.json"
    if not invariance.exists():
        command = [
            sys.executable,
            "-m",
            "benchmarks.qwen35.dense_qualification",
            "--checkpoint",
            args.checkpoint,
            "--bundle",
            args.bundle,
            "--references",
            args.references,
            "--out",
            str(invariance),
        ]
        with (root / "batch-invariance.log").open("wb") as log:
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        print("PASS batch invariance C1-C256", flush=True)
    selected = args.profiles or list(matrix)
    for name in selected:
        config = matrix[name]
        args.profile = name
        print("START", name, flush=True)
        try:
            for phase in ("qualify",):
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.qwen35.dense_ablation",
                    phase,
                    "--checkpoint",
                    args.checkpoint,
                    "--bundle",
                    args.bundle,
                    "--references",
                    args.references,
                    "--out",
                    args.out,
                    "--profile",
                    name,
                ]
                with (root / f"{name}-{phase}.log").open("wb") as log:
                    subprocess.run(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            command = [
                sys.executable,
                "-m",
                "akas",
                "serve",
                "--factory",
                "examples.serving:qwen35_dense",
                "--config",
                json.dumps(
                    dict(
                        checkpoint=str(args.checkpoint),
                        bundle=str(args.bundle),
                        mtp=bool(config["window"]),
                        verification_window=config["window"] or 4,
                        single_graph=config["single_graph"],
                        deferred=config["deferred"],
                        fused_head=config["fused_head"],
                    )
                ),
                "--port",
                str(args.port),
            ]
            atomic_json(
                root / f"{name}-launch.json",
                dict(command=command, configuration=config),
            )
            with (root / f"{name}-server.log").open("wb") as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    for _ in range(300):
                        if process.poll() is not None:
                            raise RuntimeError(f"server exited {process.returncode}")
                        try:
                            with urllib.request.urlopen(
                                f"http://127.0.0.1:{args.port}/health", timeout=3
                            ):
                                break
                        except Exception:
                            time.sleep(2)
                    else:
                        raise RuntimeError("server startup timeout")
                    asyncio.run(prewarm(args, config))
                    # One replay invocation per C retains exact per-point counters.
                    for count in CONCURRENCIES:
                        before = get_info(args.port)
                        asyncio.run(
                            replay(
                                "tensor",
                                args.port,
                                args.workloads,
                                str(root / "results" / name),
                                speculative=bool(config["window"]),
                                concurrencies=[count],
                            )
                        )
                        after = get_info(args.port)
                        atomic_json(
                            root / f"{name}-c{count}-counters.json",
                            dict(before=before, after=after),
                        )
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            command = [
                sys.executable,
                "-m",
                "benchmarks.qwen35.dense_ablation",
                "profile",
                "--checkpoint",
                args.checkpoint,
                "--bundle",
                args.bundle,
                "--references",
                args.references,
                "--out",
                args.out,
                "--profile",
                name,
            ]
            with (root / f"{name}-profile.log").open("wb") as log:
                subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            print("DONE", name, flush=True)
        except Exception as error:
            failures.append(dict(profile=name, error=str(error)))
            atomic_json(root / "failures.json", failures)
            print("FAILED", name, str(error), flush=True)
    if failures:
        raise RuntimeError(
            f"{len(failures)} configurations failed; retained evidence in {root}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("sweep", "qualify", "profile"))
    for name in ("checkpoint", "bundle", "references", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workloads")
    parser.add_argument("--port", type=int, default=8103)
    parser.add_argument("--profile", choices=tuple(configurations()))
    parser.add_argument("--profiles", nargs="+", choices=tuple(configurations()))
    args = parser.parse_args()
    if args.phase == "sweep":
        if not args.workloads:
            parser.error("sweep requires --workloads")
        sweep(args)
    else:
        if not args.profile:
            parser.error("qualification/profile requires --profile")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        config = configurations()[args.profile]
        (qualify if args.phase == "qualify" else profile_gpu)(args, config)


if __name__ == "__main__":
    main()
