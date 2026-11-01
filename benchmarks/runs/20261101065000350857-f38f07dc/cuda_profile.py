"""Attribute one compiled phase's dependency chain with CUDA events.

Uncaptured intervals include host submission gaps. Captured mode uses CUPTI
activities; both modes diagnose kernels separately from serving measurements.
Counter mode marks a captured launch for an external Nsight Compute collector.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from akas.benchmarking.history import digest, write
from akas.benchmarking.research import definition
from akas.benchmarking.state import RecurrentSnapshot
from akas.benchmarking.workload import synthetic_prompts
from akas.optimization.context import discover


def prepare_phase(model, slots, prompts, args):
    for slot in slots:
        model.release(slot)
        model.admit_at(slot)
    inputs = prompts
    if args.position:
        prefix = synthetic_prompts(model.config.vocab, args.batch, args.position)
        predicted = model.prefill(slots, prefix, chunk=model.prefill_geometry(2048))
        if args.chunk == 1:
            inputs = [[token] for token in predicted]
    frame = model.batch(args.batch, args.chunk)
    frame.prepare(slots, inputs, model.positions)
    return frame


def captured_timing(model, slots, prompts, args):
    """Use CUPTI kernel activities from the actual captured dependency chain."""
    import torch

    frame = prepare_phase(model, slots, prompts, args)
    frame.forward.run()
    model.device.synchronize()
    timings, traces, elapsed = [], [], []
    for sample in range(args.samples):
        frame = prepare_phase(model, slots, prompts, args)
        model.device.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            with torch.profiler.record_function("akas_captured_phase"):
                frame.forward.run()
                model.device.synchronize()
        path = args.out.with_name(args.out.stem + f".sample-{sample}.trace.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(path))
        events = sorted((e for e in json.loads(path.read_text())["traceEvents"]
                         if e.get("cat") == "kernel"), key=lambda e: e["ts"])
        calls = tuple(frame.forward.plan)
        if len(events) != len(calls):
            raise ValueError(f"CUPTI captured {len(events)} kernels; plan contains {len(calls)}")
        for event, entry in zip(events, calls):
            if entry[0].manifest["entrypoint"] not in event["name"]:
                raise ValueError("CUPTI kernel order differs from the captured plan")
        timings.append([event["dur"] / 1e6 for event in events])
        elapsed.append((max(e["ts"] + e["dur"] for e in events) - events[0]["ts"]) / 1e6)
        traces.append(dict(path=str(path), sha256=digest(path)))
    return frame, dict(scope="CUPTI kernel activities from captured execution",
                       gpu_seconds=timings, graph_span_seconds=elapsed,
                       median_gpu_seconds=[statistics.median(row[i] for row in timings)
                                           for i in range(len(timings[0]))],
                       samples=args.samples, traces=traces)


def counter_range(model, slots, prompts, args):
    """Mark one warm captured phase; Nsight owns counter collection and replay."""
    import ctypes

    frame = prepare_phase(model, slots, prompts, args)
    frame.forward.run()
    model.device.synchronize()
    frame = prepare_phase(model, slots, prompts, args)
    model.device.synchronize()
    driver = ctypes.CDLL("libcuda.so.1")
    if driver.cuProfilerStart() != 0:
        raise RuntimeError("CUDA profiler start failed")
    try:
        frame.forward.run()
        model.device.synchronize()
    finally:
        if driver.cuProfilerStop() != 0:
            raise RuntimeError("CUDA profiler stop failed")
    return frame, dict(scope="captured phase marked for external Nsight Compute counters",
                       graph_launches=1)


def observe_outputs(model, slots, prompts, args):
    """Read selected operation outputs in a separate, untimed phase replay."""
    selected = defaultdict(set)
    for selector in args.observe:
        fields = selector.split(":")
        if len(fields) != 2 or any(not name.isidentifier() for name in fields):
            raise ValueError("observations must use operation:argument selectors")
        selected[fields[0]].add(fields[1])
    frame = prepare_phase(model, slots, prompts, args)
    observations = []
    for index, (call, entry) in enumerate(zip(frame.forward.definition.calls, frame.forward.plan)):
        model.device._launch(entry[0], entry[1])
        if call.kind not in selected:
            continue
        names = dict(call.bindings) if call.bindings else {
            spec["name"]: name for spec, name in zip(entry[0].manifest["arguments"], call.arguments)
        }
        if selected[call.kind] - names.keys():
            raise ValueError("observed argument is absent from the operation")
        outputs = {}
        for argument in sorted(selected[call.kind]):
            name = names[argument]
            values = frame.forward.buffers[name].to_numpy()
            outputs[argument] = dict(buffer=name, dtype=str(values.dtype),
                                     shape=list(values.shape), values=values.tolist())
        observations.append(dict(call=index, kind=call.kind, parameters=call.parameters,
                                 outputs=outputs))
    return observations


def profile(args):
    import tensor

    if args.captured:
        import torch
        torch.cuda.init()

    declaration = definition(args.definition or discover(args.checkpoint))
    with tensor.Device() as device, declaration(args.checkpoint, args.program, device,
            graphs=args.captured or args.counter_range, checkpoint_options={"tokenizer": False}) as model:
        cases = json.loads(args.cases.read_text()) if args.cases else [{}]
        if not cases or any(set(case) - {"batch", "chunk", "position"} for case in cases):
            raise ValueError("profile cases must name batch, chunk and/or position")
        results = []
        for index, case in enumerate(cases):
            values = {**vars(args), **case}
            if args.cases:
                values["out"] = args.out.with_name(args.out.stem + f".case-{index}.json")
            options = argparse.Namespace(**values)
            results.append(phase_profile(model, options))
        if args.cases:
            write(args.out, dict(program_sha256=digest(args.program / "program.json"), cases=results))


def phase_profile(model, args):
    from tensor.compiler.tuning import measure_cuda_sequence

    device = model.device
    slots = [model.admit() for _ in range(args.batch)]
    try:
        prompts = synthetic_prompts(model.config.vocab, args.batch, args.chunk)
        if args.position and not (args.captured or args.counter_range):
            prefix = synthetic_prompts(model.config.vocab, args.batch, args.position)
            model.prefill(slots, prefix, chunk=model.prefill_geometry(2048))
        frame = model.batch(args.batch, args.chunk)
        frame.prepare(slots, prompts, model.positions)
        if args.counter_range:
            frame, timing = counter_range(model, slots, prompts, args)
        elif args.captured:
            frame, timing = captured_timing(model, slots, prompts, args)
        else:
            snapshot = RecurrentSnapshot.capture(model)
            entries = tuple(frame.forward.plan)

            def reset():
                snapshot.restore(model)
                frame.prepare(slots, prompts, model.positions)

            timing = measure_cuda_sequence(device,
                [lambda entry=entry: device._launch(entry[0], entry[1]) for entry in entries],
                reset=reset, warmup=1, samples=args.samples)
        groups = defaultdict(lambda: dict(seconds=0.0, count=0))
        for index, call in enumerate(frame.forward.definition.calls):
            row = groups[call.key]
            row.update(kind=call.kind, parameters=call.parameters)
            if "median_gpu_seconds" in timing:
                row["seconds"] += timing["median_gpu_seconds"][index]
            else:
                row.pop("seconds", None)
            row["count"] += 1
        result = dict(program_sha256=digest(args.program / "program.json"),
                      device=device.info, chunk=args.chunk, position=args.position,
                      batch=args.batch, timing=timing,
                      operations=sorted(groups.values(), key=lambda r: r.get("seconds", 0), reverse=True))
        if args.observe:
            result["observations"] = observe_outputs(model, slots, prompts, args)
        write(args.out, result)
        for row in result["operations"]:
            print(row["kind"], row["count"],
                  round(row["seconds"] * 1000, 3) if "seconds" in row else "external counters",
                  row["parameters"], flush=True)
        return result
    finally:
        for slot in slots:
            model.release(slot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--definition")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--cases", type=Path, help="JSON list of phases to profile with one loaded model")
    parser.add_argument("--observe", action="append", default=[], metavar="OPERATION:ARGUMENT",
                        help="retain output values from a separate untimed replay")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--captured", action="store_true", help="attribute captured kernels using PyTorch's CUPTI profiler")
    modes.add_argument("--counter-range", action="store_true", help="mark a warm captured phase for ncu --profile-from-start off --graph-profiling graph")
    parser.add_argument("--out", type=Path, required=True)
    profile(parser.parse_args())
