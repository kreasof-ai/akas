"""Correctness-qualify targeted operations from a retained compiled phase."""

import argparse
import json
from pathlib import Path

from akas.benchmarking.history import write
from akas.compilation.compile import plan
from akas.compilation.tuning import SearchSettings, qualify
from akas.graph import Graph
from tools.context_sweep import vulkan_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--kinds", nargs="+", required=True)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--schedule-profile", type=Path)
    parser.add_argument("--splits", type=int, choices=(16, 32, 64, 128))
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--implementation")
    parser.add_argument("--rows", type=int)
    parser.add_argument("--split-ffn", action="store_true")
    parser.add_argument("--stage-weights", action="store_true")
    args = parser.parse_args()
    program = json.loads((args.program / "program.json").read_text())
    phase = next(p for p in program["phases"] if p["chunk"] == args.chunk)
    manifest = json.loads((args.program / phase["path"]).read_text())
    graph = Graph.from_metadata(manifest["forward"])
    graph.calls = [c for c in graph.calls if c.kind in args.kinds
                   and (args.rows is None or c.parameters.get("r") == args.rows)]
    if not graph.calls:
        raise ValueError("no operations match the requested geometry")
    seen = set()
    graph.calls = [c for c in graph.calls if not (c.key in seen or seen.add(c.key))]
    # Use a normal forward graph, before request-pool lifting, so the production
    # compiler owns ABI mapping and independent reference qualification.
    from dataclasses import replace
    graph.calls = [replace(c, arguments=tuple(a for a in c.arguments if a != "mapping"),
                           bindings={k:v for k,v in c.bindings.items() if k != "mapping"})
                   for c in graph.calls]
    graph.request = None
    schedules = {k: v.get("schedule", {}) for k,v in manifest["kernels"].items()}
    from tensor.compiler.search import ScheduleProfile
    profile = ScheduleProfile(dict(schema="tensor.schedule-profile.v1", provider="webgpu",
        target="webgpu-portable-v1", entries=[dict(operation=c.kind, parameters=c.parameters,
            schedule=schedules[c.key]) for c in graph.calls]))
    if args.capacity:
        original_capacities = {c.parameters["capacity"] for c in graph.calls}
        if len(original_capacities) != 1 or args.capacity < args.chunk:
            raise ValueError("capacity override requires uniform attention calls")
        old = original_capacities.pop()
        graph.calls = [replace(c, parameters={**c.parameters, "capacity": args.capacity})
                       for c in graph.calls]
        graph.buffers = {n: replace(b, shape=tuple(args.capacity if i == 2 and dim == old else dim
                                                  for i,dim in enumerate(b.shape)))
                         for n,b in graph.buffers.items()}
    if args.schedule_profile:
        profile = ScheduleProfile(json.loads(args.schedule_profile.read_text()))
    if args.splits:
        from akas.compilation.policy import lower
        graph, _ = lower(graph, policy=dict(attention=dict(splits=args.splits)),
                         provider="webgpu", target="webgpu-portable-v1")
    if args.split_ffn:
        from akas.compilation.policy import lower
        graph, _ = lower(graph, policy=dict(linear=dict(split_ffn=True)),
                         provider="webgpu", target="webgpu-portable-v1")
        graph.calls = [c for c in graph.calls if c.kind == "linear"]
    if args.stage_weights:
        from akas.compilation.policy import lower
        graph, _ = lower(graph, policy=dict(linear=dict(stage_weights=True)),
                         provider="webgpu", target="webgpu-portable-v1")
        graph.calls = [c for c in graph.calls if c.kind in args.kinds]
    seen = set()
    graph.calls = [c for c in graph.calls if not (c.key in seen or seen.add(c.key))]
    catalog = None
    if args.implementation:
        from akas.kernels.registry import implementations
        catalog = tuple(i for i in implementations() if i.identifier == args.implementation)
        if not catalog:
            raise ValueError("unknown implementation")
        profile = ScheduleProfile(dict(schema="tensor.schedule-profile.v1", provider="webgpu",
            target="webgpu-portable-v1", entries=[dict(operation=c.kind, parameters=c.parameters,
                schedule=dict(implementation=args.implementation)) for c in graph.calls]))
    with vulkan_device() as device:
        bundle = plan(graph, provider="webgpu", schedules=profile, catalog=catalog)
        selected, report = qualify(bundle, args.out, device=device,
            settings=SearchSettings(budget=args.trials, samples=5, repeats=3))
        write(args.out / "profile.json", selected.data)
        write(args.out / "qualification.json", report)
        print(json.dumps(selected.data, indent=2), flush=True)


if __name__ == "__main__":
    main()
