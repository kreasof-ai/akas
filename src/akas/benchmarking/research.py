"""Reusable research cases over the common compiler, executor and benchmark harness.

A case names a model declaration, geometry and operation policy. It does not own
numerical execution. Prepared records point to one compiled program; older
separate decode/prefill/repair bundles must be rebuilt through this interface.
"""

import hashlib
from importlib import import_module
import json
from pathlib import Path
import time


def definition(name):
    module, separator, symbol = name.partition(":")
    if not separator or not module or not symbol:
        raise ValueError("definition must be module:class")
    return getattr(import_module(module), symbol)


def geometries(batches, chunk, windows=()):
    result = []
    for rows in batches:
        result.extend(((rows, 1), (rows, chunk)))
        if windows:
            result.extend(((rows, 1, "proposal"), (rows, chunk, "proposal")))
            for window in windows:
                result.extend(((rows, window, "verify"), (rows, window, "proposal")))
    return list(dict.fromkeys(result))


def prepare(model_definition, checkpoint, out, *, batches=(1, 4, 16, 64, 256), chunk=128,
            windows=(), context=4096, provider="cuda", target=None, policy=None,
            model_options=None, checkpoint_options=None, device=None, deferred=False):
    from akas.compilation.compile import compile_program
    from akas.artifacts.fingerprint import backbone_hashes

    model = definition(model_definition)
    loaded = model.checkpoint_type(checkpoint, **(checkpoint_options or {}))
    settings = dict(definition=model_definition, checkpoint=str(Path(checkpoint).resolve()),
                    batches=list(batches), chunk=chunk, windows=list(windows), context=context,
                    provider=provider, target=target, policy=policy or {},
                    model_options=model_options or {}, checkpoint_options=checkpoint_options or {}, deferred=deferred)
    sources = backbone_hashes()
    source_identity = hashlib.sha256(json.dumps(dict(settings=settings, sources=sources), sort_keys=True).encode()).hexdigest()
    program = Path(out).resolve()
    compile_program(loaded, program, definition=model.compose,
                    geometries=geometries(batches, chunk, windows), max_requests=max(batches),
                    max_context=context, provider=provider, target=target,
                    policy=policy, model_options=model_options, device=device, round_windows=windows, deferred=deferred)
    prepared = dict(schema="akas.research.v1", **settings, program=str(program),
                    source_identity=source_identity, source_hashes=sources,
                    program_sha256=hashlib.sha256((program / "program.json").read_bytes()).hexdigest())
    (program / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    return prepared


def validate(prepared):
    if prepared.get("schema") != "akas.research.v1":
        raise ValueError("legacy research bundles require recompilation into a program")
    path = Path(prepared["program"]) / "program.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != prepared["program_sha256"]:
        raise ValueError("prepared program manifest changed")
    return prepared


def run_case(prepared, *, batch=1, prompt=32, generated=128, depth=0, mixed=False,
             repeats=3, warmup=1, out=None):
    import tensor
    from akas.benchmarking.cases import Case
    from akas.benchmarking.runner import run
    from akas.benchmarking.session import Session

    validate(prepared)
    case = Case(batch, prompt, generated, depth, mixed)
    model = definition(prepared["definition"])
    with tensor.Device(provider=prepared["provider"]) as device:
        with model(prepared["checkpoint"], prepared["program"], device,
                   checkpoint_options=prepared["checkpoint_options"], **prepared["model_options"]) as executor:
            with Session(executor, batch=batch, chunk=prepared["chunk"]) as session:
                report = run(session, case, repeats=repeats, warmup=warmup)
                report.update(source_identity=prepared["source_identity"],
                              program_sha256=prepared["program_sha256"], policy=prepared["policy"],
                              aggregate_output_tokens_per_second=report["summary"]["decode_tokens_per_second"]["median"],
                              model_throughput_qualified=False)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_workload(prepared, workload, *, window=None, capture=False, deferred=False, out=None):
    """Measure actual ragged token-ID requests, optionally with a compiled head."""
    import tensor
    from akas.config import EngineConfig
    from akas.engine import Engine
    from akas.generation.speculative.decoding import SpeculativeDecoder
    from akas.models.qwen35.mtp import MTPHead

    validate(prepared)
    workload = Path(workload)
    data = json.loads(workload.read_text())
    requests = data["requests"]
    if not requests:
        raise ValueError("research workload must contain requests")
    model = definition(prepared["definition"])
    with tensor.Device(provider=prepared["provider"]) as device:
        with model(prepared["checkpoint"], prepared["program"], device,
                   checkpoint_options=prepared["checkpoint_options"], **prepared["model_options"]) as executor:
            backend = executor
            if window is not None:
                if window not in prepared["windows"]:
                    raise ValueError("requested speculative window was not compiled")
                backend = SpeculativeDecoder(executor, MTPHead(executor), window=window,
                                             capture=capture, deferred=deferred)
            engine = Engine(backend, config=EngineConfig(prefill_chunk=prepared["chunk"], prefill_requests=executor.capacity))
            try:
                handles = [engine.submit(row["prompt_token_ids"], max_tokens=row["output_tokens"]) for row in requests]
                device.synchronize()
                start = time.perf_counter()
                while engine.scheduler.stats["completed"] < len(handles):
                    engine.step()
                device.synchronize()
                elapsed = time.perf_counter() - start
                output_tokens = sum(len(request.output) for request in handles)
                report = dict(status="ok", seconds=elapsed, completed_requests=len(handles),
                              prompt_tokens=sum(len(request.prompt) for request in handles),
                              generated_tokens=output_tokens,
                              aggregate_output_tokens_per_second=output_tokens / elapsed,
                              output_ids=[request.output for request in handles],
                              scheduler=dict(engine.scheduler.stats),
                              speculation=None if window is None else dict(backend.stats),
                              workload_file_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
                              workload_sha256=data.get("sha256"),
                              model_revision=data.get("tokenizer", {}).get("revision"),
                              source_identity=prepared["source_identity"], program_sha256=prepared["program_sha256"],
                              policy=prepared["policy"], model_throughput_qualified=False,
                              timing="synchronized native workload including admission, prefill, decode and fill/drain")
            finally:
                engine.close()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(*, model_definition, preset=None, description=None):
    """Shared CLI for declarative research presets on any compilation target."""
    import argparse

    preset = dict(preset or {})
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--out", type=Path, default=Path("build/research"))
    parser.add_argument("--batches", type=int, nargs="+", default=preset.get("batches", [1, 4, 16, 64, 256]))
    parser.add_argument("--batch", type=int, default=preset.get("batch", 1))
    parser.add_argument("--chunk", type=int, default=preset.get("chunk", 128))
    parser.add_argument("--context", type=int, default=preset.get("context", 4096))
    parser.add_argument("--prompt", type=int, default=preset.get("prompt", 32))
    parser.add_argument("--generated", type=int, default=preset.get("generated", 128))
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--windows", type=int, nargs="*", default=preset.get("windows", []))
    parser.add_argument("--window", type=int)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--deferred", action="store_true")
    parser.add_argument("--provider", choices=("cuda", "webgpu"), default=preset.get("provider", "cuda"))
    parser.add_argument("--target", default=preset.get("target"))
    parser.add_argument("--policy", type=json.loads, default=preset.get("policy", {}))
    parser.add_argument("--model-options", type=json.loads, default=preset.get("model_options", {}))
    parser.add_argument("--checkpoint-options", type=json.loads, default={})
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.prepared:
        prepared = validate(json.loads(args.prepared.read_text()))
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required unless --prepared supplies a compiled program")
        prepared = prepare(model_definition, args.checkpoint, args.out / "program",
                           batches=args.batches, chunk=args.chunk, windows=args.windows,
                           context=args.context, provider=args.provider, target=args.target,
                           policy=args.policy, model_options=args.model_options,
                           checkpoint_options=args.checkpoint_options, deferred=args.deferred)
    if args.compile_only:
        print(json.dumps(prepared, indent=2))
    elif args.workload:
        print(json.dumps(run_workload(prepared, args.workload, window=args.window,
                                      capture=args.capture, deferred=args.deferred,
                                      out=args.out / "report.json"), indent=2))
    else:
        print(json.dumps(run_case(prepared, batch=args.batch, prompt=args.prompt,
                                  generated=args.generated, depth=args.depth, mixed=args.mixed,
                                  repeats=args.repeats, warmup=args.warmup,
                                  out=args.out / "report.json"), indent=2))
