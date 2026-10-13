"""Correctness-qualified beam search over operation-owned spaces and workloads."""

from dataclasses import dataclass, field
import math
from akas.ops.registry import candidates, select


@dataclass(frozen=True)
class Workload:
    name: str
    parameters: dict = field(default_factory=dict)
    # Passed through to the measurer (e.g. ragged lengths or cache positions).
    inputs: object = None


@dataclass(frozen=True)
class Measurement:
    seconds: float
    correct: bool

    @property
    def valid(self):
        return self.correct is True and math.isfinite(self.seconds) and self.seconds > 0


@dataclass(frozen=True)
class Result:
    implementation: str
    schedule: dict
    score: float
    measurements: tuple[float, ...]
    trials: int

    def profile(self, kind, parameters, *, provider, target):
        from tensor.compiler.search import ScheduleProfile

        return ScheduleProfile(
            dict(
                schema="tensor.schedule-profile.v1",
                provider=provider,
                target=target,
                entries=[
                    dict(
                        operation=kind,
                        parameters=parameters,
                        schedule={
                            **self.schedule,
                            "implementation": self.implementation,
                        },
                    )
                ],
            )
        )


def tune(
    kind,
    parameters,
    workloads,
    measure,
    *,
    contract,
    provider="cuda",
    phase="forward",
    target,
    budget=32,
    width=8,
    catalog=None,
    baseline=None,
):
    """Minimize the worst slowdown versus the default on representative workloads.

    The measurer compiles, checks against a reference, times, and releases each
    candidate. Failed correctness rejects the candidate across the entire set.
    The returned target-bound profile can be passed directly to forward.plan.
    """
    from tensor.compiler.search import ScheduleSearch

    workloads = tuple(workloads)
    if not workloads or type(budget) is not int or budget <= 0:
        raise ValueError("tuning requires workloads and a positive candidate budget")
    options = dict(
        contract=contract,
        provider=provider,
        phase=phase,
        target=target,
        catalog=catalog,
    )
    # Workload geometry can qualify a variant even when it is absent from the
    # common semantic selector (for example, every workload has a large batch).
    variants = candidates(kind, {**parameters, **workloads[0].parameters}, **options)
    if not variants:
        raise ValueError("no compatible tuning candidates")
    if len({item.identifier for item in variants}) != len(variants):
        raise ValueError("tuning implementation identifiers must be unique")
    baselines, baseline_modules = [], []
    for workload in workloads:
        p = {**parameters, **workload.parameters}
        preset = {} if baseline is None else dict(baseline)
        preferred = preset.pop("implementation", None)
        protected = set(p) if baseline is None else set(parameters)
        if preferred is not None:
            candidate = select(kind,p,preferred=preferred,**options)
            if candidate.semantic_keys is not None:
                protected = set(candidate.semantic_keys)
        if set(preset) & protected:
            raise ValueError("baseline schedule cannot override semantic parameters")
        base = select(kind, {**p, **preset}, preferred=preferred, **options)
        measured = measure(base, {**p, **preset}, workload)
        if not measured.valid:
            raise ValueError(f"invalid reference measurement: {workload.name}")
        baselines.append(measured.seconds)
        baseline_modules.append(base.identifier)
    spaces, seeds, by_family = {}, [], {}
    for item in variants:
        axes = dict(item.space) or {"_default": (True,)}
        protected = (set(parameters) if item.semantic_keys is None else set(item.semantic_keys))
        protected |= {name for w in workloads for name in w.parameters}
        if set(axes) & protected:
            raise ValueError("schedule axes cannot replace semantic parameters")
        spaces[item.identifier] = axes
        seeds.append(
            dict(
                family=item.identifier, **{axis: values[0] for axis, values in axes.items()}
            )
        )
        by_family[item.identifier] = item
    search = ScheduleSearch(seeds, spaces=spaces, width=width)
    # Keep the established dispatch when search seeds are worse. A search space
    # need not contain the factory's default schedule as an explicit candidate.
    best = (
        Result(baseline_modules[0], {} if baseline is None else {k:v for k,v in baseline.items() if k != "implementation"}, 1.0, tuple(baselines), 0)
        if len(set(baseline_modules)) == 1
        else None
    )
    if len(variants) == 1 and not variants[0].space and best is not None:
        return best
    trials = 0
    while trials < budget:
        try:
            config = search.next()
        except StopIteration:
            break
        trials += 1
        item = by_family[config["family"]]
        schedule = {k: v for k, v in config.items() if k not in ("family", "_default")}
        times = []
        for workload in workloads:
            p = {**parameters, **workload.parameters, **schedule}
            if not item.compatible(
                kind,
                p,
                contract=contract,
                provider=provider,
                phase=phase,
                target=target,
            ):
                break
            measured = measure(item, p, workload)
            if not measured.valid:
                break
            times.append(measured.seconds)
            if measured.seconds > baselines[len(times) - 1]:
                # A configuration already slower on one workload cannot
                # preserve the full set; avoid compiling its remaining shapes.
                break
        if len(times) != len(workloads):
            continue
        score = max(value / baseline for value, baseline in zip(times, baselines))
        search.record(config, score)
        if score <= 1 and (
            best is None or (score, sum(times)) < (best.score, sum(best.measurements))
        ):
            best = Result(item.identifier, schedule, score, tuple(times), trials)
    if best is None:
        raise ValueError(
            "no correctness-qualified candidate preserved every workload baseline"
        )
    from dataclasses import replace

    return replace(best, trials=trials)


@dataclass(frozen=True)
class SearchSettings:
    """Bounded compile-time search; no tuning hooks are required from a model."""

    budget: int = 8
    width: int = 4
    warmup: int = 1
    samples: int = 3
    repeats: int = 3

    def __post_init__(self):
        for name in ("budget", "width", "samples", "repeats"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"search {name} must be a positive integer")
        if type(self.warmup) is not int or self.warmup < 0:
            raise ValueError("search warmup must be nonnegative")


def qualify(bundle, out, *, settings=None, device=None):
    """Automatically qualify every graph operation and build a schedule profile.

    Operation-owned recipes supply finite inputs and independent output/state
    expectations. Tensor owns beam exploration and provider measurement. The
    inference compiler supplies workload geometry and retains a correct baseline.
    """
    from contextlib import ExitStack
    from dataclasses import asdict
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    import tensor
    from tensor.compiler.search import ScheduleProfile
    from tensor.compiler.tuning import measure as time_kernel
    from akas.artifacts.schema import Kernel, Source, identity
    from akas.artifacts.read import artifact_path
    from akas.artifacts.write import publish
    from akas.artifacts.fingerprint import backbone_hashes, _compiler_identity, _hashes
    from akas.compilation.emit import compile_kernel
    from akas.graph import Graph
    from akas.memory.transfer import write
    from akas.ops.registry import validation

    settings = SearchSettings() if settings is None else settings
    graph = Graph.from_metadata(bundle.manifest["forward"])
    out = Path(out)
    trace, entries = [], []
    with ExitStack() as stack:
        if device is None:
            device = stack.enter_context(tensor.Device(provider=bundle.provider))
        if device.info["provider"] != bundle.provider:
            raise ValueError("search device/provider mismatch")
        device_identity = {name: str(value) for name,value in device.info.items()}
        context = dict(
            graph=graph.metadata(), target=bundle.target, provider=bundle.provider,
            device=device_identity, settings=asdict(settings),
            sources=backbone_hashes(), compiler=_compiler_identity(bundle.provider),
            qualification_sources=_hashes(("akas.compilation.tuning", "akas.compilation.selection",
                                           "akas.compilation.state", "akas.compilation.validate")),
            tensor_search_sources=_hashes(("tensor.compiler.search", "tensor.compiler.tuning")),
            baseline=[dict(key=k.key, schedule=k.schedule) for k in bundle.kernels],
            catalog=None if bundle.catalog is None else [repr(item) for item in bundle.catalog],
        )
        context = json.loads(json.dumps(context))
        digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        cache = out / "search-cache" / (digest + ".json")
        if cache.is_file():
            try:
                report = json.loads(cache.read_text())
                if report["identity"] == digest and report["context"] == context:
                    profile = ScheduleProfile(report["profile"])
                    return profile, {**report, "cache_hit": True}
            except (OSError, ValueError, KeyError, TypeError):
                pass

        for key, (kind, semantic) in graph.requirements().items():
            measurements = []

            def measure_candidate(implementation, parameters, workload):
                buffers, loaded = {}, None
                candidate_key = identity(kind, dict(
                    parameters=parameters, implementation=implementation.identifier,
                    contract=graph.contract, phase=graph.phase,
                ))
                try:
                    declaration = Kernel(
                        candidate_key, kind, parameters,
                        Source(implementation.module_name, "make_kernel",
                               implementation.source_arguments(parameters)),
                    )
                    original = next(item for item in bundle.kernels if item.key == key)
                    if original.mapping is not None:
                        from dataclasses import replace
                        from akas.compilation.state import mapped_source
                        declaration = replace(declaration,source=mapped_source(declaration.source,original.mapping))
                    record = compile_kernel(
                        out / "search-kernels", declaration, target=bundle.target,
                        provider=bundle.provider,
                    )
                    loaded = device.load(artifact_path(out / "search-kernels", record))
                    from akas.compilation.validate import kernel_bindings
                    kernel_bindings(graph, key, loaded.manifest["arguments"])
                    validation_parameters = implementation.source_arguments(parameters)[1]
                    if original.mapping is None:
                        case = validation(kind, validation_parameters, loaded.manifest["arguments"])
                    else:
                        from akas.ops.contracts import mapped_validation
                        case = mapped_validation(kind, validation_parameters,
                                                 loaded.manifest["arguments"], original.mapping)
                    dtypes = {row["name"]: row["dtype"] for row in loaded.manifest["arguments"]}
                    for name, host in case.arguments.items():
                        buffers[name] = (device.from_numpy(host, dtype="bfloat16")
                                         if dtypes[name] == "bfloat16" else device.from_numpy(host))
                    arguments = tuple(buffers[row["name"]] for row in loaded.manifest["arguments"])

                    def check(name):
                        expected = case.expected[name]
                        actual = buffers[name].to_numpy()
                        if name in case.comparators:
                            case.comparators[name](actual, expected)
                        elif actual.dtype.kind in "biu" or np.array_equal(case.arguments[name], expected):
                            np.testing.assert_array_equal(actual, expected)
                        else:
                            np.testing.assert_allclose(actual, expected, rtol=case.rtol,
                                                       atol=case.atol, equal_nan=False)

                    loaded.launch(*arguments)
                    for name in case.expected:
                        check(name)
                    written = [name for name in case.expected
                               if not np.array_equal(case.arguments[name], case.expected[name])]

                    def reset():
                        for name in written:
                            write(device, buffers[name], case.arguments[name])

                    timing = time_kernel(
                        device, lambda: loaded.launch(*arguments), reset=reset,
                        warmup=settings.warmup, samples=settings.samples,
                        repeats=settings.repeats,
                    )
                    for name in written:
                        check(name)
                    measurements.append(dict(implementation=implementation.module,
                                             parameters=parameters, status="passed", timing=timing))
                    return Measurement(timing["median_seconds"], True)
                except (AssertionError, ValueError, RuntimeError) as error:
                    measurements.append(dict(implementation=implementation.module,
                                             parameters=parameters, status="rejected",
                                             reason=str(error)))
                    return Measurement(float("inf"), False)
                finally:
                    if loaded is not None:
                        loaded.release()
                    for value in reversed(tuple(buffers.values())):
                        value.release()

            try:
                result = tune(
                    kind, semantic, [Workload(key)], measure_candidate,
                    contract=graph.contract, provider=bundle.provider, phase=graph.phase,
                    target=bundle.target, catalog=bundle.catalog,
                    budget=settings.budget, width=settings.width,
                    baseline=next(k.schedule for k in bundle.kernels if k.key == key),
                )
            except ValueError as error:
                rejected = [row for row in measurements if row["status"] == "rejected"]
                if rejected:
                    raise ValueError(f"{kind} qualification failed: {rejected[-1]['reason']}") from error
                raise
            entries.append(dict(operation=kind, parameters=semantic,
                                schedule={**result.schedule, "implementation": result.implementation}))
            trace.append(dict(key=key, kind=kind, parameters=semantic,
                              selected=asdict(result), candidates=measurements))

        profile_data = dict(schema="tensor.schedule-profile.v1", provider=bundle.provider,
                            target=bundle.target, entries=entries)
        report = dict(identity=digest, context=context, profile=profile_data,
                      operations=trace, cache_hit=False)
        publish(cache, report)
        return ScheduleProfile(profile_data), report
