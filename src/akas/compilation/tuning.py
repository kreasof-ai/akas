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
    if len({item.module for item in variants}) != len(variants):
        raise ValueError("tuning implementation modules must be unique")
    baselines = []
    for workload in workloads:
        p = {**parameters, **workload.parameters}
        base = select(kind, p, **options)
        measured = measure(base, p, workload)
        if not measured.valid:
            raise ValueError(f"invalid reference measurement: {workload.name}")
        baselines.append(measured.seconds)
    spaces, seeds, by_family = {}, [], {}
    for item in variants:
        axes = dict(item.space) or {"_default": (True,)}
        if set(axes) & (
            set(parameters) | {name for w in workloads for name in w.parameters}
        ):
            raise ValueError("schedule axes cannot replace semantic parameters")
        spaces[item.module] = axes
        seeds.append(
            dict(
                family=item.module, **{axis: values[0] for axis, values in axes.items()}
            )
        )
        by_family[item.module] = item
    search = ScheduleSearch(seeds, spaces=spaces, width=width)
    best, trials = None, 0
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
        if len(times) != len(workloads):
            continue
        score = max(value / baseline for value, baseline in zip(times, baselines))
        search.record(config, score)
        if best is None or score < best.score:
            best = Result(item.module, schedule, score, tuple(times), trials)
    if best is None:
        raise ValueError("no correctness-qualified candidate covered every workload")
    from dataclasses import replace

    return replace(best, trials=trials)
