"""Run a declared model matrix serially, with cumulative campaign and per-model budgets."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from akas.benchmarking.history import source_identity, utc, write
from akas.optimization.context import Context, discover, duration, identity, workload
from akas.optimization.supervisor import lock, optimize


def declarations(spec, root):
    return {
        name: json.loads((root / "benchmarks/models" / (name + ".json")).read_text())
        for name in spec["models"]
    }


def entries(spec, root):
    result = []
    for name, model in declarations(spec, root).items():
        for precision in spec["formats"]:
            path = (root / model["weights"][precision]["path"]).resolve()
            try:
                definition, error = discover(path), None
            except (ValueError, OSError) as failure:
                definition, error = None, f"{type(failure).__name__}: {failure}"
            for backend in spec["backends"]:
                for cache in spec["caches"]:
                    for mode in model["modes"]:
                        key = "-".join((name, precision, backend, cache, mode))
                        if error:
                            result.append(
                                dict(
                                    id=key,
                                    status="failed",
                                    error=error,
                                    checkpoint=str(path),
                                )
                            )
                            continue
                        context = Context(
                            str(path),
                            definition,
                            workload(spec["workload"]),
                            backend=backend,
                            kv_dtype=None if cache == "default" else cache,
                            mode=mode,
                        )
                        result.append(dict(id=key, context=asdict(context)))
    return result


def run(specification, out, *, budget, slice_seconds, history):
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(Path(specification).read_text())
    models = declarations(spec, root)
    frozen = dict(
        spec=spec,
        models=models,
        workload=workload(spec["workload"]),
        source=source_identity(root)["tree_sha256"],
    )
    fingerprint = identity(frozen)
    cap = duration(spec.get("configuration_budget", "1h"))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / ".lock"):
        path = out / "campaign.json"
        if path.exists():
            state = json.loads(path.read_text())
            if state["identity"] != fingerprint or state["budget_seconds"] != budget:
                raise ValueError(
                    "campaign source, specification, model declarations or total budget changed; use a new output directory"
                )
        else:
            state = dict(
                schema="akas.matrix-campaign.v1",
                identity=fingerprint,
                specification=spec,
                models=models,
                source=source_identity(root),
                created_at=utc(),
                budget_seconds=budget,
                configuration_budget_seconds=cap,
                spent_seconds=0,
                entries=entries(spec, root),
            )
        # A hard-killed coordinator cannot know how much of its last grant was
        # used. Charge the reserved grant rather than silently reset that time.
        active = state.pop("active", None)
        if active:
            state["spent_seconds"] = max(
                state["spent_seconds"], active["spent_before"] + active["allowance"]
            )
            entry = next(e for e in state["entries"] if e["id"] == active["id"])
            entry["spent_seconds"] = entry.get("spent_seconds", 0) + active["allowance"]
            entry["recovered_interruption"] = utc()
        previous_spent = state["spent_seconds"]

        def save():
            state.update(
                spent_seconds=previous_spent + time.monotonic() - started,
                updated_at=utc(),
            )
            write(path, state)

        state["status"] = "running"
        save()
        try:
            while time.monotonic() - started + previous_spent < budget:
                if (
                    source_identity(root)["tree_sha256"] != frozen["source"]
                    or declarations(spec, root) != models
                    or json.loads(Path(specification).read_text()) != spec
                    or workload(spec["workload"]) != frozen["workload"]
                ):
                    state["status"] = "invalidated"
                    raise ValueError("campaign inputs changed during execution")
                unfinished = [
                    e
                    for e in state["entries"]
                    if e.get("status") not in ("complete", "unsupported", "failed")
                ]
                pending = [e for e in unfinished if e.get("spent_seconds", 0) < cap]
                if not pending:
                    state["status"] = (
                        "configuration_budgets_exhausted" if unfinished else "complete"
                    )
                    break
                # Give each configuration a first pass before deepening one.
                entry = min(pending, key=lambda e: e.get("spent_seconds", 0))
                remaining = budget - (time.monotonic() - started + previous_spent)
                allowance = min(
                    slice_seconds, remaining, cap - entry.get("spent_seconds", 0)
                )
                print(entry["id"], f"allowance={allowance:.1f}s", flush=True)
                save()
                state["active"] = dict(
                    id=entry["id"],
                    allowance=allowance,
                    spent_before=state["spent_seconds"],
                    started_at=utc(),
                )
                write(path, state)
                tick = time.monotonic()
                try:
                    result = optimize(
                        Context(**entry["context"]),
                        out / entry["id"],
                        budget=allowance,
                        history=history,
                    )
                    entry.update(
                        status=result["status"],
                        deployment=result.get("deployment"),
                        winner=result.get("winner"),
                    )
                except KeyboardInterrupt:
                    entry["status"] = "interrupted"
                    raise
                except Exception as error:
                    entry.update(
                        status="failed", error=f"{type(error).__name__}: {error}"
                    )
                finally:
                    entry["spent_seconds"] = (
                        entry.get("spent_seconds", 0) + time.monotonic() - tick
                    )
                    entry["updated_at"] = utc()
                    state.pop("active", None)
                    save()
            else:
                state["status"] = "budget_exhausted"
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            raise
        finally:
            save()
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=Path("benchmarks/campaigns/standard-v1.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--budget",
        required=True,
        help="cumulative allowance, e.g. 4d; downtime is excluded",
    )
    parser.add_argument(
        "--slice",
        default="30m",
        help="maximum grant before moving to the next configuration",
    )
    parser.add_argument("--history", type=Path, default=Path("benchmarks"))
    parser.add_argument(
        "--list",
        action="store_true",
        help="show the requested matrix without running GPU work",
    )
    args = parser.parse_args()
    if args.list:
        print(
            json.dumps(entries(json.loads(args.spec.read_text()), Path.cwd()), indent=2)
        )
        return
    state = run(
        args.spec,
        args.out,
        budget=duration(args.budget),
        slice_seconds=duration(args.slice),
        history=args.history,
    )
    print(
        json.dumps(
            dict(
                status=state["status"],
                spent_seconds=state["spent_seconds"],
                configurations=len(state["entries"]),
            ),
            indent=2,
        )
    )
    if any(e.get("status") == "failed" for e in state["entries"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
