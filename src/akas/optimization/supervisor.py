"""Deadline-bound, resumable optimization with one isolated GPU worker at a time."""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from akas.benchmarking.history import digest, source_identity, utc, write, rebuild_index
from akas.optimization.context import Context, identity, measurement_limits, source_root
from akas.optimization.planning import recipes, refine, score, search_order, software


@contextmanager
def lock(path):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another optimizer owns " + str(path)) from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def seal(directory, outcome, error):
    """Seal a worker's interrupted provenance after its process group is gone."""
    pointer = directory / "history.json"
    if not pointer.exists():
        return
    path = Path(json.loads(pointer.read_text())["directory"]) / "run.json"
    record = json.loads(path.read_text())
    if record["status"] != "running":
        return
    record.update(status=outcome, completed_at=utc())
    record["metadata"]["error"] = error
    for row in record["results"]:
        if row["status"] == "pending":
            row.update(status="error", error=error)
    report = path.parent / "report.json"
    if report.exists():
        write(
            report,
            dict(
                schema="akas.benchmark.v1",
                metadata=record["metadata"],
                results=record["results"],
            ),
        )
        record["artifacts"].append(
            dict(path="report.json", sha256=digest(report), bytes=report.stat().st_size)
        )
    for name in ("worker.log", "result.json"):
        source = directory / name
        if source.exists():
            import shutil

            target = path.parent / name
            shutil.copyfile(source, target)
            record["artifacts"].append(
                dict(path=name, sha256=digest(target), bytes=target.stat().st_size)
            )
    write(path, record)
    rebuild_index(path.parents[2])


def execute(
    job, directory, *, deadline, seconds, changed=lambda: False, checkpoint=None
):
    """Timeouts include import/JIT/model load, compilation, validation and timing."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    job = {**job, "supervisor_pid": os.getpid()}
    write(directory / "job.json", job)
    result_path = directory / "result.json"
    result_path.unlink(missing_ok=True)
    start = time.monotonic()
    end = min(deadline, start + seconds)
    process = None
    result = None
    telemetry = []
    next_sample = 0
    try:
        if "VK_ICD_FILENAMES" not in os.environ:
            icd = source_root() / "build/baselines/nvidia-graphics/local_egl_icd.json"
            if icd.is_file():
                os.environ["VK_ICD_FILENAMES"] = str(icd)
                graphics = str(icd.parent)
                ld = os.environ.get("LD_LIBRARY_PATH", "")
                if graphics not in ld.split(":"):
                    os.environ["LD_LIBRARY_PATH"] = (
                        f"{graphics}:{ld}" if ld else graphics
                    )
                runtime = source_root() / "build/baselines/runtime"
                runtime.mkdir(exist_ok=True)
                runtime.chmod(0o700)
                os.environ.setdefault("XDG_RUNTIME_DIR", str(runtime))
        with (directory / "worker.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "akas.optimization.worker",
                    str(directory / "job.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if checkpoint:
                checkpoint(process.pid)
            while process.poll() is None:
                phase = None
                progress_path = directory / "progress.json"
                if progress_path.is_file():
                    try:
                        phase = json.loads(progress_path.read_text()).get("phase")
                    except (OSError, ValueError):
                        pass
                if time.monotonic() >= end:
                    result = dict(
                        status="budget_exhausted",
                        error=f"worker wall-clock deadline exceeded during {phase or 'startup'}",
                    )
                    break
                if changed():
                    result = dict(
                        status="invalidated",
                        error="source or campaign inputs changed during optimization",
                    )
                    break
                device = job.get("device", {})
                if device.get("uuid") and time.monotonic() >= next_sample:
                    next_sample = time.monotonic() + 1
                    try:
                        raw = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--id=" + device["uuid"],
                                "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=1,
                        )
                        used = int(raw.strip()) * 1024**2
                        telemetry.append(
                            dict(
                                seconds=time.monotonic() - start, device_used_bytes=used
                            )
                        )
                        write(
                            directory / "telemetry.json",
                            dict(
                                scope="sampled whole-device use",
                                samples=telemetry,
                                sampled_peak_device_bytes=max(
                                    s["device_used_bytes"] for s in telemetry
                                ),
                            ),
                        )
                        ceiling = (
                            job["context"].get("memory_bytes")
                            or device.get("total_bytes", float("inf")) * 0.95
                        )
                        if used > ceiling:
                            result = dict(
                                status="oom",
                                error="sampled device memory exceeded the GPU ceiling",
                            )
                            break
                    except (OSError, subprocess.SubprocessError, ValueError):
                        pass
                time.sleep(min(0.2, max(0, end - time.monotonic())))
            if result is None:
                result = (
                    json.loads(result_path.read_text())
                    if result_path.exists()
                    else dict(
                        status="failed",
                        error=f"worker exited with code {process.returncode}",
                    )
                )
    except BaseException:
        if process is not None:
            terminate(process)
        seal(directory, "interrupted", "supervisor interrupted")
        raise
    finally:
        if process is not None:
            terminate(process)
    result["elapsed_seconds"] = time.monotonic() - start
    result["memory_samples"] = telemetry
    result["sampled_peak_device_bytes"] = max(
        (s["device_used_bytes"] for s in telemetry), default=None
    )
    result["completed_at"] = utc()
    write(result_path, result)
    seal(
        directory,
        "failed" if result["status"] == "oom" else result["status"],
        result.get("error", "worker did not seal its record"),
    )
    return result


def publish(candidate, out):
    """Atomically replace the serving pointer after all candidate checks passed."""
    import os
    from akas.optimization.deployment import load

    path = Path(candidate)
    data = load(path)
    data["program"] = os.path.relpath((path.parent / data["program"]).resolve(), out)
    for entry in data["artifacts"]:
        entry["path"] = os.path.relpath((path.parent / entry["path"]).resolve(), out)
    write(Path(out) / "deployment.json", data)


def verify_result(result):
    for name in ("report", "evidence", "deployment"):
        if digest(result[name]) != result[name + "_sha256"]:
            raise ValueError("optimization evidence changed: " + name)


def optimize(context: Context, out, *, budget, history="benchmarks"):
    from akas.cli import checkpoint_digest

    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("optimization budget must be positive")
    started = time.monotonic()
    deadline = started + budget
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = checkpoint_digest(context.checkpoint)
    source = source_identity(source_root())
    settings = context.metadata()
    frozen = dict(
        context=settings,
        checkpoint_sha256=checkpoint_hash,
        source_sha256=source["tree_sha256"],
        software=software(),
        measurement_limits=measurement_limits(),
    )
    fingerprint = identity(frozen)
    with (
        lock(out / ".lock"),
        lock(
            Path(tempfile.gettempdir())
            / f"akas-optimization-device-{context.device}.lock"
        ),
    ):
        state_path = out / "campaign.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["identity"] != fingerprint:
                raise ValueError(
                    "campaign source, checkpoint, dependencies or context changed; use a new output directory"
                )
        else:
            state = dict(
                schema="akas.optimization.v1",
                identity=fingerprint,
                created_at=utc(),
                inputs=frozen,
                probes=[],
                jobs=[],
                winner=None,
                deployment=None,
                spent_seconds=0,
            )
        for previous in state["jobs"]:
            if previous["status"] == "running":
                previous.update(
                    status="interrupted",
                    completed_at=utc(),
                    error="previous supervisor exited",
                )
                seal(Path(previous["directory"]), "interrupted", previous["error"])
        if state.get("deployment"):
            from akas.optimization.deployment import load

            load(state["deployment"])
        state.pop("error", None)
        invocation = dict(started_at=utc(), budget_seconds=budget, status="running")
        state.setdefault("invocations", []).append(invocation)
        state.update(status="running", updated_at=utc(), last_budget_seconds=budget)
        write(state_path, state)
        # Cheap stat checks are not evidence: rehash source periodically, and the
        # complete checkpoint again before publication/end of this invocation.
        last_check = [0, False]

        def changed():
            if time.monotonic() - last_check[0] > 2:
                last_check[:] = [
                    time.monotonic(),
                    source_identity(source_root())["tree_sha256"]
                    != source["tree_sha256"],
                ]
            return last_check[1]

        def save():
            state["updated_at"] = utc()
            write(state_path, state)

        common = dict(
            context=settings,
            identity=fingerprint,
            checkpoint_sha256=checkpoint_hash,
            history=str(Path(history).resolve()),
        )
        try:
            providers = (
                ("cuda", "webgpu") if context.backend == "auto" else (context.backend,)
            )
            probes = []
            for provider in providers:
                if time.monotonic() >= deadline:
                    break
                probe = execute(
                    {**common, "action": "probe", "provider": provider},
                    out / "probes" / provider,
                    deadline=deadline,
                    seconds=min(45, budget * 0.15),
                    changed=changed,
                )
                if probe["status"] == "ok":
                    stable = lambda p: {
                        k: v for k, v in p["device"].items() if k != "free_bytes"
                    }
                    previous = state.setdefault("hardware", {}).get(provider)
                    if previous is not None and previous != stable(probe):
                        raise ValueError(
                            "campaign hardware changed; use a new output directory"
                        )
                    state["hardware"][provider] = stable(probe)
                probes.append(probe)
            state["probes"] = probes
            save()
            skip_autotune = os.environ.get("AKAS_SKIP_AUTOTUNE") == "1"
            plans = recipes(settings, probes)
            if plans and not skip_autotune:
                plans = search_order(plans, settings["workload"])
            elif plans and skip_autotune:
                if settings.get("mode") == "mtp":
                    ar_graph = [
                        p for p in plans if p["window"] is None and p["graphs"]
                    ] or [p for p in plans if p["window"] is None][:1]
                    mtp_graph = [
                        p for p in plans if p["window"] is not None and p["graphs"]
                    ][:1] or [p for p in plans if p["window"] is not None][:1]
                    plans = ar_graph + mtp_graph
                else:
                    plans = [p for p in plans if p["graphs"]][:1] or plans[:1]
            state["planned_candidates"] = len(plans)
            reference_job = next(
                (j for j in state["jobs"] if j.get("status") == "passed"), None
            )
            if (
                reference_job is None
                and skip_autotune
                and settings.get("mode") == "mtp"
                and out.name.endswith("-mtp")
            ):
                sibling = out.parent / (out.name[:-4] + "-ar") / "campaign.json"
                if sibling.is_file():
                    try:
                        sib_state = json.loads(sibling.read_text())
                        sib_job = next(
                            (
                                j
                                for j in sib_state.get("jobs", [])
                                if j.get("status") == "passed"
                            ),
                            None,
                        )
                        if sib_job is not None:
                            verify_result(sib_job)
                            reference_job = sib_job
                            plans = [p for p in plans if p["window"] is not None]
                            state["planned_candidates"] = len(plans)
                    except (OSError, ValueError, KeyError):
                        pass
            reference = None
            if reference_job:
                verify_result(reference_job)
                reference = json.loads(Path(reference_job["report"]).read_text())
                from akas.optimization.deployment import load

                load(reference_job["deployment"])
            performance_reference = reference
            if settings.get("mode") == "mtp":
                previous_mtp = next(
                    (
                        j
                        for j in state["jobs"]
                        if j.get("status") == "passed" and j["recipe"]["window"]
                    ),
                    None,
                )
                performance_reference = None
                if previous_mtp:
                    verify_result(previous_mtp)
                    performance_reference = json.loads(
                        Path(previous_mtp["report"]).read_text()
                    )
            pending = False
            for plan_index, recipe in enumerate(plans):
                if "refine" in recipe:
                    if state["winner"] is None:
                        continue
                    incumbent = next(
                        j["recipe"]
                        for j in state["jobs"]
                        if j["recipe"]["id"] == state["winner"]["recipe"]
                    )
                    recipe = refine(incumbent, recipe["refine"])
                    if recipe is None:
                        continue
                if time.monotonic() >= deadline:
                    pending = True
                    break
                if reference is None and recipe["window"] is not None:
                    # Speculation must preserve an independently measured AR
                    # target; a broken reference cannot qualify itself as MTP.
                    continue
                if changed():
                    raise ValueError("campaign source changed during optimization")
                previous = [
                    j for j in state["jobs"] if j["recipe"]["id"] == recipe["id"]
                ]
                if any(
                    j["status"] not in ("running", "interrupted", "budget_exhausted")
                    for j in previous
                ):
                    continue
                device = next(
                    p["device"]
                    for p in probes
                    if p.get("provider") == recipe["provider"] and p["status"] == "ok"
                )
                key = device.get(
                    "uuid", identity({k: device.get(k) for k in ("name", "ordinal")})
                )
                directory = (
                    out / "jobs" / recipe["id"] / f"attempt-{len(previous) + 1:04d}"
                )
                entry = dict(
                    recipe=recipe,
                    status="running",
                    started_at=utc(),
                    directory=str(directory),
                )
                state["jobs"].append(entry)
                save()
                job = {
                    **common,
                    "action": "candidate",
                    "recipe": recipe,
                    "device": device,
                }
                shared_cache = source_root() / "build/shared-program-cache"
                sibling_programs = [
                    str(p)
                    for p in sorted(out.parent.glob("*/jobs/*/attempt-*/program"))
                    if p != directory / "program"
                ]
                local_programs = [
                    str(Path(j["directory"]) / "program") for j in state["jobs"][:-1]
                ]
                sources = (
                    ([str(shared_cache)] if shared_cache.is_dir() else [])
                    + local_programs
                    + sibling_programs[-2:]
                )
                job["cache_sources"] = list(dict.fromkeys(sources))
                if reference_job:
                    job.update(
                        reference=reference_job["report"],
                        evidence=reference_job["evidence"],
                        parent_run=reference_job["run_id"],
                    )
                remaining = deadline - time.monotonic()
                from akas.optimization.planning import worker_allowance

                allowance = worker_allowance(
                    budget,
                    remaining,
                    state["jobs"],
                    reference=reference,
                    provider=recipe["provider"],
                )
                print(
                    f"optimize {recipe['id']} {recipe['provider']} chunk={recipe['chunk']} window={recipe['window']} allowance={allowance:.1f}s",
                    flush=True,
                )
                with lock(
                    Path(tempfile.gettempdir()) / f"akas-optimization-{key}.lock"
                ):
                    result = execute(
                        job,
                        directory,
                        deadline=deadline,
                        seconds=allowance,
                        changed=changed,
                        checkpoint=lambda pid: (entry.update(pid=pid), save()),
                    )
                entry.update(result)
                cache = directory / "program/search-cache"
                entry["completed_operations"] = len(list(cache.glob("*.op.json")))
                if shared_cache.is_dir():
                    import shutil

                    def _link(src, dst):
                        Path(dst).unlink(missing_ok=True)
                        try:
                            os.link(src, dst)
                        except OSError:
                            shutil.copy2(src, dst)

                    for _sub in ("search-cache", "search-kernels", "compiler-cache"):
                        _s = directory / "program" / _sub
                        if _s.is_dir():
                            shutil.copytree(
                                _s,
                                shared_cache / _sub,
                                copy_function=_link,
                                dirs_exist_ok=True,
                            )
                save()
                if result["status"] == "invalidated":
                    raise ValueError(result["error"])
                if result["status"] != "passed":
                    # Complete a usable baseline before spending time on broader
                    # tuning. Retry once only when compilation made progress.
                    prior_progress = max(
                        (j.get("completed_operations", 0) for j in previous), default=0
                    )
                    if (
                        reference is None
                        and result["status"] == "budget_exhausted"
                        and len(previous) < 1
                        and entry["completed_operations"] > prior_progress
                    ):
                        plans.insert(plan_index + 1, recipe)
                    continue
                verify_result(result)
                report = json.loads(Path(result["report"]).read_text())
                if reference is None:
                    reference, reference_job = report, entry
                if performance_reference is None and (
                    settings.get("mode") != "mtp" or recipe["window"]
                ):
                    performance_reference = report
                value = score(report, performance_reference or reference, settings)
                entry["score"] = value
                eligible = settings.get("mode") != "mtp" or recipe["window"] is not None
                if (
                    eligible
                    and value is not None
                    and (
                        state["winner"] is None
                        or value > state["winner"]["score"] * 1.01
                    )
                ):
                    if (
                        checkpoint_digest(context.checkpoint) != checkpoint_hash
                        or changed()
                    ):
                        raise ValueError("campaign inputs changed before publication")
                    publish(result["deployment"], out)
                    state["winner"] = dict(
                        recipe=recipe["id"],
                        score=value,
                        run_id=result["run_id"],
                        candidate=result["deployment"],
                    )
                    state["deployment"] = str(out / "deployment.json")
                save()
                if skip_autotune and state["winner"] is not None:
                    break
            if checkpoint_digest(context.checkpoint) != checkpoint_hash or changed():
                raise ValueError("campaign inputs changed during optimization")
            state["status"] = (
                "budget_exhausted"
                if pending or time.monotonic() >= deadline
                else "complete"
            )
            if any(
                j["status"] in ("budget_exhausted", "interrupted")
                and not any(
                    k["recipe"]["id"] == j["recipe"]["id"] and k["status"] == "passed"
                    for k in state["jobs"]
                )
                for j in state["jobs"]
            ):
                state["status"] = "budget_exhausted"
            if not plans:
                state["status"] = (
                    "unsupported"
                    if time.monotonic() < deadline
                    and all(p["status"] != "budget_exhausted" for p in probes)
                    else "budget_exhausted"
                )
            elif state["status"] == "complete" and state["deployment"] is None:
                state.update(
                    status="failed",
                    error="no candidate qualified for every workload shape and constraint",
                )
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            raise
        except BaseException as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            elapsed = time.monotonic() - started
            state["spent_seconds"] += elapsed
            invocation.update(
                completed_at=utc(),
                elapsed_seconds=elapsed,
                status=state["status"],
                error=state.get("error"),
            )
            save()
        return dict(
            status=state["status"],
            deployment=state["deployment"],
            winner=state["winner"],
            campaign=str(state_path),
            spent_seconds=state["spent_seconds"],
        )
