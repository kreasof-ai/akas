"""Find compatible historical schedules; current compilation qualifies them again."""

import json
from pathlib import Path

from akas.benchmarking.history import digest, records
from akas.benchmarking.archive import directory
from akas.optimization.context import identity


def seed(history, graphs, *, provider, target, checkpoint_sha256, hardware):
    from tensor.compiler.search import ScheduleProfile
    from akas.kernels.registry import select

    required = {}
    for phase, graph in graphs:
        for kind, parameters in graph.requirements().values():
            required.setdefault((kind, identity(parameters)), []).append(
                (phase, graph.contract, parameters)
            )
    entries, origins = {}, []
    for record in reversed(
        list(records(history, provider=provider, status="passed", runtime="akas", available_only=True))
    ):
        meta = record["metadata"]
        if (
            meta.get("checkpoint_sha256") != checkpoint_sha256
            or meta.get("target") != target
            or meta.get("device", {}).get("name") != hardware
        ):
            continue
        for phase in record["search"]:
            compatible = []
            for entry in phase["profile"]["entries"]:
                key = entry["operation"], identity(entry["parameters"])
                if key in entries or key not in required:
                    continue
                contexts = required[key]
                if not any(
                    p == phase["phase"] and c == phase["contract"]
                    for p, c, _ in contexts
                ):
                    continue
                try:
                    for purpose, contract, parameters in contexts:
                        select(
                            entry["operation"],
                            parameters,
                            contract=contract,
                            phase=purpose,
                            provider=provider,
                            target=target,
                            preferred=entry["schedule"].get("implementation"),
                        )
                except ValueError:
                    continue
                compatible.append((key, entry))
            if not compatible:
                continue
            evidence = next(
                (a for a in record["artifacts"] if a["path"] == phase["qualification"]),
                None,
            )
            path = directory(history, "runs", record["id"]) / phase["qualification"]
            if evidence is None or digest(path) != evidence["sha256"]:
                raise ValueError("historical incumbent evidence checksum mismatch")
            qualified = json.loads(path.read_text())
            if qualified.get("tuning", {}).get("profile") != phase["profile"]:
                raise ValueError("historical incumbent differs from its qualification")
            entries.update(compatible)
            origins.append(
                dict(run_id=record["id"], phase=phase["phase"], entries=len(compatible))
            )
        if not record["search"]:
            from akas.optimization.context import source_root

            report_roots = [Path(history) / "reports"]
            repo_reports = source_root() / "benchmarks/reports"
            if repo_reports not in report_roots:
                report_roots.append(repo_reports)
            schedule_paths = [
                p
                for r in report_roots
                if r.is_dir()
                for p in sorted(r.glob("*/SCHEDULES.json"), reverse=True)
            ]
            for schedule_path in schedule_paths:
                try:
                    published = json.loads(schedule_path.read_text())
                except (OSError, ValueError):
                    continue
                for conf in published.get("configurations", {}).values():
                    if (
                        conf.get("run_id") != record["id"]
                        or not conf.get("verification_run_id")
                    ):
                        continue
                    ordered_phases = sorted(
                        conf.get("phases", ()),
                        key=lambda p: (
                            0 if p.get("phase") == "decode" else 1,
                            p.get("batch", 0),
                        ),
                    )
                    for phase in ordered_phases:
                        verification = phase.get("verification", {})
                        verified_kernels = verification.get("kernels", ())
                        if not verified_kernels or not all(
                            k.get("status") == "passed" for k in verified_kernels
                        ):
                            continue
                        verified_hashes = {
                            k["key"]: k["sha256"] for k in verified_kernels
                        }
                        kernel_hashes = {
                            k: v.get("sha256")
                            for k, v in phase.get("kernels", {}).items()
                        }
                        if verified_hashes != kernel_hashes:
                            raise ValueError(
                                "historical schedule verification checksum mismatch"
                            )
                        compatible = []
                        for rec in phase.get("kernels", {}).values():
                            key = rec["kind"], identity(rec["parameters"])
                            if key in entries or key not in required:
                                continue
                            contexts = required[key]
                            try:
                                for purpose, contract, parameters in contexts:
                                    select(
                                        rec["kind"],
                                        parameters,
                                        contract=contract,
                                        phase=purpose,
                                        provider=provider,
                                        target=target,
                                        preferred=rec["schedule"].get("implementation"),
                                    )
                            except ValueError:
                                continue
                            compatible.append(
                                (
                                    key,
                                    dict(
                                        operation=rec["kind"],
                                        parameters=rec["parameters"],
                                        schedule=rec["schedule"],
                                    ),
                                )
                            )
                        if compatible:
                            entries.update(compatible)
                            origins.append(
                                dict(
                                    run_id=record["id"],
                                    verification_run_id=conf["verification_run_id"],
                                    phase=phase["phase"],
                                    entries=len(compatible),
                                )
                            )
    profile = (
        None
        if not entries
        else ScheduleProfile(
            dict(
                schema="tensor.schedule-profile.v1",
                provider=provider,
                target=target,
                entries=list(entries.values()),
            )
        )
    )
    return profile, origins
