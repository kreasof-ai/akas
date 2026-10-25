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
