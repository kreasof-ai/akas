"""Recurrent commit policies preserve scan arithmetic and state ownership."""


def lower(graph, commit, settings, *, provider, target):
    from dataclasses import replace
    from akas.ops.gdn.operation import journal_prefixes

    if set(settings) - {"commit"}:
        raise ValueError("unknown recurrent policy key")
    mode = settings.get("commit", "snapshots")
    if mode not in ("snapshots", "recompute"):
        raise ValueError("recurrent commit must be snapshots or recompute")
    if mode == "snapshots" or graph.contract != "block-fp8" or commit is None:
        return graph, commit
    scans = [call for call in graph.calls if call.kind in ("gdn_conv", "gdn_scan")]
    if not scans:
        return graph, commit
    if not all(call.parameters.get("full_checkpoints") for call in scans):
        raise ValueError("recompute policy requires a non-mutating verification graph")
    journals = {call.bindings["checkpoints"] for call in scans}
    graph.calls = [
        replace(
            call,
            parameters={
                key: value
                for key, value in call.parameters.items()
                if key != "full_checkpoints"
            },
            bindings={
                key: value
                for key, value in call.bindings.items()
                if key != "checkpoints"
            },
        )
        if call in scans
        else call
        for call in graph.calls
    ]
    commit.calls = [call for call in commit.calls if call.kind != "restore"]
    later = commit.calls
    commit.calls = []
    parameters = scans[0].parameters
    lengths = graph.buffers[graph.request["inputs"]["lengths"]]
    journal_prefixes(
        graph,
        commit,
        slots=parameters["slots"],
        chunk=parameters["chunk"],
        lengths=lengths,
        mode="recompute",
    )
    # Verification parameters disable writes; replay must commit its prefix.
    commit.calls = [
        replace(
            call,
            parameters={
                key: value for key, value in call.parameters.items() if key != "commit"
            },
        )
        for call in commit.calls
    ] + later
    graph.buffers = {
        name: spec for name, spec in graph.buffers.items() if name not in journals
    }
    commit.buffers = graph.buffers
    return graph, commit
