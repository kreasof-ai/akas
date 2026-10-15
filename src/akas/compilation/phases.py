"""Phase geometry and graph derivation independent of model family."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    name: str
    chunk: int
    filename: str


def token_phases(prefill_chunk):
    if type(prefill_chunk) is not int or prefill_chunk < 1:
        raise ValueError("prefill chunk must be a positive integer")
    phases = []
    if prefill_chunk > 1:
        phases.append(Phase("prefill", prefill_chunk, "prefill.json"))
    phases.append(Phase("decode", 1, "forward.json"))
    return tuple(phases)


def graphs(definition, config, phases, options):
    for phase in phases:
        graph = definition(
            config, **options, **({"chunk": phase.chunk} if phase.chunk > 1 else {})
        )
        graph.phase = phase.name
        yield phase, graph


def round_controls(*, slots, window, width):
    """Declare device controls between proposal, verification, and repair phases."""
    from akas.graph import Graph

    graph = Graph(contract="speculation", phase="round")
    control = graph.buffer("control", (slots, 5), "int32", role="input")
    result = graph.buffer("result", (slots, window + 2), "int32", role="output")
    fields = {}
    for phase, tokens, heads in (
        ("verify", slots * window, slots * window),
        ("draft", slots, slots),
        ("repair", slots * window, slots),
    ):
        for name, count in (
            ("mapping", slots),
            ("positions", slots),
            ("lengths", slots),
            ("tokens", tokens),
            ("head_positions", heads),
            ("head_active", heads),
            ("predicted", heads),
        ):
            fields[phase + "." + name] = graph.buffer(
                phase + "." + name, (count,), "int32", role="input"
            )
        if phase != "verify":
            fields[phase + ".mode"] = graph.buffer(
                phase + ".mode", (1,), "int32", role="input"
            )
    for phase in ("verify", "repair"):
        fields[phase + ".hidden"] = graph.buffer(
            phase + ".hidden", (slots * window, width), "bfloat16", role="input"
        )

    def inputs(phase, *names):
        return [fields[phase + "." + name] for name in names]

    common = dict(slots=slots, window=window)
    names = (
        "mapping",
        "positions",
        "lengths",
        "tokens",
        "head_positions",
        "head_active",
    )
    graph("spec_setup", common, control, *inputs("verify", *names), label="setup")
    for depth in range(1, window - 1):
        graph(
            "spec_draft",
            dict(common, depth=depth),
            control,
            fields["verify.tokens"],
            *inputs("draft", *names, "mode"),
            label=f"draft.{depth}",
        )
        graph(
            "spec_proposal",
            dict(common, depth=depth),
            fields["draft.predicted"],
            fields["verify.tokens"],
            label=f"proposal.{depth}",
        )
    graph(
        "spec_accept",
        common,
        control,
        *inputs("verify", "tokens", "predicted", "lengths"),
        result,
        label="accept",
    )
    graph(
        "spec_repair",
        dict(common, width=width),
        control,
        result,
        fields["verify.hidden"],
        fields["repair.hidden"],
        *inputs("repair", *names, "mode"),
        label="repair",
    )
    graph("spec_result", common, fields["repair.predicted"], result, label="result")
    used = {name for call in graph.calls for name in call.arguments}
    graph.buffers = {name: spec for name, spec in graph.buffers.items() if name in used}
    return graph
