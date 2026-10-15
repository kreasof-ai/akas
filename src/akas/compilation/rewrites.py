"""Conservative operation fusion; no compiler, provider or model imports."""


def token_feedback(graph, inputs, prediction, rows):
    """Make a decode phase replayable without host token copies or readbacks."""
    tokens, lengths = inputs["tokens"], inputs["lengths"]
    if graph.buffers[tokens].shape != (rows,) or graph.buffers[prediction].shape != (
        rows,
    ):
        return False
    existing = tokens == prediction or any(
        call.kind == "next_tokens" and call.arguments == (prediction, tokens)
        for call in graph.calls
    )
    position = inputs.get("positions")
    advance = position is not None and graph.buffers[position].role == "input"
    if existing and not advance:
        return True
    arguments = [graph.buffers[name] for name in (prediction, tokens, lengths)]
    if advance:
        arguments.append(graph.buffers[position])
    graph("token_feedback", dict(rows=rows, advance=advance), *arguments)
    return True


def last_rows(plan):
    """Compute only each slot's final row after its last stateful operation.

    Persistent state still consumes the full chunk. Only the terminal pointwise
    graph is cropped, with gathers for inputs crossing that boundary. Unknown
    operations keep the original graph rather than assuming row independence.
    """
    from dataclasses import replace
    from math import prod

    from akas.graph import Call
    from akas.kernels.registry import row_independent

    last = next(
        (i for i, call in enumerate(plan.calls) if call.kind == "chunk_last"), None
    )
    if last is None:
        return plan
    select = plan.calls[last]
    batch, chunk = select.parameters["r"], select.parameters["chunk"]
    rows = batch * chunk
    boundary = (
        max(
            (
                i
                for i, call in enumerate(plan.calls[:last])
                if any(
                    plan.buffers[name].role == "state"
                    for name in (*call.arguments, *call.bindings.values())
                )
            ),
            default=-1,
        )
        + 1
    )
    tail = plan.calls[boundary:last]
    if not tail or any(
        not row_independent(call)
        or call.parameters.get("r") != rows
        or call.parameters.get("matrix_output", False)
        for call in tail
    ):
        return plan
    produced, gathers, replacements = set(), [], {}
    for call in tail:
        produced.add(call.arguments[-1])
        for name in call.arguments[:-1]:
            spec = plan.buffers[name]
            if spec.role == "weight" or name in produced or name in replacements:
                continue
            if (
                spec.role != "scratch"
                or spec.dtype != "float32"
                or len(spec.shape) != 1
                or prod(spec.shape) % rows
            ):
                return plan
            width = prod(spec.shape) // rows
            target = name + ".last"
            replacements[name] = target
            gathers.append((name, target, width))
    prefix = []
    for source, target, width in gathers:
        plan.buffer(target, (batch * width,), "float32")
        prefix.append(
            Call(
                "chunk_last",
                dict(r=batch, c=width, chunk=chunk),
                (source, target, select.arguments[-1]),
            )
        )
    for name in produced:
        spec = plan.buffers[name]
        plan.buffers[name] = replace(spec, shape=(batch * prod(spec.shape) // rows,))
        object.__setattr__(plan.buffers[name], "_graph", plan)
    cropped = [
        replace(
            call,
            parameters={**call.parameters, "r": batch},
            arguments=tuple(replacements.get(name, name) for name in call.arguments),
        )
        for call in tail
    ]
    after = [
        replace(
            call,
            arguments=tuple(
                select.arguments[0] if name == select.arguments[1] else name
                for name in call.arguments
            ),
        )
        for call in plan.calls[last + 1 :]
    ]
    plan.calls = [*plan.calls[:boundary], *prefix, *cropped, *after]
    return plan


def causal_tail(plan, *, tail=8):
    """Prune a pointwise/convolution suffix after the final KV cache write.

    KV updates retain the complete chunk. Query-only ancestors move after that
    boundary, then all remaining token work uses each request's live suffix.
    Unknown operations or shared query intermediates keep the original graph.
    The retained window must cover the complete convolution receptive field.
    """
    from dataclasses import replace
    from math import prod

    from akas.graph import Call
    from akas.kernels.registry import graph_outputs, row_independent

    caches = [i for i, call in enumerate(plan.calls) if call.kind == "cache_write"]
    selectors = [i for i, call in enumerate(plan.calls) if call.kind == "chunk_last"]
    if not caches or not selectors:
        return plan
    boundary, last = caches[-1], selectors[-1]
    cache, select = plan.calls[boundary], plan.calls[last]
    chunk = cache.parameters.get("chunk", 1)
    slots = select.parameters["r"]
    if not cache.parameters.get("optimized") or not 1 < tail < chunk:
        return plan
    rows = slots * chunk
    if cache.parameters["r"] != rows or select.parameters["chunk"] != chunk:
        return plan
    lengths, positions, enabled = (
        select.arguments[-1],
        cache.arguments[-2],
        cache.arguments[-1],
    )

    def base(name):
        while plan.buffers[name].alias_of is not None:
            name = plan.buffers[name].alias_of
        return name

    def output(call):
        declared = graph_outputs(call, contract=plan.contract)
        if declared is None:
            return None
        scratch = [name for name in declared if plan.buffers[name].role == "scratch"]
        return scratch[0] if len(scratch) == 1 else None

    suffix = plan.calls[boundary + 1 : last]
    allowed = {"attention", "attention_partial", "attention_merge", "conv"}
    if any(
        call.bindings or (call.kind not in allowed and not row_independent(call))
        for call in suffix
    ):
        return plan
    if tail < 1 + sum(
        call.parameters.get("length", 3) - 1 for call in suffix if call.kind == "conv"
    ):
        return plan
    # Query projection/norm/rotation has no consumers in the complete KV path.
    attention = next(
        (call for call in suffix if call.kind in ("attention", "attention_partial")),
        None,
    )
    if attention is None:
        return plan
    producers, consumers = {}, {}
    for index, call in enumerate(plan.calls[:last]):
        for name in call.arguments:
            consumers.setdefault(base(name), set()).add(index)
        if call.kind != "cache_write" and call.arguments:
            destination = output(call)
            if destination is not None:
                producers.setdefault(base(destination), []).append(index)
    moved = set()

    def query_ancestors(name):
        for index in producers.get(base(name), ()):
            if index >= boundary or index in moved:
                continue
            call = plan.calls[index]
            if call.kind not in ("linear", "rms", "rope"):
                continue
            if any(
                other <= boundary and other != index and other not in moved
                for other in consumers.get(base(output(call)), ())
            ):
                continue
            moved.add(index)
            query_ancestors(call.arguments[0])

    query_ancestors(attention.arguments[0])
    cropped_calls = [plan.calls[i] for i in sorted(moved)] + suffix
    if any(output(call) is None for call in cropped_calls):
        return plan
    produced = {base(output(call)) for call in cropped_calls}
    # Attention merge consumes partial buffers but produces only its final output;
    # partial kernels also write statistics. Multiple-output operations require
    # their own declared lowering rather than assuming the final argument alone.
    if any(
        call.kind in ("attention_partial", "attention_merge") for call in cropped_calls
    ):
        return plan
    replacements = {}
    additions = []
    tail_lengths, tail_positions, tail_active = (
        "tail.lengths",
        "tail.positions",
        "tail.active",
    )
    if any(
        name in plan.buffers for name in (tail_lengths, tail_positions, tail_active)
    ):
        return plan
    replacements.update(
        {positions: tail_positions, enabled: tail_active, lengths: tail_lengths}
    )
    for call in cropped_calls:
        for name in call.arguments:
            spec = plan.buffers[name]
            if (
                base(name) in produced
                or name in replacements
                or spec.role in ("weight", "state")
            ):
                continue
            if (
                spec.role != "scratch"
                or spec.dtype != "float32"
                or prod(spec.shape) % rows
            ):
                return plan
            target = name + ".tail"
            replacements[name] = target
            additions.append((name, target, prod(spec.shape) // rows))
    resized = {}
    for name, spec in plan.buffers.items():
        if base(name) not in produced:
            continue
        if spec.role != "scratch" or prod(spec.shape) % rows:
            return plan
        shape = (
            (slots * tail * prod(spec.shape) // rows,)
            if len(spec.shape) == 1
            else (spec.shape[0] * tail // chunk, *spec.shape[1:])
        )
        if prod(shape) != prod(spec.shape) * tail // chunk:
            return plan
        resized[name] = replace(spec, shape=shape)
    plan.buffers.update(resized)
    for spec in resized.values():
        object.__setattr__(spec, "_graph", plan)
    plan.buffer(tail_lengths, (slots,), "int32")
    plan.buffer(tail_positions, (slots * tail,), "int32")
    plan.buffer(tail_active, (slots * tail,), "int32")
    prefix = [
        Call(
            "tail_controls",
            dict(r=slots, chunk=chunk, tail=tail),
            (positions, lengths, tail_positions, tail_active, tail_lengths),
        )
    ]
    for source, target, width in additions:
        plan.buffer(target, (slots * tail * width,), "float32")
        prefix.append(
            Call(
                "chunk_tail",
                dict(r=slots, c=width, chunk=chunk, tail=tail),
                (source, lengths, target),
            )
        )
    rewritten = []
    for call in cropped_calls:
        parameters = dict(call.parameters)
        if call.kind == "conv":
            parameters["chunk"] = tail
        else:
            if "r" not in parameters or parameters["r"] % chunk:
                raise ValueError("tail lowering requires token-local row dimensions")
            parameters["r"] = parameters["r"] * tail // chunk
            if "chunk" in parameters:
                parameters["chunk"] = tail
        rewritten.append(
            replace(
                call,
                parameters=parameters,
                arguments=tuple(
                    replacements.get(name, name) for name in call.arguments
                ),
            )
        )
    select = replace(
        select,
        parameters={**select.parameters, "chunk": tail},
        arguments=(*select.arguments[:-1], tail_lengths),
    )
    plan.calls = [
        *(call for i, call in enumerate(plan.calls[: boundary + 1]) if i not in moved),
        *prefix,
        *rewritten,
        select,
        *plan.calls[last + 1 :],
    ]
    return plan


def optimize(plan):
    """Apply row pruning and rewrites declared by operation owners."""
    from akas.kernels.registry import rewrites

    plan = last_rows(causal_tail(plan))
    for rewrite in rewrites():
        plan = rewrite(plan)
    return plan


def deferred_state(verification):
    """Replace recurrent recomputation with exact operation-owned journals.

    Geometry and bindings come from GDN calls, rather than model configuration
    or buffer-name conventions. The ordinary verification graph remains intact.
    """
    from akas.graph import Graph

    graph = Graph.from_metadata(verification.metadata())
    scans = [
        call
        for call in graph.calls
        if call.kind == "gdn_scan" and call.parameters.get("commit") is False
    ]
    if not scans:
        return None
    first = scans[0].parameters
    slots, window, pool = (first[name] for name in ("slots", "chunk", "pool"))
    prefix = f"journal.w{window}."
    counts = graph.buffer(prefix + "counts", (pool,), "int32", role="state")
    banks = graph.buffer(prefix + "banks", (pool,), "int32", role="state")
    materialize = Graph(contract="bf16", phase="materialize")
    mapping = materialize.buffer("mapping", (slots,), "int32", role="input")
    materialize.buffers.update(
        {name: value for name, value in graph.buffers.items() if value.role == "state"}
    )
    commit = Graph(contract="bf16", phase="commit")
    replacement = {}
    common = dict(slots=slots, window=window, pool=pool)
    for scan in scans:
        if scan.bindings:
            raise ValueError("deferred GDN requires its declared ordered ABI")
        p = scan.parameters
        if tuple(p[name] for name in ("slots", "chunk", "pool")) != (
            slots,
            window,
            pool,
        ):
            raise ValueError("deferred layers must share verification geometry")
        names = dict(
            zip(
                ("q", "k", "v", "g", "beta", "mapping", "lengths", "state", "out"),
                scan.arguments,
            )
        )
        shape = (2, pool, window, p["heads"])
        saved = [
            graph.buffer(
                names["state"] + "." + prefix + field,
                shape + ((p["d"],) if field in ("k", "v") else ()),
                "float32",
                role="state",
            )
            for field in ("k", "v", "g", "b")
        ]
        params = dict(common, heads=p["heads"], d=p["d"], tile=p.get("tile", 32))
        from akas.graph import Call

        replacement[id(scan)] = Call(
            "gdn_deferred",
            params,
            (
                *scan.arguments,
                *(value.name for value in saved),
                counts.name,
                banks.name,
            ),
            label=scan.label,
        )
        for value in saved:
            materialize.buffers[value.name] = value
        materialize(
            "gdn_materialize", params, mapping, names["state"], *saved, counts, banks
        )
    convs = [
        call
        for call in graph.calls
        if call.kind == "gdn_conv" and call.parameters.get("commit") is False
    ]
    for conv in convs:
        x, weight, request, lengths, state, out = conv.arguments
        commit.calls.append(
            Call(
                "conv_commit",
                dict(
                    common,
                    channels=conv.parameters["channels"],
                    projection=conv.parameters["projection"],
                ),
                (x, request, lengths, state),
            )
        )
    scan = scans[0]
    commit.calls.append(
        Call(
            "defer_accept",
            common,
            (scan.arguments[5], scan.arguments[6], counts.name, banks.name),
        )
    )
    materialize("defer_clear", common, mapping, counts)
    graph.calls = [replacement.get(id(call), call) for call in graph.calls]
    used = {name for call in commit.calls for name in call.arguments}
    commit.buffers = {
        name: value for name, value in graph.buffers.items() if name in used
    }
    used = {name for call in materialize.calls for name in call.arguments}
    materialize.buffers = {
        name: value for name, value in materialize.buffers.items() if name in used
    }
    return graph, commit, materialize, (counts.name, banks.name)
