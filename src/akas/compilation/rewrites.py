"""Conservative operation fusion; no compiler, provider or model imports."""


def token_feedback(graph, inputs, prediction, rows):
    """Make a decode phase replayable without host token copies or readbacks."""
    tokens, lengths = inputs["tokens"], inputs["lengths"]
    if graph.buffers[tokens].shape != (rows,) or graph.buffers[prediction].shape != (rows,):
        return False
    existing = tokens == prediction or any(
        call.kind == "next_tokens" and call.arguments == (prediction, tokens)
        for call in graph.calls)
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
    from akas.ops.registry import row_independent

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


def optimize(plan):
    """Apply row pruning and rewrites declared by operation owners."""
    from akas.ops.registry import rewrites

    plan = last_rows(plan)
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
    scans = [call for call in graph.calls if call.kind=="gdn_scan" and call.parameters.get("commit") is False]
    if not scans:
        return None
    first = scans[0].parameters
    slots, window, pool = (first[name] for name in ("slots","chunk","pool"))
    prefix = f"journal.w{window}."
    counts = graph.buffer(prefix+"counts",(pool,),"int32",role="state")
    banks = graph.buffer(prefix+"banks",(pool,),"int32",role="state")
    materialize = Graph(contract="bf16",phase="materialize")
    mapping = materialize.buffer("mapping",(slots,),"int32",role="input")
    materialize.buffers.update({name:value for name,value in graph.buffers.items() if value.role=="state"})
    commit = Graph(contract="bf16",phase="commit")
    replacement = {}
    common = dict(slots=slots,window=window,pool=pool)
    for scan in scans:
        if scan.bindings:
            raise ValueError("deferred GDN requires its declared ordered ABI")
        p = scan.parameters
        if tuple(p[name] for name in ("slots","chunk","pool")) != (slots,window,pool):
            raise ValueError("deferred layers must share verification geometry")
        names = dict(zip(("q","k","v","g","beta","mapping","lengths","state","out"),scan.arguments))
        shape = (2,pool,window,p["heads"])
        saved = [graph.buffer(names["state"]+"."+prefix+field,shape+((p["d"],) if field in ("k","v") else ()),"float32",role="state")
                 for field in ("k","v","g","b")]
        params = dict(common,heads=p["heads"],d=p["d"],tile=p.get("tile",32))
        from akas.graph import Call
        replacement[id(scan)] = Call("gdn_deferred",params,
                                     (*scan.arguments,*(value.name for value in saved),counts.name,banks.name),label=scan.label)
        for value in saved:
            materialize.buffers[value.name] = value
        materialize("gdn_materialize",params,mapping,names["state"],*saved,counts,banks)
    convs = [call for call in graph.calls if call.kind=="gdn_conv" and call.parameters.get("commit") is False]
    for conv in convs:
        x,weight,request,lengths,state,out = conv.arguments
        commit.calls.append(Call("conv_commit",dict(common,channels=conv.parameters["channels"],projection=conv.parameters["projection"]),
                                 (x,request,lengths,state)))
    scan = scans[0]
    commit.calls.append(Call("defer_accept",common,(scan.arguments[5],scan.arguments[6],counts.name,banks.name)))
    materialize("defer_clear",common,mapping,counts)
    graph.calls = [replacement.get(id(call),call) for call in graph.calls]
    used = {name for call in commit.calls for name in call.arguments}
    commit.buffers = {name:value for name,value in graph.buffers.items() if name in used}
    used = {name for call in materialize.calls for name in call.arguments}
    materialize.buffers = {name:value for name,value in materialize.buffers.items() if name in used}
    return graph,commit,materialize,(counts.name,banks.name)
