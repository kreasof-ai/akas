"""One graph compilation and program publication coordinator."""

from pathlib import Path

from akas.artifacts.schema import Bundle
from akas.artifacts.write import publish
from akas.compilation.emit import compile_kernel
from akas.compilation.selection import kernels as assemble


def build(
    bundle: Bundle,
    out,
    *,
    progress=None,
    search=True,
    search_settings=None,
    device=None,
):
    """Build any profile's kernel declarations through one common pipeline."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / bundle.filename
    if not destination.resolve().is_relative_to(out.resolve()):
        raise ValueError("manifest path must remain inside the output bundle")
    if len({kernel.key for kernel in bundle.kernels}) != len(bundle.kernels):
        raise ValueError("duplicate kernel declaration")
    destination.unlink(missing_ok=True)
    manifest = dict(bundle.manifest)
    if "forward" in manifest:
        if search:
            from akas.compilation.selection import kernels
            from akas.compilation.tuning import qualify
            from akas.graph import Graph

            schedules, report = qualify(
                bundle, out, settings=search_settings, device=device
            )
            graph = Graph.from_metadata(manifest["forward"])
            mappings = {
                item.key: item.mapping
                for item in bundle.kernels
                if item.mapping is not None
            }
            bundle.kernels = kernels(
                graph.requirements(),
                contract=graph.contract,
                provider=bundle.provider,
                phase=graph.phase,
                target=bundle.target,
                schedules=schedules,
                catalog=bundle.catalog,
                record_implementation=True,
            )
            if mappings:
                from dataclasses import replace

                from akas.compilation.state import mapped_source

                bundle.kernels = [
                    replace(
                        item,
                        source=mapped_source(item.source, mappings[item.key]),
                        mapping=mappings[item.key],
                    )
                    if item.key in mappings
                    else item
                    for item in bundle.kernels
                ]
            manifest["tuning"] = report
        else:
            manifest["tuning"] = {"status": "explicitly disabled"}
    records = dict(manifest.get("kernels", {}))
    for kernel in bundle.kernels:
        records[kernel.key] = compile_kernel(
            out,
            kernel,
            target=kernel.target or bundle.target,
            provider=bundle.provider,
            source_directory=bundle.source_directory,
            artifact_directory=bundle.artifact_directory,
            path_field=bundle.path_field,
        )
        if progress:
            progress(kernel.kind, kernel.parameters)
    manifest["kernels"] = records
    if "status" in manifest:
        manifest["status"] = "built"
    publish(destination, manifest)
    return manifest


def plan(
    forward,
    *,
    provider="cuda",
    target=None,
    filename="forward.json",
    schedules=None,
    catalog=None,
):
    from akas.compilation.buffers import scratch_slots

    target = target or ("sm_89" if provider == "cuda" else "webgpu-portable-v1")
    kernels = assemble(
        forward.requirements(),
        contract=forward.contract,
        provider=provider,
        phase=forward.phase,
        target=target,
        schedules=schedules,
        catalog=catalog,
        record_implementation=True,
    )
    from akas.compilation.state import map_requests

    forward, kernels = map_requests(forward, kernels)
    return Bundle(
        dict(
            schema="akas.forward.v1",
            target=target,
            provider=provider,
            forward=forward.metadata(),
            allocations=scratch_slots(forward),
        ),
        filename,
        kernels,
        provider=provider,
        catalog=None if catalog is None else tuple(catalog),
    )


def produce(forward, out, *, search=True, search_settings=None, device=None, **options):
    return build(
        plan(forward, **options),
        out,
        search=search,
        search_settings=search_settings,
        device=device,
    )


def compile_model(
    checkpoint,
    out,
    *,
    definition,
    batch=1,
    capacity=4096,
    provider="cuda",
    target=None,
    kv_dtype="float16",
    schedules=None,
    max_weight_bytes=64 * 1024 * 1024,
    prefill_chunk=1,
    search=True,
    search_settings=None,
    device=None,
):

    from akas.compilation.phases import graphs, token_phases

    options = dict(
        batch=batch,
        capacity=capacity,
        provider=provider,
        kv_dtype=kv_dtype,
        max_weight_bytes=max_weight_bytes,
        **getattr(checkpoint, "definition_options", {}),
    )
    result = None
    for phase, forward in graphs(
        definition, checkpoint.config, token_phases(prefill_chunk), options
    ):
        bundle = plan(
            forward,
            provider=provider,
            target=target,
            schedules=schedules,
            filename=phase.filename,
        )
        bundle.source_directory = f"phases/{phase.name}/sources"
        bundle.artifact_directory = f"phases/{phase.name}/kernels"
        bundle.manifest["reference"] = dict(
            batch=batch,
            capacity=capacity,
            kv_dtype=kv_dtype,
            max_weight_bytes=max_weight_bytes,
            **({"prefill_chunk": prefill_chunk} if prefill_chunk > 1 else {}),
            **(
                {"matrix_dtype": "float16"}
                if any(
                    b.role == "weight" and b.dtype == "float16"
                    for b in forward.buffers.values()
                )
                else {}
            ),
        )
        built = build(
            bundle, out, search=search, search_settings=search_settings, device=device
        )
        if phase.name == "decode":
            result = built
    return result


def compile_program(
    checkpoint,
    out,
    *,
    definition,
    geometries,
    max_requests,
    max_context,
    inputs=None,
    prediction=None,
    reset_buffers=(),
    upload_options=None,
    provider="cuda",
    target=None,
    search=True,
    search_settings=None,
    round_windows=(),
    deferred=False,
    model_options=None,
    schedules=None,
    device=None,
    policy=None,
    catalog=None,
):
    """Compile all model-supplied phase graphs through the common pipeline.

    Definitions receive geometry and state-pool bounds, not a runtime adapter.
    Input roles and reset declarations are data; no model-owned tuner is needed.
    """
    from dataclasses import asdict
    from uuid import uuid4

    from akas.artifacts.write import checksum

    out = Path(out)
    if any(
        type(value) is not int or value < 1 for value in (max_requests, max_context)
    ):
        raise ValueError("program capacities must be positive integers")
    geometries = tuple(geometries)
    if not geometries:
        raise ValueError("program needs at least one geometry")
    # Each build owns immutable phase/artifact paths. A failed replacement must
    # leave the previously published program usable, including its kernel bytes.
    generation = f"programs/{uuid4().hex}"
    model_options = dict(model_options or {})
    if set(model_options) & {
        "rows",
        "chunk",
        "max_requests",
        "max_context",
        "provider",
        "phase",
    }:
        raise ValueError("model options cannot override program geometry or target")
    records = []
    state_specs = {}
    weight_specs = {}
    identities = set()
    resets = set(reset_buffers)
    for geometry in geometries:
        if len(geometry) not in (2, 3):
            raise ValueError(
                "program geometry requires rows, chunk, and optional phase"
            )
        rows, chunk = geometry[:2]
        purpose = geometry[2] if len(geometry) == 3 else None
        if (
            type(rows) is not int
            or type(chunk) is not int
            or not 1 <= rows <= max_requests
            or not 1 <= chunk <= max_context
        ):
            raise ValueError("invalid program geometry")
        phase = purpose or ("decode" if chunk == 1 else "prefill")
        if not isinstance(phase, str) or not phase.isidentifier():
            raise ValueError("program phase must be an identifier")
        if (phase, rows, chunk) in identities:
            raise ValueError("duplicate program geometry")
        identities.add((phase, rows, chunk))
        composed = definition(
            checkpoint,
            rows=rows,
            chunk=chunk,
            max_requests=max_requests,
            max_context=max_context,
            provider=provider,
            **model_options,
            **({"phase": purpose} if purpose is not None else {}),
        )
        from akas.models.definition import Composition

        if isinstance(composed, Composition):
            graph, commit = composed.graph, composed.commit
        else:
            graph, commit = (
                composed if isinstance(composed, tuple) else (composed, None)
            )
        from akas.compilation.policy import lower

        graph, commit = lower(
            graph, commit, policy=policy, provider=provider, target=target
        )
        bindings = graph.request or {}
        phase_inputs = inputs if inputs is not None else bindings.get("inputs")
        phase_prediction = (
            prediction if prediction is not None else bindings.get("prediction")
        )
        if phase_inputs is None or phase_prediction is None:
            raise ValueError("program graph needs request bindings")
        feedback = False
        if phase == "decode":
            from akas.compilation.rewrites import token_feedback

            feedback = token_feedback(graph, phase_inputs, phase_prediction, rows)
        resets.update(bindings.get("reset", ()))
        if graph.phase == "forward":
            graph.phase = "decode" if chunk == 1 else "prefill"
        for name, spec in graph.buffers.items():
            if spec.role == "weight" and spec.alias_of is None:
                previous = weight_specs.setdefault(name, spec)
                if (previous.shape, previous.dtype) != (spec.shape, spec.dtype):
                    raise ValueError("weight layout differs across program geometries")
            if spec.role == "state" and spec.alias_of is None:
                previous = state_specs.setdefault(name, spec)
                if (previous.shape, previous.dtype) != (spec.shape, spec.dtype):
                    raise ValueError("state layout differs across program geometries")
        filename = f"{generation}/phases/{phase}-b{rows}-t{chunk}.json"
        bundle = plan(
            graph,
            provider=provider,
            target=target,
            filename=filename,
            schedules=schedules,
            catalog=catalog,
        )
        bundle.source_directory = (
            f"{generation}/phases/{phase}-b{rows}-t{chunk}/sources"
        )
        bundle.artifact_directory = (
            f"{generation}/phases/{phase}-b{rows}-t{chunk}/kernels"
        )
        built = build(
            bundle, out, search=search, search_settings=search_settings, device=device
        )
        record = dict(
            phase=phase,
            rows=rows,
            chunk=chunk,
            path=filename,
            sha256=checksum(out / filename),
            inputs=dict(phase_inputs),
            prediction=phase_prediction,
            feedback=feedback,
        )
        if bindings.get("features"):
            record["features"] = bindings["features"]
        if commit is not None and commit.calls:
            used = {
                name
                for call in commit.calls
                for name in (*call.arguments, *call.bindings.values())
            }
            # Row-local state kernels acquire mapping during plan(), so preserve
            # that declared input even before its explicit call binding exists.
            if "mapping" in commit.buffers:
                used.add("mapping")
            commit.buffers = {
                name: spec for name, spec in commit.buffers.items() if name in used
            }
            commit.phase = "commit"
            commit_file = f"{generation}/phases/commit-b{rows}-t{chunk}.json"
            commit_bundle = plan(
                commit,
                provider=provider,
                target=built["target"],
                filename=commit_file,
                catalog=catalog,
            )
            commit_bundle.source_directory = (
                f"{generation}/phases/commit-b{rows}-t{chunk}/sources"
            )
            commit_bundle.artifact_directory = (
                f"{generation}/phases/commit-b{rows}-t{chunk}/kernels"
            )
            build(
                commit_bundle,
                out,
                search=search,
                search_settings=search_settings,
                device=device,
            )
            record["commit"] = dict(
                path=commit_file, sha256=checksum(out / commit_file)
            )
        records.append(record)
        if phase == "prefill":
            from akas.compilation.buffers import scratch_slots
            from akas.compilation.liveness import state_prefill
            from akas.graph import Graph

            state_graph = state_prefill(Graph.from_metadata(built["forward"]))
            if state_graph is not None:
                state_file = f"{generation}/phases/prefill-state-b{rows}-t{chunk}.json"
                state_manifest = {
                    **built,
                    "forward": state_graph.metadata(),
                    "allocations": scratch_slots(state_graph),
                    "kernels": {
                        key: value
                        for key, value in built["kernels"].items()
                        if key in state_graph.requirements()
                    },
                }
                state_manifest["tuning"] = {
                    **built.get("tuning", {}),
                    "inherited_from": filename,
                }
                publish(out / state_file, state_manifest)
                records.append(
                    dict(
                        phase="prefill_state",
                        rows=rows,
                        chunk=chunk,
                        path=state_file,
                        sha256=checksum(out / state_file),
                        inputs=dict(phase_inputs),
                        prediction=phase_prediction,
                        state_only=True,
                    )
                )
        target = built["target"]
    if not records:
        raise ValueError("program needs at least one geometry")
    if deferred:
        import json

        from akas.compilation.rewrites import deferred_state
        from akas.graph import Graph

        for record in records:
            if record["phase"] != "verify":
                continue
            original = Graph.from_metadata(
                json.loads((out / record["path"]).read_text())["forward"]
            )
            variant = deferred_state(original)
            if variant is None:
                continue
            verification, commit, materialize, clear = variant
            descriptor = {}
            for name, graph in (
                ("verify", verification),
                ("commit", commit),
                ("materialize", materialize),
            ):
                filename = f"{generation}/journals/{name}-b{record['rows']}-w{record['chunk']}.json"
                bundle = plan(
                    graph,
                    provider=provider,
                    target=target,
                    filename=filename,
                    catalog=catalog,
                )
                bundle.source_directory = f"{generation}/journals/{name}-b{record['rows']}-w{record['chunk']}/sources"
                bundle.artifact_directory = f"{generation}/journals/{name}-b{record['rows']}-w{record['chunk']}/kernels"
                build(
                    bundle,
                    out,
                    search=search,
                    search_settings=search_settings,
                    device=device,
                )
                descriptor[name] = dict(path=filename, sha256=checksum(out / filename))
            record["journal"] = descriptor
            resets.update(clear)
            for name, spec in verification.buffers.items():
                if spec.role == "state":
                    previous = state_specs.setdefault(name, spec)
                    if (previous.shape, previous.dtype) != (spec.shape, spec.dtype):
                        raise ValueError("journal state differs across geometries")
    reset_record = None
    if resets:
        from math import prod

        from akas.graph import Graph

        reset_graph = Graph(contract="state", phase="reset")
        mapping = reset_graph.buffer("mapping", (1,), "int32", role="input")
        for name in sorted(resets):
            spec = state_specs[name]
            if spec.shape[0] != max_requests:
                raise ValueError("reset state needs a request dimension")
            state = reset_graph.buffer(name, spec.shape, spec.dtype, role="state")
            width = prod(spec.shape[1:])
            rows = reset_graph.view(name + ".rows", state, (max_requests, width))
            reset_graph(
                "clear_rows",
                dict(rows=1, pool=max_requests, width=width, dtype=spec.dtype),
                mapping,
                rows,
            )
        filename = f"{generation}/phases/reset.json"
        reset_bundle = plan(
            reset_graph,
            provider=provider,
            target=target,
            filename=filename,
            catalog=catalog,
        )
        reset_bundle.source_directory = f"{generation}/phases/reset/sources"
        reset_bundle.artifact_directory = f"{generation}/phases/reset/kernels"
        build(
            reset_bundle,
            out,
            search=search,
            search_settings=search_settings,
            device=device,
        )
        reset_record = dict(
            path=filename, sha256=checksum(out / filename), mapping="mapping"
        )
    rounds = []
    for window in round_windows:
        from math import prod

        from akas.compilation.phases import round_controls
        from akas.graph import Graph

        if type(window) is not int or window < 2:
            raise ValueError("invalid speculative round window")
        for phase in records:
            if phase["phase"] != "verify" or phase["chunk"] != window:
                continue
            import json

            verification = Graph.from_metadata(
                json.loads((out / phase["path"]).read_text())["forward"]
            )
            feature = verification.buffers[phase["features"]["hidden"]]
            width = prod(feature.shape) // (phase["rows"] * window)
            controls = round_controls(
                slots=phase["rows"],
                window=window,
                width=width,
                dtype=feature.dtype,
            )
            filename = f"{generation}/rounds/control-b{phase['rows']}-w{window}.json"
            control_bundle = plan(
                controls,
                provider=provider,
                target=target,
                filename=filename,
                catalog=catalog,
            )
            control_bundle.source_directory = (
                f"{generation}/rounds/b{phase['rows']}-w{window}/sources"
            )
            control_bundle.artifact_directory = (
                f"{generation}/rounds/b{phase['rows']}-w{window}/kernels"
            )
            build(
                control_bundle,
                out,
                search=search,
                search_settings=search_settings,
                device=device,
            )
            rounds.append(
                dict(
                    rows=phase["rows"],
                    window=window,
                    path=filename,
                    sha256=checksum(out / filename),
                )
            )
    if upload_options is None and hasattr(checkpoint, "program_upload_options"):
        upload_options = checkpoint.program_upload_options(
            weight_specs, model_options=model_options
        )
    manifest = dict(
        schema="akas.program.v1",
        provider=provider,
        target=target,
        config=asdict(checkpoint.config),
        max_requests=max_requests,
        max_context=max_context,
        phases=records,
        reset=reset_record,
        rounds=rounds,
        upload_options=upload_options or {},
        model_options=model_options,
        policy=dict(policy or {}),
    )
    from akas.artifacts.read import validate_program

    validate_program(out, manifest)
    publish(out / "program.json", manifest)
    return manifest
