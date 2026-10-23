"""Load a selected deployment without invoking compilation or search."""

from contextlib import contextmanager
import json
from pathlib import Path

from akas.benchmarking.history import digest
from akas.optimization.context import symbol


def load(path, *, candidate=False):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "akas.deployment.v1":
        raise ValueError("unsupported deployment manifest")
    if manifest.get("status") != "qualified" and not (
        candidate and manifest.get("status") == "candidate"
    ):
        raise ValueError("deployment has not passed qualification")
    if manifest.get("runtime_source") != runtime_source():
        raise ValueError("deployment runtime source changed; optimize again")
    from akas.optimization.planning import software

    installed = software()
    if any(
        installed.get(name) != value
        for name, value in manifest.get("runtime_packages", {}).items()
    ):
        raise ValueError("deployment runtime dependencies changed; optimize again")
    for entry in manifest["artifacts"]:
        artifact = (path.parent / entry["path"]).resolve()
        if digest(artifact) != entry["sha256"]:
            raise ValueError("deployment artifact checksum mismatch: " + str(artifact))
    return manifest


@contextmanager
def service(path, *, candidate=False):
    import tensor
    from akas.config import EngineConfig
    from akas.engine import Engine
    from akas.serving.http import Service

    path = Path(path).resolve()
    manifest = load(path, candidate=candidate)
    context, recipe = manifest["context"], manifest["recipe"]
    checkpoint = Path(context["checkpoint"])
    from akas.cli import checkpoint_digest

    if checkpoint_digest(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("deployment checkpoint bytes have changed")
    program = (path.parent / manifest["program"]).resolve()
    model = symbol(context["definition"])
    with tensor.Device(
        context["device"],
        provider=recipe["provider"],
        max_buffer_size=manifest.get("max_buffer_size"),
    ) as device:
        if device.info["name"] != manifest["hardware"]["name"]:
            raise ValueError(
                "deployment was qualified on different hardware; optimize again"
            )
        with model(checkpoint, program, device, graphs=recipe["graphs"]) as target:
            target.prefill_group_size = recipe["prefill_requests"]
            backend = target
            if recipe["window"]:
                from akas.generation.speculative.decoding import SpeculativeDecoder

                options = dict(recipe.get("head_options", {}))
                for name in ("checkpoint", "bundle"):
                    if name in options:
                        options[name] = str((path.parent / options[name]).resolve())
                backend = SpeculativeDecoder(
                    target,
                    symbol(recipe["head"])(target, **options),
                    window=recipe["window"],
                    capture=recipe.get("round_capture", recipe["graphs"]),
                    deferred=recipe["deferred"],
                )
            with Engine(
                backend,
                config=EngineConfig(
                    prefill_chunk=recipe["chunk"],
                    prefill_requests=recipe["prefill_requests"],
                ),
            ) as engine:
                tokenizer = getattr(target.checkpoint, "tokenizer", None)
                yield Service(
                    engine,
                    None if tokenizer is None else tokenizer.encode,
                    dict(
                        model=context["definition"],
                        optimization=manifest["campaign_identity"],
                        run_id=manifest["run_id"],
                        provider=recipe["provider"],
                        workload=context["workload"],
                    ),
                )


def runtime_source():
    from akas.optimization.context import identity

    root = Path(__file__).resolve().parents[1]
    return identity(
        {
            str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*.py"))
            if ".ipynb_checkpoints" not in str(p)
        }
    )


def describe(
    context,
    recipe,
    program,
    *,
    checkpoint_sha256,
    campaign_identity,
    run_id,
    max_buffer_size=None,
    hardware=None,
):
    """Create a private candidate descriptor; publication follows the HTTP check."""
    from akas.artifacts.read import read_program

    program = Path(program).resolve()
    manifest = read_program(program)
    files = {program / "program.json"}

    def visit(value, root=program):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                path = (root / value["path"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("deployment dependency escapes program")
                if digest(path) != value["sha256"]:
                    raise ValueError("deployment dependency checksum mismatch")
                if path not in files:
                    files.add(path)
                    if path.suffix == ".json":
                        visit(json.loads(path.read_text()), root)
            for child in value.values():
                visit(child, root)
        elif isinstance(value, list):
            for child in value:
                visit(child, root)

    visit(manifest)
    # Paths are relative to the deployment directory, which is the program's parent.
    import os
    from akas.optimization.planning import software

    base = program.parent
    recipe = dict(recipe)
    if recipe.get("head_options"):
        options = dict(recipe["head_options"])
        if "bundle" in options:
            draft_program = Path(options["bundle"]).resolve()
            draft_manifest = read_program(draft_program)
            files.add(draft_program / "program.json")
            visit(draft_manifest, draft_program)
            options["program_sha256"] = digest(draft_program / "program.json")
            options["bundle"] = os.path.relpath(draft_program, base)
        if "checkpoint" in options:
            draft_checkpoint = Path(options["checkpoint"]).resolve()
            options["checkpoint_sha256"] = digest(draft_checkpoint)
            options["checkpoint"] = os.path.relpath(draft_checkpoint, base)
        recipe["head_options"] = options
    return dict(
        schema="akas.deployment.v1",
        status="candidate",
        context=context,
        recipe=recipe,
        checkpoint_sha256=checkpoint_sha256,
        campaign_identity=campaign_identity,
        run_id=run_id,
        program=os.path.relpath(program, base),
        max_buffer_size=max_buffer_size,
        hardware=hardware,
        runtime_source=runtime_source(),
        runtime_packages={
            k: v
            for k, v in software().items()
            if k in ("tensor-workspace", "numpy")
            or (k == "wgpu" and recipe["provider"] == "webgpu")
        },
        artifacts=[
            dict(path=os.path.relpath(file, base), sha256=digest(file))
            for file in sorted(files)
        ],
    )
