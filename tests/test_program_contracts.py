"""Program publication preserves phase artifacts and operation storage contracts."""

from types import SimpleNamespace

import pytest

from akas.graph import Graph
from akas.compilation.validate import kernel_bindings


def test_candidate_cannot_change_weight_precision_or_output_layout():
    graph = Graph(contract="reference")
    x = graph.buffer("x", (8,), "float32", role="input")
    weight = graph.buffer("weight", (32,), "float16", role="weight")
    out = graph.buffer("out", (4,), "float32", role="output")
    call = graph("linear", dict(r=1, k=8, o=4), x, weight, out)
    signature = [dict(name=name, shape=list(spec.shape), dtype=spec.dtype)
                 for name, spec in graph.buffers.items()]
    kernel_bindings(graph, call.key, signature)
    wrong = [dict(item) for item in signature]
    wrong[1]["dtype"] = "float32"
    with pytest.raises(ValueError, match="storage contract for weight"):
        kernel_bindings(graph, call.key, wrong)
    wrong = [dict(item) for item in signature]
    wrong[2]["shape"] = [1, 4]
    with pytest.raises(ValueError, match="storage contract for out"):
        kernel_bindings(graph, call.key, wrong)


def test_decode_feedback_advances_input_positions_without_duplicating_state_updates():
    from akas.compilation.rewrites import token_feedback

    graph = Graph(contract="foreign")
    for name in ("tokens", "lengths", "positions"):
        graph.buffer(name, (4,), "int32", role="input")
    graph.buffer("prediction", (4,), "int32", role="output")
    inputs = {name: name for name in ("tokens", "lengths", "positions")}
    assert token_feedback(graph, inputs, "prediction", 4)
    assert graph.calls[-1].parameters == dict(rows=4, advance=True)
    assert graph.calls[-1].arguments == ("prediction", "tokens", "lengths", "positions")

    other = Graph(contract="foreign")
    tokens = other.buffer("tokens", (4,), "int32", role="input")
    other.buffer("lengths", (4,), "int32", role="input")
    assert token_feedback(other, {"tokens": "tokens", "lengths": "lengths"}, tokens.name, 4)
    assert not other.calls


def test_resident_feedback_cannot_continue_from_a_reused_request_slot():
    import numpy as np
    from akas.execution.executor import Executor

    launched = []
    frame = SimpleNamespace(feedback=True, slots=(0,), expected_positions=np.array([0]),
                            expected_generations=np.array([1], dtype="uint64"),
                            forward=SimpleNamespace(run=lambda: launched.append(True)))
    bank = SimpleNamespace(select=lambda *args, **kwargs: frame)
    executor = Executor(capacity=2, context=32, batches=bank, reset=lambda slot: None)
    executor.config = SimpleNamespace(vocab=8)
    executor.admit_at(0)
    executor.release(0)
    executor.admit_at(0)
    with pytest.raises(ValueError, match="current request state"):
        executor.decode_steps([0], None, 1, read=False)
    assert not launched


def test_workspace_rejects_overlapping_scratch_through_views():
    from akas.execution.workspace import Workspace

    graph = Graph(contract="test")
    first = graph.buffer("first", (4,), "float32")
    second = graph.buffer("second", (4,), "float32")
    view = first.view((2, 2))
    graph("copy", {}, view, second)
    with pytest.raises(ValueError, match="overlapping lifetimes"):
        Workspace(graph, object(), {}, [["first", "second"]])


def test_program_keeps_each_phase_kernel_directory(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from akas.compilation import compile as coordinator
    from akas.artifacts.schema import Bundle
    from akas.artifacts.write import publish

    @dataclass
    class Config:
        vocab: int = 8

    checkpoint = SimpleNamespace(config=Config())
    directories = []
    device = object()

    def definition(checkpoint, **geometry):
        graph = Graph(contract="test")
        graph.buffer("tokens", (geometry["rows"] * geometry["chunk"],), "int32", role="input")
        graph.buffer("lengths", (geometry["rows"],), "int32", role="input")
        graph.buffer("sampled", (geometry["rows"],), "int32", role="output")
        return graph.bind_requests(inputs={"tokens": "tokens", "lengths": "lengths"}, prediction="sampled")

    def build(bundle, out, **options):
        assert options["device"] is device
        directories.append((bundle.source_directory, bundle.artifact_directory))
        publish(tmp_path / bundle.filename, bundle.manifest)
        return bundle.manifest

    monkeypatch.setattr(coordinator, "plan", lambda graph, **options: Bundle(
        dict(target="sm_89", provider="cuda", forward=graph.metadata()), options["filename"]))
    monkeypatch.setattr(coordinator, "build", build)
    result = coordinator.compile_program(checkpoint, tmp_path, definition=definition,
                                         geometries=[(2, 1), (2, 4)], max_requests=4, max_context=32, device=device)
    assert len(set(directories)) == 2
    assert result["phases"][0]["inputs"] == result["phases"][1]["inputs"]
    with pytest.raises(ValueError, match="optional phase"):
        coordinator.compile_program(checkpoint, tmp_path, definition=definition,
                                    geometries=[(2, 1, "decode", "extra")], max_requests=4, max_context=32)


def test_failed_program_replacement_preserves_published_phase_bytes(tmp_path, monkeypatch):
    from dataclasses import dataclass
    import json
    from akas.compilation import compile as coordinator
    from akas.artifacts.schema import Bundle
    from akas.artifacts.write import publish

    @dataclass
    class Config:
        vocab: int = 8

    def definition(checkpoint, **geometry):
        graph = Graph(contract="test")
        graph.buffer("tokens", (geometry["rows"] * geometry["chunk"],), "int32", role="input")
        graph.buffer("lengths", (geometry["rows"],), "int32", role="input")
        graph.buffer("sampled", (geometry["rows"],), "int32", role="output")
        return graph.bind_requests(inputs={"tokens": "tokens", "lengths": "lengths"}, prediction="sampled")

    monkeypatch.setattr(coordinator, "plan", lambda graph, **options: Bundle(
        dict(target="sm_89", provider="cuda", forward=graph.metadata(), marker="original"), options["filename"]))
    def build(bundle, out, **options):
        publish(out / bundle.filename, bundle.manifest)
        return bundle.manifest
    monkeypatch.setattr(coordinator, "build", build)
    options = dict(definition=definition, geometries=[(2, 1), (2, 4)], max_requests=4, max_context=32)
    coordinator.compile_program(SimpleNamespace(config=Config()), tmp_path, **options)
    previous = (tmp_path / "program.json").read_bytes()
    phases = {row["path"]: (tmp_path / row["path"]).read_bytes()
              for row in json.loads(previous)["phases"]}
    calls = []
    def fail_after_first_phase(bundle, out, **options):
        calls.append(bundle.filename)
        bundle.manifest["marker"] = "replacement"
        if len(calls) == 2:
            raise RuntimeError("compiler failed")
        return build(bundle, out, **options)
    monkeypatch.setattr(coordinator, "build", fail_after_first_phase)
    with pytest.raises(RuntimeError, match="compiler failed"):
        coordinator.compile_program(SimpleNamespace(config=Config()), tmp_path, **options)
    assert (tmp_path / "program.json").read_bytes() == previous
    assert all((tmp_path / path).read_bytes() == data for path, data in phases.items())
    assert not set(calls) & phases.keys()

    # A compiler may return successfully yet emit a manifest the loader rejects.
    # Publication must preserve the working generation in that case as well.
    def invalid_build(bundle, out, **options):
        bundle.manifest["provider"] = "wrong-provider"
        return build(bundle, out, **options)
    monkeypatch.setattr(coordinator, "build", invalid_build)
    with pytest.raises(ValueError, match="target/provider mismatch"):
        coordinator.compile_program(SimpleNamespace(config=Config()), tmp_path, **options)
    assert (tmp_path / "program.json").read_bytes() == previous
    assert all((tmp_path / path).read_bytes() == data for path, data in phases.items())


def test_kernel_cache_tracks_codec_changes_without_unrelated_executor_changes(tmp_path, monkeypatch):
    from pathlib import Path
    from tensor.compiler.frontend import source as serializer
    from akas.artifacts.schema import Source
    from akas.compilation.emit import source_dependencies

    name = "akas.ops.attention.cuda.floating"
    dependencies = source_dependencies(name)
    assert "akas.quantization.kv.fp8" in dependencies
    assert not any(module.startswith("akas.execution.") for module in dependencies)
    declaration = Source(name, "make_kernel", ("attention", dict(kv_dtype="fp8")), dependencies=())
    baseline = declaration.render()
    original = serializer.find_spec
    workspace = tmp_path / "workspace.py"
    workspace.write_text("# unrelated executor edit\n")
    codec = tmp_path / "fp8.py"
    codec.write_text(Path(original("akas.quantization.kv.fp8").origin).read_text())

    def locate(module):
        if module == "akas.execution.workspace":
            return SimpleNamespace(origin=str(workspace))
        if module == "akas.quantization.kv.fp8":
            return SimpleNamespace(origin=str(codec))
        return original(module)

    monkeypatch.setattr(serializer, "find_spec", locate)
    assert declaration.render() == baseline
    workspace.write_text("# another executor edit\n")
    assert declaration.render() == baseline
    codec.write_text(codec.read_text() + "\n# changed codec implementation\n")
    assert declaration.render() != baseline


def test_control_uploads_validate_together_and_share_one_cuda_barrier():
    import numpy as np
    from akas.memory.transfer import write_many

    calls = []
    device = SimpleNamespace(info={"provider":"cuda"}, synchronize=lambda:calls.append("barrier"),
                             driver=SimpleNamespace(call=lambda name,*arguments:calls.append(name)))
    def buffer(size):
        return SimpleNamespace(shape=(size,), dtype="int32", device=device, pointer=1, _check=lambda:None)
    first, second = buffer(2), buffer(3)
    write_many(device, [(first, np.zeros(2,dtype="int32")), (second,np.zeros(3,dtype="int32"))])
    assert calls == ["barrier", "cuMemcpyHtoD_v2", "cuMemcpyHtoD_v2", "cuStreamSynchronize"]
    calls.clear()
    with pytest.raises(ValueError, match="shape mismatch"):
        write_many(device, [(first,np.zeros(2,dtype="int32")), (second,np.zeros(2,dtype="int32"))])
    assert calls == []


def test_round_controls_expose_phase_fields_without_a_model_name():
    from akas.compilation.phases import round_controls

    graph = round_controls(slots=2,window=4,width=128)
    assert graph.contract == "speculation"
    assert [call.label for call in graph.calls] == ["setup", "draft.1", "proposal.1", "draft.2", "proposal.2", "accept", "repair", "result"]
    short = round_controls(slots=2,window=2,width=128)
    assert not any(name.startswith("draft.") for name in short.buffers)
    assert graph.buffers["verify.hidden"].shape == (8,128)


def test_commit_retains_mapping_until_row_local_state_lowering(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from akas.compilation import compile as coordinator
    from akas.artifacts.schema import Bundle
    from akas.artifacts.write import publish

    @dataclass
    class Config:
        vocab: int = 8

    def definition(checkpoint, **geometry):
        graph = Graph(contract="test", phase="verify")
        for name, shape, role in (("tokens", (2,), "input"), ("lengths", (2,), "input"),
                                  ("mapping", (2,), "input"), ("sampled", (2,), "output"),
                                  ("state", (4, 3), "state")):
            graph.buffer(name, shape, "int32", role=role)
        graph.bind_requests(inputs={"tokens": "tokens", "lengths": "lengths", "mapping": "mapping"},
                            prediction="sampled")
        commit = Graph(contract="test", phase="commit")
        commit.buffers.update(graph.buffers)
        commit("restore", {"r": 2}, "state")
        return graph, commit

    phases = []
    def plan(graph, **options):
        phases.append(graph.phase)
        if graph.phase == "commit":
            assert "mapping" in graph.buffers
        return Bundle(dict(target="sm_89", provider="cuda", forward=graph.metadata()), options["filename"])

    def build(bundle, out, **options):
        publish(out / bundle.filename, bundle.manifest)
        return bundle.manifest

    monkeypatch.setattr(coordinator, "plan", plan)
    monkeypatch.setattr(coordinator, "build", build)
    coordinator.compile_program(SimpleNamespace(config=Config()), tmp_path, definition=definition,
                                geometries=[(2, 1, "verify")], max_requests=4, max_context=16)
    assert phases == ["verify", "commit"]
