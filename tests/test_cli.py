"""The command surface accepts caller-supplied model definitions."""

from dataclasses import dataclass
import json
import sys
import pytest
from akas.cli import main


@dataclass
class Config:
    width: int = 7


class ForeignCheckpoint:
    def __init__(self, path, *, marker=None):
        self.config = Config()
        self.marker = marker

    def inventory(self):
        return dict(marker=self.marker)


class ForeignModel:
    checkpoint_type = ForeignCheckpoint
    opened = None

    def __init__(self, checkpoint, bundle, device, *, scale=1):
        type(self).opened = (checkpoint, bundle, device, scale)

    def generate(self, prompt, *, max_tokens, chat):
        return dict(text=f"{prompt}:{max_tokens}:{chat}")

    def __enter__(self):
        return self

    def __exit__(self, *error):
        pass


def test_compile_defaults_to_common_program_with_model_composition(monkeypatch, tmp_path):
    from akas.compilation import compile_checkpoint

    compose = object()
    ForeignModel.compose = compose
    calls = []
    monkeypatch.setattr("akas.compilation.import_module", lambda name: type("Compiler", (), {
        "compile_program": staticmethod(lambda loaded, out, **options: calls.append((loaded, out, options)))
    }))
    compile_checkpoint(ForeignModel, "weights", tmp_path, geometries=[(1, 1)],
                       max_requests=4, max_context=32)
    assert len(calls) == 1
    assert calls[0][2] == dict(definition=compose, geometries=[(1, 1)], max_requests=4, max_context=32)


def test_definition_opens_shared_executor_and_rejects_capacity_override(monkeypatch):
    import pytest
    from akas.models.definition import Definition

    class ForeignDefinition(Definition):
        checkpoint_type = ForeignCheckpoint

    monkeypatch.setattr("akas.artifacts.read.read_program", lambda bundle: dict(max_requests=4, max_context=32))
    result = object()
    calls = []
    monkeypatch.setattr("akas.models.definition.Executor.from_program",
                        lambda loaded, bundle, device, **options: calls.append((loaded, bundle, device, options)) or result)
    assert ForeignDefinition("weights", "bundle", "device", graphs=False) is result
    assert calls[0][3] == dict(graphs=False)
    with pytest.raises(ValueError, match="max_requests"):
        ForeignDefinition("weights", "bundle", "device", capacity=8)
    assert len(calls) == 1


def test_inspect_uses_the_supplied_checkpoint_reader(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "akas",
            "inspect",
            "--definition",
            __name__ + ":ForeignModel",
            "--model",
            "foreign",
            "--checkpoint-options",
            '{"marker":"custom"}',
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out) == dict(
        config=dict(width=7), inventory=dict(marker="custom")
    )


def test_generate_routes_constructor_options_without_model_dispatch(
    monkeypatch, capsys
):
    import tensor

    class Device:
        def __enter__(self):
            return self

        def __exit__(self, *error):
            pass

    device = Device()
    monkeypatch.setattr(tensor, "Device", lambda *args, **kwargs: device)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "akas",
            "generate",
            "--definition",
            __name__ + ":ForeignModel",
            "--model",
            "foreign",
            "--bundle",
            "compiled",
            "--prompt",
            "hello",
            "--max-tokens",
            "3",
            "--raw",
            "--config",
            '{"scale":5}',
        ],
    )
    main()
    assert capsys.readouterr().out.strip() == "hello:3:False"
    assert ForeignModel.opened[2:] == (device, 5)


def test_benchmark_compiles_one_program_bank_for_all_requested_batches(monkeypatch, tmp_path):
    import tensor
    from types import SimpleNamespace
    from akas import cli

    compiled, opened = [], []
    class Device:
        info = {"arch": "sm_89"}
        def __init__(self, *args, **options):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *error):
            pass

    class Model:
        compilation_profile = "program"
        def __init__(self, checkpoint, bundle, device, **options):
            opened.append((bundle, options))
        def __enter__(self):
            return self
        def __exit__(self, *error):
            pass

    class Session:
        kv_dtype = "float16"
        def __init__(self, model, *, batch, chunk):
            self.prefill_chunk = chunk
        def __enter__(self):
            return self
        def __exit__(self, *error):
            pass

    checkpoint = tmp_path / "weights.json"
    checkpoint.write_text('{}')
    monkeypatch.setattr(tensor, "Device", Device)
    monkeypatch.setattr(cli, "_factory", lambda value: Model)
    monkeypatch.setattr("akas.benchmarking.session.Session", Session)
    monkeypatch.setattr("akas.compilation.compile_checkpoint",
                        lambda *args, **options: compiled.append((args, options)))
    monkeypatch.setattr(cli, "run", lambda session, case, **options:
                        dict(status="ok", case=case.metadata()))
    monkeypatch.setattr(cli, "table", lambda rows: "")
    monkeypatch.setattr(cli, "write_report", lambda out, metadata, results:
                        dict(metadata=metadata, results=results))
    monkeypatch.setattr(sys, "argv", ["akas", "bench", "--history", str(tmp_path / "history"), "--definition", "foreign:Model",
        "--checkpoint", str(checkpoint), "--bundles", str(tmp_path / "bank"),
        "--compile", "--batches", "1", "4", "--prompts", "4", "--generated", "2",
        "--prefill-chunk", "4", "--checkpoint-options", '{"tokenizer":false}',
        "--out", str(tmp_path / "report.json")])
    cli.main()
    assert len(compiled) == 1
    options = compiled[0][1]
    assert options["geometries"] == [(1, 1), (1, 4), (4, 1), (4, 4)]
    assert options["max_requests"] == 4 and options["max_context"] == 6
    assert options["checkpoint_options"] == {"tokenizer": False}
    assert len(opened) == 2 and opened[0][0] == opened[1][0]
    assert all(options["checkpoint_options"] == {"tokenizer": False} for _, options in opened)
    recorded = json.loads(next((tmp_path / "history/runs").glob("*/run.json")).read_text())
    assert recorded["status"] == "passed" and len(recorded["results"]) == 2


def test_benchmark_records_device_startup_failure(monkeypatch, tmp_path):
    import tensor
    from akas import cli

    class Model:
        pass

    def unavailable(*args, **options):
        raise RuntimeError("adapter unavailable")

    monkeypatch.setattr(tensor, "Device", unavailable)
    monkeypatch.setattr(cli, "_factory", lambda value: Model)
    monkeypatch.setattr(sys, "argv", ["akas", "bench", "--definition", "test:Model",
        "--checkpoint", str(tmp_path / "weights"), "--bundles", str(tmp_path / "bank"),
        "--batches", "1", "--prompts", "32", "--out", str(tmp_path / "report.json"),
        "--history", str(tmp_path / "history")])
    with pytest.raises(RuntimeError, match="adapter unavailable"):
        cli.main()
    recorded = json.loads(next((tmp_path / "history/runs").glob("*/run.json")).read_text())
    assert recorded["status"] == "failed"
    assert recorded["results"][0]["error"] == "adapter unavailable"


def test_checkpoint_file_identity_depends_on_bytes_not_filename(tmp_path):
    import hashlib
    from akas.cli import checkpoint_digest

    first, renamed = tmp_path / "original.gguf", tmp_path / "renamed.gguf"
    first.write_bytes(b"same checkpoint bytes")
    renamed.write_bytes(first.read_bytes())
    assert checkpoint_digest(first) == checkpoint_digest(renamed)
    assert checkpoint_digest(first) == hashlib.sha256(first.read_bytes()).hexdigest()
