"""The command surface accepts caller-supplied model definitions."""

from dataclasses import dataclass
import json
import sys
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
