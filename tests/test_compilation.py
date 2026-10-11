"""Frozen kernel selection, common cache invalidation and complete publication."""

from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from akas.compilation import compile_checkpoint, PROFILES
from akas.compilation.build import build, checksum
from akas.compilation.bundles import copy
from akas.compilation.plan import Bundle, Kernel
from akas.models.lfm2.weights import LFM2Checkpoint
from akas.models.qwen35.config import DenseConfig, Qwen35Config

CASES = json.loads(
    (Path(__file__).parent / "fixtures/compilation-profiles.json").read_text()
)


@pytest.fixture(scope="module")
def packed_checkpoint(tmp_path_factory):
    from benchmarks.lfm2.diagnostic import fixture

    return LFM2Checkpoint(
        fixture(tmp_path_factory.mktemp("compilation") / "model.gguf")
    )


@pytest.mark.parametrize("case", CASES)
def test_profiles_preserve_original_kernel_sources_and_specializations(
    case, packed_checkpoint, monkeypatch
):
    def trace(module, factory, *args, dependencies=()):
        return json.dumps(
            dict(module=module, factory=factory, args=args, dependencies=dependencies)
        )

    monkeypatch.setattr("tensor.compiler.entry.export_source", trace)
    module = importlib.import_module(case["profile"])
    if case["profile"].endswith("packed"):
        checkpoint = packed_checkpoint
    else:
        cls = DenseConfig if case["profile"].endswith("batched") else Qwen35Config
        values = {**case["config"], "layers": tuple(case["config"]["layers"])}
        checkpoint = SimpleNamespace(config=cls(**values))
    plan = module.plan(checkpoint, **case["options"])
    actual = [
        dict(
            key=k.key,
            kind=k.kind,
            parameters=k.parameters,
            source=json.loads(k.source.render()),
        )
        for k in plan.kernels
    ]
    assert actual == case["kernels"]


def test_batched_profile_reuses_different_widths_and_vocabularies():
    from akas.compilation.profiles.batched import plan, recurrent_plan

    config = DenseConfig(
        width=512,
        intermediate=1024,
        vocab=4096,
        layers=("linear_attention", "full_attention"),
        heads=8,
        kv_heads=2,
        head_dim=64,
        key_heads=4,
        value_heads=8,
        key_dim=64,
        value_dim=64,
        epsilon=1e-6,
        theta=10000,
        rotary_dim=32,
        context_limit=1024,
    )
    bundle = plan(SimpleNamespace(config=config), slots=2, chunk=4, pool=8, context=512)
    embedding = next(k for k in bundle.kernels if k.kind == "embedding")
    assert embedding.parameters == dict(r=8, c=512, vocab=4096)
    recurrent = recurrent_plan(config, slots=2, window=4, pool=8)
    conv = next(k for k in recurrent.kernels if k.kind == "conv_commit")
    assert conv.parameters["channels"] == 1024 and conv.parameters["projection"] == 1552
    assert (
        next(k for k in recurrent.kernels if k.kind == "gdn_deferred").parameters[
            "heads"
        ]
        == 8
    )


def test_fixed_layout_profile_rejects_incompatible_geometry():
    from akas.compilation.profiles.hybrid.decode import plan

    config = Qwen35Config(("full_attention",), 1024, 1e-6, 10000)
    with pytest.raises(ValueError, match="width=2048"):
        plan(SimpleNamespace(config=replace(config, width=1024)))


@pytest.fixture
def compiler(monkeypatch):
    import akas.compilation.build as pipeline

    calls = []
    version = ["compiler-v1"]
    monkeypatch.setattr(
        pipeline,
        "_compiler_identity",
        lambda provider: dict(lowering_sha256=version[0]),
    )
    monkeypatch.setattr(
        "tensor.artifacts.format.read_artifact",
        lambda path: (json.loads(Path(path).read_text()), None),
    )

    def compile(source, destination, **options):
        calls.append(options)
        Path(destination).write_text(
            json.dumps(
                dict(
                    target=options["target"],
                    provider=options["provider"],
                    compiler=dict(lowering_sha256=version[0]),
                    source=Path(source).read_text(),
                )
            )
        )

    monkeypatch.setattr("tensor.build", compile)
    return calls, version


def bundle():
    return Bundle(
        dict(schema="test", target="sm_89", status="building"),
        "inference.json",
        [
            Kernel("first", "test", {"width": 16}, "source one"),
            Kernel("second", "test", {"width": 32}, "source two"),
        ],
    )


def test_common_build_cache_checks_source_target_provider_and_compiler(
    tmp_path, compiler
):
    calls, version = compiler
    declaration = bundle()
    result = build(
        declaration,
        tmp_path,
        progress=lambda *_: (
            pytest.fail("partial manifest exposed")
            if (tmp_path / "inference.json").exists()
            else None
        ),
    )
    assert len(calls) == 2 and result["status"] == "built"
    assert result["kernels"]["first"]["sha256"] == checksum(tmp_path / "first.tbin")
    build(declaration, tmp_path)
    assert len(calls) == 2
    declaration.kernels[0] = replace(declaration.kernels[0], source="source changed")
    build(declaration, tmp_path)
    assert len(calls) == 3
    declaration.manifest["target"] = "sm_86"
    build(declaration, tmp_path)
    assert len(calls) == 5
    version[0] = "compiler-v2"
    build(declaration, tmp_path)
    assert len(calls) == 7
    declaration.provider = "webgpu"
    declaration.manifest["target"] = "webgpu-portable-v1"
    build(declaration, tmp_path)
    assert len(calls) == 9
    (tmp_path / "first.tbin").write_text("broken artifact")
    build(declaration, tmp_path)
    assert len(calls) == 10


def test_failed_build_never_publishes_complete_manifest(
    tmp_path, compiler, monkeypatch
):
    build(bundle(), tmp_path)
    declaration = bundle()
    declaration.kernels = [
        replace(k, source=k.source + " changed") for k in declaration.kernels
    ]
    import tensor

    compile = tensor.build

    def fail_second(source, destination, **options):
        if source.stem == "second":
            raise RuntimeError("injected compile failure")
        return compile(source, destination, **options)

    monkeypatch.setattr(tensor, "build", fail_second)
    with pytest.raises(RuntimeError, match="compile failure"):
        build(declaration, tmp_path)
    assert not (tmp_path / "inference.json").exists()
    assert not (tmp_path / "inference.json.tmp").exists()


def test_bundle_paths_and_duplicate_declarations_are_rejected(tmp_path, compiler):
    declaration = bundle()
    declaration.filename = "../escaped.json"
    with pytest.raises(ValueError, match="manifest path"):
        build(declaration, tmp_path / "out")
    declaration = bundle()
    declaration.kernels[0] = replace(declaration.kernels[0], key="../escape")
    with pytest.raises(ValueError, match="kernel paths"):
        build(declaration, tmp_path / "out")
    declaration = bundle()
    declaration.kernels.append(declaration.kernels[0])
    with pytest.raises(ValueError, match="duplicate"):
        build(declaration, tmp_path / "out")


def test_checked_bundle_copy_includes_nested_artifact_roles(tmp_path, compiler):
    source = tmp_path / "source"
    out = tmp_path / "derived"
    result = build(bundle(), source)
    result["kernels"]["first"]["logical"] = result["kernels"]["second"]
    (source / "inference.json").write_text(json.dumps(result))
    copied = copy(source, out, "inference.json")
    assert copied == result and (out / "second.tbin").is_file()
    assert not (out / "inference.json").exists()
    assert (out / "first.py").read_text() == "source one"
    (source / "second.tbin").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        copy(source, tmp_path / "bad", "inference.json")
    assert not (tmp_path / "bad").exists()


def test_bundle_copy_rejects_absolute_paths_even_inside_source(tmp_path, compiler):
    source = tmp_path / "source"
    result = build(bundle(), source)
    result["kernels"]["first"]["path"] = str((source / "first.tbin").resolve())
    (source / "inference.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="relative"):
        copy(source, tmp_path / "out", "inference.json")


def test_shared_entry_uses_supplied_checkpoint_reader_and_profile(
    tmp_path, monkeypatch
):
    calls = []
    definition = SimpleNamespace(
        checkpoint_type=lambda path, **options: calls.append((path, options))
        or SimpleNamespace(config="geometry")
    )
    module = SimpleNamespace(
        produce=lambda checkpoint, out, **options: (checkpoint.config, out, options)
    )
    monkeypatch.setitem(PROFILES, "test", ("test_profile", "produce"))
    monkeypatch.setattr("akas.compilation.import_module", lambda name: module)
    result = compile_checkpoint(
        definition,
        "checkpoint",
        tmp_path,
        profile="test",
        checkpoint_options={"branch": "draft"},
        rows=[1, 32],
    )
    assert calls == [("checkpoint", {"branch": "draft"})]
    assert result == ("geometry", tmp_path, {"rows": [1, 32]})
