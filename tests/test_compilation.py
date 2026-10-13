"""Frozen kernel selection, common cache invalidation and complete publication."""

from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from akas.compilation import compile_checkpoint
from akas.compilation import PROFILES
from akas.compilation.compile import build
from akas.artifacts.write import checksum
from akas.artifacts.write import copy
from akas.artifacts.schema import Bundle
from akas.artifacts.schema import Kernel
from akas.models.lfm2.weights import LFM2Checkpoint
from akas.models.qwen35.config import DenseConfig
from akas.models.qwen35.config import Qwen35Config

CASES = json.loads(
    (Path(__file__).parent / "fixtures/compilation-profiles.json").read_text()
)


@pytest.mark.parametrize("case",[case for case in CASES if case["profile"].endswith("packed")])
def test_frozen_packed_kernel_parameters_use_uniform_definitions(case):
    from akas.compilation.selection import source

    provider = case["options"].get("provider","cuda")
    for record in case["kernels"]:
        original = record["source"]
        kind,parameters = original["args"]
        contract = "packed-baseline" if original["factory"] == "packed_baseline_factory" else "packed"
        declaration = source(kind,parameters,contract=contract,provider=provider,
                             target=case["options"].get("target"))
        assert declaration.module == "akas.ops.definition"
        spec,schedule,capabilities,implementation = declaration.args
        assert spec["kind"] == kind
        assert spec["parameters"] == parameters
        assert capabilities["provider"] == provider
        assert implementation["contract"] == contract


@pytest.fixture(scope="module")
def packed_checkpoint(tmp_path_factory):
    from support.diagnostic import fixture

    return LFM2Checkpoint(
        fixture(tmp_path_factory.mktemp("compilation") / "model.gguf")
    )


# Retired MoE producer inventories are replaced by complete frozen forward
# contracts and independently qualified physical policies in test_qwen_graph
# and test_native_pipeline_policy; those fixtures remain unchanged.
@pytest.mark.parametrize("case", [case for case in CASES if case["profile"].endswith("bf16")])
def test_profiles_preserve_original_kernel_sources_and_specializations(
    case, packed_checkpoint, monkeypatch
):
    def trace(module, factory, *args, dependencies=()):
        return json.dumps(
            dict(module=module, factory=factory, args=args, dependencies=dependencies)
        )

    monkeypatch.setattr("tensor.compiler.frontend.source.export_source", trace)
    dense = case["profile"].endswith("bf16")
    if dense:
        from plan_fixture import dense_checkpoint
        from akas.models.qwen35.model import compose_dense
        from akas.compilation.compile import plan as graph_plan
        values = {**case["config"], "layers": tuple(case["config"]["layers"])}
        checkpoint = dense_checkpoint(DenseConfig(**values))
        options = dict(case["options"])
        target = options.pop("target", "sm_89")
        graph, commit = compose_dense(checkpoint, **options)
        plan = graph_plan(graph, target=target)
        if commit.calls:
            extra = graph_plan(commit, target=target)
            plan.kernels = list({kernel.key: kernel for kernel in [*plan.kernels, *extra.kernels]}.values())
    else:
        module = importlib.import_module(case["profile"])
        if case["profile"].endswith("packed"):
            checkpoint = packed_checkpoint
        else:
            values = {**case["config"], "layers": tuple(case["config"]["layers"])}
            checkpoint = SimpleNamespace(config=Qwen35Config(**values))
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
    # Only source ownership changed; frozen keys, parameters and schedule arguments
    # remain the pre-migration numerical contract.
    expected = json.loads(json.dumps(case["kernels"]))
    for record in expected:
        if isinstance(record["source"], dict):
            assert record["source"]["dependencies"] == ["$shared-operation-closure"]
            del record["source"]["dependencies"]
            if record["source"]["module"] == "akas.compilation.factories.block_fp8":
                from akas.ops.registry import select
                original = record["source"]
                kind, parameters = original["args"]
                variant = select(kind, parameters, contract="block-fp8", provider="cuda",
                                 phase=original["factory"].removesuffix("_factory"), target=None)
                original["module"] = variant.module_name
                original["factory"] = variant.factory
                original["args"] = json.loads(json.dumps(variant.source_arguments(parameters)))
            if record["source"]["module"].startswith("akas.ops.ffn."):
                from activation_parity import old_dispatch
                original = record["source"]
                kind, parameters = original["args"]
                if kind == "swiglu_experts":
                    parameters = {**parameters, "expert_rows":True, "output_dtype":"bfloat16", "round_inputs":True}
                else:
                    has_rows = "r" in parameters
                    parameters = old_dispatch(original["module"],kind,parameters)
                    if not has_rows:
                        parameters.pop("r", None)
                original["module"] = "akas.ops.activation.kernel"
                original["args"] = json.loads(json.dumps([kind,parameters]))
            elif record["source"]["module"].startswith("akas.ops.normalization.") and record["source"]["module"] != "akas.ops.normalization.kernel":
                from normalization_parity import migrated_parameters
                original = record["source"]
                kind, parameters = original["args"]
                original["args"] = [kind, migrated_parameters(original["module"], kind, parameters)]
                original["module"] = "akas.ops.normalization.kernel"
            elif record["source"]["module"].startswith("akas.ops.residual."):
                record["source"]["module"] = "akas.ops.residual.kernel"
            elif record["source"]["module"] == "akas.ops.quantization.cuda.block_fp8":
                record["source"]["module"] = "akas.quantization.activations.fp8"
    # The old placeholder described a package-wide cache hash, not arithmetic.
    # Import-scoped cache invalidation has its own behavioral tests below.
    for record in actual:
        dependencies = record["source"].pop("dependencies")
        assert dependencies == sorted(set(dependencies))
        assert "tensor.compiler.frontend.source" in dependencies
        if not record["source"]["module"].startswith("akas.compilation.factories."):
            assert not any(name.startswith("akas.execution.") for name in dependencies)
    # The canonical boundary adds routing data, while leaf specialization is
    # the unchanged numerical contract checked against the original fixture.
    for record in actual:
        source = record["source"]
        if source["module"] == "akas.ops.definition":
            assert source["factory"] == "define"
            spec, schedule, capabilities, choice = source["args"]
            assert choice["provider"] == capabilities["provider"]
            record["source"] = dict(module=choice["module"], factory=choice["factory"],
                                    args=[choice["kernel_kind"],
                                          {**choice["defaults"], **spec["parameters"], **schedule}])
    if dense:
        # The old inventory also emitted unreachable GDN, merge and fused-head
        # kernels. Only actual forward/commit calls enter the common compiler.
        reference = {item["key"]: item for item in expected}
        assert set(item["key"] for item in actual) <= reference.keys()
        assert {item["key"]: item for item in actual} == {item["key"]: reference[item["key"]] for item in actual}
    else:
        assert actual == expected


def test_batched_profile_reuses_different_widths_and_vocabularies():
    from akas.compilation.compile import plan
    from akas.compilation.rewrites import deferred_state
    from akas.models.qwen35.model import compose_dense
    from plan_fixture import dense_checkpoint

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
    checkpoint = dense_checkpoint(config)
    graph, _ = compose_dense(checkpoint, slots=2, chunk=4, pool=8, context=512)
    bundle = plan(graph)
    embedding = next(k for k in bundle.kernels if k.kind == "embedding")
    assert embedding.parameters == dict(r=8, c=512, vocab=4096)
    verification, _ = compose_dense(checkpoint, slots=2, chunk=4, pool=8, context=512, verify=True)
    deferred, commit, materialize, _ = deferred_state(verification)
    recurrent = plan(commit)
    recurrent.kernels.extend(plan(materialize).kernels)
    conv = next(k for k in recurrent.kernels if k.kind == "conv_commit")
    assert conv.parameters["channels"] == 1024 and conv.parameters["projection"] == 1552
    assert (
        next(k for k in recurrent.kernels if k.kind == "gdn_materialize").parameters[
            "heads"
        ]
        == 8
    )


def test_fixed_layout_profile_rejects_incompatible_geometry():
    from akas.models.qwen35.model import compose_moe_program
    from test_qwen_graph import moe_checkpoint

    checkpoint = moe_checkpoint()
    checkpoint.config = replace(checkpoint.config,width=1024)
    with pytest.raises(ValueError,match="width=2048"):
        compose_moe_program(checkpoint,rows=2,chunk=1,max_requests=4,max_context=64)


@pytest.fixture
def compiler(monkeypatch):
    import akas.compilation.emit as pipeline

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
        checkpoint_type=lambda path, **options: (
            calls.append((path, options)) or SimpleNamespace(config="geometry")
        )
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
