"""The common kernel boundary separates ABI, schedules and target routing."""

import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from akas.compilation.selection import selected_source
from akas.ops import Capabilities
from akas.ops import KernelSpec
from akas.ops import define
from akas.kernels.contracts import Implementation
from akas.kernels.definition import definition
from akas.kernels.registry import candidates


@pytest.mark.parametrize("kind", ["linear", "ffn"])
@pytest.mark.parametrize("encoding", [1, 2, 14])
def test_packed_prefill_search_can_explore_weight_reuse_without_changing_default(kind, encoding):
    from akas.kernels.registry import select

    parameters = dict(r=32, k=2048, o=512, ggml=True, type=encoding)
    options = dict(contract="reference", provider="webgpu", target="webgpu-portable-v1")
    choices = candidates(kind, parameters, **options)
    outer = next(choice for choice in choices if choice.variant == "outer-prefill")
    assert dict(outer.defaults)["outer_rows"] == 32
    assert select(kind, parameters, **options).variant != "outer-prefill"
    assert not any(choice.variant == "outer-prefill" for choice in
                   candidates(kind, dict(parameters, r=1), **options))


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_saved_profile_selects_nested_packed_weight_parameters(tmp_path, provider):
    from tensor.compiler.search import ScheduleProfile
    from akas.compilation.selection import kernels

    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    parameters = dict(
        r=33,
        top=2,
        experts=4,
        k=256,
        o=128,
        tile_map=True,
        input_dtype="float32",
        weight_parts=({"rows": 512, "format": "ggml", "options": {"encoding": 2}},),
    )
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            dict(
                schema="tensor.schedule-profile.v1",
                provider=provider,
                target=target,
                entries=[
                    dict(
                        operation="quantized_experts",
                        parameters=parameters,
                        schedule=dict(block_m=32, columns=64, block_k=64),
                    )
                ],
            )
        )
    )
    profile = ScheduleProfile(json.loads(path.read_text()))

    def selected(p):
        return kernels(
            {"experts": ("quantized_experts", p)},
            contract="reference",
            provider=provider,
            target=target,
            schedules=profile,
            record_implementation=True,
        )[0]

    declaration = selected(parameters)
    assert declaration.schedule["block_m"] == 32
    assert declaration.schedule["columns"] == declaration.schedule["block_k"] == 64
    assert declaration.source.args[1]["block_m"] == 32
    assert isinstance(parameters["weight_parts"], tuple)
    changed = {
        **parameters,
        "weight_parts": ({"rows": 512, "format": "ggml", "options": {"encoding": 8}},),
    }
    assert "block_m" not in selected(changed).schedule


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_same_definition_api_passes_only_resolved_leaf_arguments(provider, monkeypatch):
    choice = Implementation(
        "project",
        "linear.fake",
        "layout",
        provider=provider,
        defaults=(("threads", 64),),
        factory="matmul",
        semantic_keys=("r", "k", "o"),
        space=(("threads", (64, 128)),),
    )
    observed = []
    monkeypatch.setattr(
        "akas.kernels.definition.import_module",
        lambda module: SimpleNamespace(
            matmul=lambda *args: observed.append((module, args))
        ),
    )
    spec = KernelSpec("project", dict(r=4, k=256, o=32), "layout", "prefill")
    define(spec, dict(threads=128), Capabilities(provider), choice.declaration())
    assert observed == [
        (
            "akas.ops.linear.fake",
            (
                replace(spec, parameters=dict(threads=64, r=4, k=256, o=32)),
                dict(threads=128),
            ),
        )
    ]


def test_schedule_cannot_redefine_a_semantic_dimension():
    choice = Implementation(
        "project",
        "linear.fake",
        "layout",
        semantic_keys=("r", "k"),
        space=(("threads", (64, 128)),),
        factory="kernel",
    )
    with pytest.raises(ValueError, match="semantic"):
        definition(
            dict(kind="project", parameters=dict(r=4, k=256), contract="layout"),
            dict(r=1),
            catalog=(choice,),
        )


def test_legacy_registration_does_not_allow_schedule_to_change_its_abi():
    choice = Implementation("project", "linear.fake", "layout", factory="kernel")
    with pytest.raises(ValueError, match="semantic"):
        definition(
            dict(kind="project", parameters=dict(k=256), contract="layout"),
            dict(k=128),
            catalog=(choice,),
        )


def test_hardware_choice_requires_compatible_capabilities_and_geometry():
    base = Implementation(
        "project", "linear.fake", "layout", semantic_keys=("k",), factory="kernel"
    )
    hopper = replace(
        base,
        variant="hopper",
        targets=("sm_90a",),
        priority=20,
        accepts=lambda p: p["k"] % 128 == 0,
    )
    spec = dict(kind="project", parameters=dict(k=256), contract="layout")
    assert (
        definition(spec, capabilities=dict(target="sm_89"), catalog=(base, hopper))[0]
        is base
    )
    assert (
        definition(spec, capabilities=dict(target="sm_90a"), catalog=(base, hopper))[0]
        is hopper
    )
    with pytest.raises(ValueError, match="target"):
        define(spec, {}, dict(target="sm_89"), hopper.declaration())


@pytest.mark.parametrize(
    "changed,error",
    [
        ({"kind": "other"}, "operation"),
        ({"provider": "webgpu"}, "provider"),
        ({"contract": "other"}, "contract"),
        ({"phase": "decode"}, "phase"),
    ],
)
def test_exported_choice_rejects_mismatched_routing(changed, error):
    choice = Implementation(
        "project", "linear.fake", "layout", factory="kernel"
    ).declaration()
    with pytest.raises(ValueError, match=error):
        define(
            dict(kind="project", contract="layout", phase="prefill"),
            {},
            {},
            {**choice, **changed},
        )


def test_serialized_source_hashes_selected_dynamic_leaf(monkeypatch):
    choice = Implementation(
        "project", "akas.ops.linear.kernel", "layout", factory="linear"
    )
    source = selected_source(
        dict(kind="project", parameters={}, contract="layout"),
        {},
        dict(provider="cuda", target="sm_89"),
        choice,
    )
    monkeypatch.setattr(
        "tensor.compiler.frontend.source.export_source",
        lambda module, factory, *args, dependencies=(): json.dumps(
            dict(module=module, factory=factory, args=args, dependencies=dependencies)
        ),
    )
    serialized = json.loads(source.render())
    assert serialized["module"] == "akas.kernels.definition"
    assert serialized["factory"] == "define"
    assert "akas.ops.linear.kernel" in serialized["dependencies"]
    assert "akas.kernels.reductions" in serialized["dependencies"]
    assert not any(
        (name.startswith("akas.execution.") for name in serialized["dependencies"])
    )


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("kind", ["linear", "ffn", "embedding"])
def test_packed_weights_route_under_common_graph_contract(provider, kind):
    from akas.kernels.registry import select

    parameters = dict(r=4, k=256, o=32, type=2, ggml=True)
    eligible = candidates(
        kind,
        parameters,
        contract="reference",
        provider=provider,
        phase="prefill",
        target=None,
    )
    assert eligible
    chosen = select(kind, parameters, contract="reference", provider=provider,
                    phase="prefill", target=None)
    assert chosen.factory in (
        kind,
        "kernel_" + kind,
        "embedding_packed",
        "packed_word_embedding",
    )
    assert chosen.name is not None


def test_existing_hardware_algorithm_uses_common_boundary_without_dispatch_wrapper(
    monkeypatch,
):
    choice = Implementation(
        "attention_partial",
        "akas.ops.attention.cuda",
        "layout",
        targets=("sm_90a",),
        factory="hopper_prefill_split_partial",
        name=None,
    )
    observed = []
    monkeypatch.setattr(
        "akas.kernels.definition.import_module",
        lambda module: SimpleNamespace(
            hopper_prefill_split_partial=lambda spec, schedule: observed.append(
                (module, dict(spec.parameters), schedule)
            )
        ),
    )
    define(
        dict(
            kind="attention_partial",
            parameters=dict(slots=2, chunk=8),
            contract="layout",
        ),
        {},
        dict(target="sm_90a"),
        choice.declaration(),
    )
    assert observed == [("akas.ops.attention.cuda", dict(slots=2, chunk=8), {})]


def test_output_abi_is_operation_owned_and_unknown_kinds_are_conservative():
    from akas.graph import Call
    from akas.kernels.registry import graph_outputs

    assert graph_outputs(
        Call("rope", {}, ("x", "rotated", "positions")), contract="reference"
    ) == ("rotated",)
    assert graph_outputs(
        Call("conv", {}, ("x", "w", "state", "out", "enabled")), contract="reference"
    ) == ("state", "out")
    assert graph_outputs(
        Call("attention_merge", {}, ("parts", "out", "enabled")), contract="reference"
    ) == ("out",)
    assert (
        graph_outputs(Call("unknown", {}, ("x", "out")), contract="reference") is None
    )


def test_schedule_cannot_introduce_undeclared_storage_or_algorithm_parameters():
    choice = Implementation(
        "project",
        "linear.fake",
        "layout",
        space=(("threads", (64, 128)),),
        factory="kernel",
    )
    with pytest.raises(ValueError, match="undeclared"):
        definition(
            dict(kind="project", parameters={}, contract="layout"),
            dict(weight_dtype="float16"),
            catalog=(choice,),
        )


def test_named_native_outputs_include_residual_state_and_compound_pipeline_writes():
    from akas.graph import Call
    from akas.kernels.registry import graph_outputs

    assert graph_outputs(
        Call(
            "add_rms", {}, bindings=dict(x="x", residual="state", w="w", out="normal")
        ),
        contract="bf16",
    ) == ("state", "normal")
    assert graph_outputs(
        Call(
            "gdn_conv",
            dict(full_checkpoints=True, commit=False),
            bindings=dict(
                x="x",
                w="w",
                lengths="lengths",
                state="state",
                out="out",
                checkpoints="journal",
            ),
        ),
        contract="block-fp8",
    ) == ("out", "journal")
    assert graph_outputs(
        Call("attention_chunk_partial", {}, bindings=dict(out="parts", stats="stats")),
        contract="block-fp8",
    ) == ("parts", "stats")
    assert graph_outputs(
        Call(
            "quantize_q16",
            {},
            bindings=dict(packed="packed", scales="scales", sums="sums"),
        ),
        contract="reference",
    ) == ("packed", "scales", "sums")
