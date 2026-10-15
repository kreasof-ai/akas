"""New model composition, target routing, beam qualification and real execution."""

from dataclasses import replace
import json
import os
from types import SimpleNamespace
import numpy as np
import pytest
from akas.graph import Graph
from akas.execution.plan import bind_graph
from akas.compilation.compile import plan as compile_plan
from akas.compilation.tuning import Measurement
from akas.compilation.tuning import Workload
from akas.compilation.tuning import tune
from akas.kernels.registry import Implementation
from akas.kernels.registry import select
from examples.graphs.residual.config import Config
from examples.graphs.residual.graph import compose


def test_new_model_uses_both_providers_without_compiler_branches():
    forward = compose(Config(), rows=3)
    for provider, target in (("cuda", "sm_89"), ("webgpu", "webgpu-portable-v1")):
        bundle = compile_plan(forward, provider=provider, target=target)
        assert len(bundle.kernels) == 2
        assert bundle.manifest["forward"] == forward.metadata()
        assert (
            bundle.kernels[0].source.args[3]["module"] == f"akas.ops.linear.{provider}"
        )
        restored = Graph.from_metadata(json.loads(json.dumps(forward.metadata())))
        assert restored.requirements() == forward.requirements()


def test_calls_derive_unique_kernel_requirements():
    forward = compose(Config())
    call = forward.calls[0]
    forward(call.kind, call.parameters, *call.arguments)
    assert len(forward.calls) == 3
    assert len(compile_plan(forward).kernels) == 2


def test_named_bindings_follow_compiled_argument_order(monkeypatch):
    forward = Graph(contract="test")
    a = forward.buffer("a", (1,), "float32", role="input")
    b = forward.buffer("b", (1,), "float32", role="output")
    call = forward("op", {}, x=a, out=b)
    kernel = SimpleNamespace(manifest={"arguments": [{"name": "out"}, {"name": "x"}]})
    values = {name: SimpleNamespace(shape=(1,), dtype="float32") for name in ("a", "b")}
    monkeypatch.setattr("akas.execution.plan.bind", lambda device, kernel, args: args)
    result = bind_graph(forward, None, {call.key: kernel}, values)
    assert result[0][1] == (values["b"], values["a"])
    values["b"].shape = (2,)
    with pytest.raises(ValueError, match="incompatible forward buffer"):
        bind_graph(forward, None, {call.key: kernel}, values)


def test_recording_and_binding_share_kernel_emitter():
    from akas.execution.plan import KernelPlan

    forward = Graph(contract="test")
    x = forward.buffer("x", (1,), "float32")
    KernelPlan(forward, None)("op", dict(r=1), x=x, out=x)
    assert forward.calls[0].bindings == {"x": "x", "out": "x"}


def test_hardware_variant_is_qualified_before_routing():
    generic = Implementation(
        "add", "residual.cuda.packed_baseline", "packed", factory="kernel"
    )
    hopper = replace(
        generic,
        module="residual.cuda.hopper",
        targets=("sm_90a",),
        priority=20,
        accepts=lambda p: p["c"] % 128 == 0,
    )
    catalog = [generic, hopper]
    options = dict(contract="packed", catalog=catalog)
    assert select("add", {"c": 128}, target="sm_90a", **options) is hopper
    assert select("add", {"c": 64}, target="sm_90a", **options) is generic
    assert select("add", {"c": 128}, target="sm_89", **options) is generic
    with pytest.raises(ValueError, match="no compatible"):
        select(
            "add", {"c": 128}, target="sm_89", preferred=hopper.identifier, **options
        )
    with pytest.raises(ValueError, match="ambiguous"):
        select(
            "add",
            {"c": 128},
            target="sm_89",
            contract="packed",
            catalog=[generic, generic],
        )


def test_existing_fp8_linear_variant_preserves_large_batch_selection():
    for rows, expected in ((8, "linear.cuda.decode"), (16, "linear.cuda.block_fp8")):
        variant = select(
            "fp8_linear",
            dict(r=rows, k=2048, o=8192),
            contract="block-fp8",
            phase="decode",
            target="sm_89",
        )
        assert variant.identifier == expected
        if rows == 16:
            assert dict(variant.defaults)["columns"] == 64


def test_beam_search_rejects_incorrect_and_batch_regressing_candidates():
    baseline = Implementation(
        "add", "baseline", "packed", space=(("threads", (64, 128)),), factory="kernel"
    )
    fastest_small = replace(baseline, module="small")
    wrong = replace(baseline, module="wrong")
    stable = replace(baseline, module="residual.cuda.packed_baseline")
    catalog = [
        baseline,
        replace(fastest_small, priority=0),
        replace(wrong, priority=0),
        replace(stable, priority=0),
    ]

    def measure(variant, p, workload):
        seconds = {
            "baseline": (10, 10),
            "small": (2, 30),
            "wrong": (1, 1),
            stable.module: (6, 6),
        }[variant.module][p["r"] != 1]
        return Measurement(seconds, variant.module != "wrong")

    result = tune(
        "add",
        {"c": 256},
        [Workload("single", {"r": 1}), Workload("large", {"r": 64})],
        measure,
        contract="packed",
        target="sm_89",
        catalog=catalog,
        budget=20,
    )
    assert result.implementation == stable.identifier
    assert result.score == 0.6
    assert result.trials == 8
    profile = result.profile("add", {"c": 256}, provider="cuda", target="sm_89")
    forward = Graph(contract="packed")
    x = forward.buffer("x", (256,), "float32")
    forward("add", {"c": 256, "r": 1}, x, x, x)
    bundle = compile_plan(forward, schedules=profile, catalog=catalog)
    assert bundle.kernels[0].source.args[3]["module"] == "akas.ops." + stable.module
    with pytest.raises(ValueError, match="provider/target"):
        compile_plan(forward, target="sm_90a", schedules=profile, catalog=catalog)


@pytest.mark.parametrize(
    "seconds,correct", [(0, True), (float("nan"), True), (1, False)]
)
def test_invalid_reference_cannot_qualify_tuning(seconds, correct):
    with pytest.raises(ValueError, match="reference measurement"):
        tune(
            "add",
            {"c": 256},
            [Workload("test")],
            lambda *args: Measurement(seconds, correct),
            contract="packed",
            target="sm_89",
        )


@pytest.mark.skipif(
    os.environ.get("AKAS_FORWARD_CUDA") != "1",
    reason="set AKAS_FORWARD_CUDA=1 for CUDA qualification",
)
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("graphs", [False, True])
def test_three_file_model_matches_numpy_on_cuda(tmp_path, rows, graphs):
    import struct
    import tensor
    from akas.compilation.compile import produce
    from akas.execution.forward import Forward
    from examples.graphs.residual.weights import load

    config = Config()
    rng = np.random.default_rng(31)
    w = rng.normal(0, 0.05, (config.width, config.width)).astype("float32")
    header = json.dumps(
        {
            "projection": dict(
                dtype="F32", shape=list(w.shape), data_offsets=[0, w.nbytes]
            )
        }
    ).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + w.tobytes()
    )
    forward = compose(config, rows=rows)
    produce(forward, tmp_path / "bundle", target="sm_89")
    with tensor.Device() as device:
        weights = load(tmp_path, config, device)
        x = device.empty((rows * config.width,), "float32")
        out = device.empty(x.shape, "float32")
        try:
            with Forward(
                tmp_path / "bundle",
                device,
                {**weights, "input": x, "output": out},
                graphs=graphs,
            ) as runtime:
                for _ in range(2):
                    import ctypes as ct

                    values = rng.normal(0, 0.05, (rows, config.width)).astype("float32")
                    device.driver.call(
                        "cuMemcpyHtoD_v2",
                        x.pointer,
                        ct.c_void_p(values.ctypes.data),
                        values.nbytes,
                    )
                    runtime.run()
                    np.testing.assert_allclose(
                        out.to_numpy().reshape(values.shape),
                        values + values @ w.T,
                        rtol=0.03,
                        atol=0.0005,
                    )
            runtime.close()
            assert not x._released
        finally:
            for value in [*weights.values(), x, out]:
                value.release()


@pytest.mark.skipif(
    os.environ.get("AKAS_FORWARD_CUDA") != "1", reason="CUDA tuning qualification"
)
def test_beam_winner_builds_and_runs_on_representative_cuda_batches(tmp_path):
    import time
    import tensor
    from tensor.compiler.search import ScheduleProfile
    from akas.compilation.compile import build
    from akas.execution.forward import Forward

    config = Config()
    rng = np.random.default_rng(7)
    weight = rng.normal(0, 0.05, (config.width, config.width)).astype("float32")
    with tensor.Device() as device:
        w = device.from_numpy(weight.reshape(-1))

        def measure(implementation, parameters, workload):
            rows = parameters["r"]
            schedule = {
                key: value
                for key, value in parameters.items()
                if key not in ("r", "k", "o", "type")
            }
            profile = ScheduleProfile(
                dict(
                    schema="tensor.schedule-profile.v1",
                    provider="cuda",
                    target="sm_89",
                    entries=[
                        dict(
                            operation="linear",
                            parameters={},
                            schedule={
                                **schedule,
                                "implementation": implementation.module,
                            },
                        )
                    ],
                )
            )
            bundle = compile_plan(compose(config, rows=rows), schedules=profile)
            path = tmp_path / workload.name
            build(bundle, path, search=False)
            values = rng.normal(0, 0.05, (rows, config.width)).astype("float32")
            x = device.from_numpy(values.reshape(-1))
            out = device.empty(x.shape, "float32")
            try:
                with Forward(
                    path,
                    device,
                    {"input": x, "projection": w, "output": out},
                    graphs=True,
                ) as runtime:
                    runtime.run()
                    correct = bool(
                        np.allclose(
                            out.to_numpy().reshape(values.shape),
                            values + values @ weight.T,
                            rtol=0.03,
                            atol=0.0005,
                        )
                    )
                    start = time.perf_counter()
                    for _ in range(10):
                        runtime.run()
                    device.synchronize()
                    return Measurement((time.perf_counter() - start) / 10, correct)
            finally:
                x.release()
                out.release()

        try:
            result = tune(
                "linear",
                dict(k=config.width, o=config.width, type=0),
                [Workload("one", {"r": 1}), Workload("four", {"r": 4})],
                measure,
                contract="packed",
                target="sm_89",
                budget=2,
            )
            assert result.trials == 2
            assert len(result.measurements) == 2
            profile = result.profile("linear", {}, provider="cuda", target="sm_89")
            for rows in (1, 4):
                selected = profile.select("linear", {}, provider="cuda", target="sm_89")
                implementation = select(
                    "linear",
                    dict(r=rows, k=config.width, o=config.width, type=0),
                    contract="packed",
                    target="sm_89",
                    preferred=selected.pop("implementation"),
                )
                assert measure(
                    implementation,
                    dict(r=rows, k=config.width, o=config.width, type=0, **selected),
                    Workload(f"winner-{rows}"),
                ).correct
        finally:
            w.release()


def test_forward_failure_releases_only_owned_buffers(tmp_path, monkeypatch):
    from akas.execution.forward import Forward

    forward = compose(Config())
    manifest = dict(schema="akas.forward.v1", forward=forward.metadata(), kernels={})
    (tmp_path / "forward.json").write_text(json.dumps(manifest))
    allocated = []

    class Buffer:
        released = False

        def __init__(self, shape, dtype):
            self.shape, self.dtype = (shape, dtype)

        def release(self):
            self.released = True

    class Device:
        def empty(self, shape, dtype):
            value = Buffer(shape, dtype)
            allocated.append(value)
            return value

    supplied = {
        name: Buffer(spec.shape, spec.dtype)
        for name, spec in forward.buffers.items()
        if spec.role != "scratch"
    }

    def fail(*args, **kwargs):
        raise ValueError("bad artifact")

    monkeypatch.setattr("akas.execution.forward.load_kernels", fail)
    with pytest.raises(ValueError, match="bad artifact"):
        Forward(tmp_path, Device(), supplied)
    assert len(allocated) == 1 and allocated[0].released
    assert not any((value.released for value in supplied.values()))


def test_forward_import_and_metadata_need_no_compiler():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "\nimport importlib.abc, sys\nclass Guard(importlib.abc.MetaPathFinder):\n    def find_spec(self, name, path=None, target=None):\n        if name.startswith(('akas.compilation', 'tensor.compiler')) or name.split('.')[0] in {'torch', 'tilelang', 'tvm', 'tvm_ffi', 'triton'}:\n            raise ImportError(name)\nsys.meta_path.insert(0, Guard())\nfrom akas.graph import Graph\nfrom akas.execution.plan import bind_graph\nfrom akas.execution.forward import Forward\nfrom examples.graphs.residual.config import Config\nfrom examples.graphs.residual.graph import compose\nplan = compose(Config())\nassert Graph.from_metadata(plan.metadata()).requirements() == plan.requirements()\n",
        ],
        check=True,
    )


def test_new_operation_registration_is_discovered_without_central_table(
    tmp_path, monkeypatch
):
    import sys
    import akas.kernels.registry as registry

    package = tmp_path / "ops" / "future_op"
    package.mkdir(parents=True)
    (package / "definition.py").write_text("# registration module supplied below\n")
    item = Implementation("future", "future_op.cuda", "new-contract", factory="kernel")
    monkeypatch.setitem(
        sys.modules,
        "akas.ops.future_op.definition",
        SimpleNamespace(IMPLEMENTATIONS=(item,)),
    )
    monkeypatch.setattr(registry, "__file__", str(tmp_path / "kernels" / "registry.py"))
    registry.implementations.cache_clear()
    try:
        assert (
            registry.select("future", {}, contract="new-contract", target="sm_89")
            is item
        )
    finally:
        registry.implementations.cache_clear()


def test_tuning_qualifies_using_workload_geometry():
    variant = Implementation(
        "linear",
        "large_only",
        "packed",
        accepts=lambda p: p.get("r", 0) >= 16,
        factory="kernel",
    )
    result = tune(
        "linear",
        {"k": 256},
        [Workload("large", {"r": 32})],
        lambda *args: Measurement(1, True),
        contract="packed",
        target="sm_89",
        catalog=[variant],
        budget=1,
    )
    assert result.implementation == variant.identifier


@pytest.mark.parametrize(
    "provider,compiler,nvrtc",
    [("cuda", "nvrtc", "/cuda-only"), ("webgpu", "wgsl", None)],
)
def test_provider_build_does_not_leak_cuda_options(
    tmp_path, monkeypatch, provider, compiler, nvrtc
):
    import tensor
    from akas.compilation.emit import build_artifact

    monkeypatch.setenv("TENSOR_NVRTC_HOME", "/cuda-only")
    monkeypatch.setattr(tensor, "build", lambda *args, **kwargs: kwargs)
    options = build_artifact(
        tmp_path / "source.py",
        tmp_path / "result.tbin",
        target="test",
        provider=provider,
    )
    assert options["compiler"] == compiler
    assert options["nvrtc_home"] == nvrtc


def test_forward_provider_has_correct_default_target():
    assert (
        compile_plan(compose(Config()), provider="webgpu").target
        == "webgpu-portable-v1"
    )


@pytest.mark.parametrize("rows", [1, 4])
def test_new_model_compiles_to_webgpu_without_device(tmp_path, monkeypatch, rows):
    from akas.compilation.compile import produce
    from tensor.artifacts.format import read_artifact

    monkeypatch.setenv("TENSOR_NVRTC_HOME", "/cuda-option-must-not-reach-webgpu")
    manifest = produce(
        compose(Config(), rows=rows), tmp_path, provider="webgpu", search=False
    )
    for row in manifest["kernels"].values():
        artifact, _ = read_artifact(tmp_path / row["path"])
        assert artifact["provider"] == "webgpu"
        assert artifact["target"] == "webgpu-portable-v1"


def test_webgpu_prepared_capture_accepts_labeled_forward_calls():
    from akas.tensor.executor import capture

    device = SimpleNamespace(
        info={"provider": "webgpu"}, prepare_plan=lambda calls: calls
    )
    labeled = [("kernel", "bound-call", "layer-label")]
    assert capture(device, labeled, resources=()) == [("kernel", "bound-call")]
    assert labeled[0][2] == "layer-label"


def test_beam_search_keeps_default_when_candidates_regress_a_workload():
    variant = Implementation(
        "linear", "baseline", "reference", space=(("threads", (32,)),), factory="kernel"
    )

    def measure(item, parameters, workload):
        if "threads" not in parameters:
            return Measurement(10, True)
        return Measurement(2 if parameters["r"] == 1 else 20, True)

    result = tune(
        "linear",
        {},
        [Workload("small", {"r": 1}), Workload("large", {"r": 256})],
        measure,
        contract="reference",
        target="sm_89",
        catalog=[variant],
        budget=4,
    )
    assert result.schedule == {}
    assert result.score == 1
    assert result.measurements == (10, 10)


def test_beam_search_accepts_improvement_when_an_unchanged_workload_ties():
    variant = Implementation(
        "linear", "baseline", "reference", space=(("threads", (32,)),), factory="kernel"
    )

    def measure(item, parameters, workload):
        return Measurement(
            5 if "threads" in parameters and parameters["r"] > 1 else 10, True
        )

    result = tune(
        "linear",
        {},
        [Workload("small", {"r": 1}), Workload("large", {"r": 256})],
        measure,
        contract="reference",
        target="sm_89",
        catalog=[variant],
        budget=4,
    )
    assert result.schedule == {"threads": 32}
    assert result.score == 1
    assert result.measurements == (10, 5)


def test_operation_owner_can_extend_row_pruning_and_fusion(monkeypatch):
    from akas.compilation.rewrites import optimize
    from akas.kernels.registry import planning_rules

    original = planning_rules()
    forward = Graph(contract="reference")
    x = forward.buffer("x", (8,), "float32")
    y = forward.buffer("y", (8,), "float32")
    final = forward.buffer("final", (4,), "float32", role="output")
    counts = forward.buffer("counts", (2,), "int32", role="input")
    forward("new_activation", dict(r=4, c=2), x, y)
    forward("chunk_last", dict(r=2, c=2, chunk=2), y, final, counts)
    assert len(optimize(forward).calls) == 2
    observed = []

    def rewrite(plan):
        observed.append(plan.calls[-1].kind)
        return plan

    monkeypatch.setattr(
        "akas.kernels.registry.planning_rules",
        lambda: (
            *original,
            SimpleNamespace(ROW_INDEPENDENT=("new_activation",), REWRITES=(rewrite,)),
        ),
    )
    optimized = optimize(forward)
    assert optimized.calls[-1].kind == "new_activation"
    assert optimized.calls[-1].parameters["r"] == 2
    assert optimized.buffers["y"].shape == (4,)
    assert observed == ["new_activation"]


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_new_model_searches_by_default_and_reuses_qualified_cache(
    tmp_path, monkeypatch, provider
):
    if os.environ.get("AKAS_FORWARD_" + provider.upper()) != "1":
        pytest.skip("enable provider for default search qualification")
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings

    graph = compose(Config(), rows=4)
    settings = SearchSettings(budget=2, samples=2, repeats=2)
    manifest = produce(graph, tmp_path, provider=provider, search_settings=settings)
    report = manifest["tuning"]
    assert not report["cache_hit"]
    assert len(report["operations"]) == len(graph.requirements())
    assert all(
        (row["candidates"][0]["status"] == "passed" for row in report["operations"])
    )
    assert all((row["selected"]["measurements"][0] > 0 for row in report["operations"]))

    def unexpected_measurement(*args, **kwargs):
        pytest.fail("compatible cached qualification must not rerun timing")

    monkeypatch.setattr("tensor.compiler.tuning.measure", unexpected_measurement)
    cached = produce(graph, tmp_path, provider=provider, search_settings=settings)
    assert cached["tuning"]["cache_hit"]
    assert cached["kernels"] == manifest["kernels"]


def test_view_lifetime_prevents_scratch_reuse_until_last_alias_read():
    from akas.compilation.buffers import scratch_slots

    graph = Graph(contract="test")
    base = graph.buffer("base", (6,), "float32")
    view = graph.view("view", base, (2, 3))
    other = graph.buffer("other", (6,), "float32")
    graph("write", {}, base)
    graph("write", {}, other)
    graph("read", {}, view)
    assert scratch_slots(graph) == [["base"], ["other"]]
    restored = Graph.from_metadata(json.loads(json.dumps(graph.metadata())))
    assert restored.buffers["view"].alias_of == "base"
    assert restored.metadata() == graph.metadata()
    with pytest.raises(ValueError, match="same dtype and size"):
        graph.view("bad", base, (7,))
    foreign = Graph(contract="test").buffer("base", (6,), "float32")
    with pytest.raises(ValueError, match="another graph"):
        graph.view("foreign", foreign, (6,))
