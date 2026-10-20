"""Cache materialization preserves FP8 precision, bounds and request isolation."""

from contextlib import ExitStack
import os

import numpy as np
import pytest


def reference_graph(*, chunk=256, capacity=4096, layers=2):
    from akas.graph import Graph

    graph = Graph(contract="reference", phase="prefill")
    slots, heads, kv_heads, depth = 2, 8, 2, 64
    rows = slots * chunk
    for name, shape, dtype, role in (
        ("q", (rows, heads * depth), "float32", "input"),
        ("kc", (3, kv_heads, capacity, depth), "uint8", "state"),
        ("vc", (3, kv_heads, capacity, depth), "uint8", "state"),
        ("ks", (3, kv_heads, capacity, 1), "float32", "state"),
        ("vs", (3, kv_heads, capacity, 1), "float32", "state"),
        ("positions", (rows,), "int32", "input"),
        ("enabled", (rows,), "int32", "input"),
        ("mapping", (slots,), "int32", "input"),
    ):
        graph.buffer(name, shape, dtype, role=role)
    p = dict(
        r=rows,
        h=heads,
        kh=kv_heads,
        d=depth,
        chunk=chunk,
        capacity=capacity,
        kv_dtype="fp8",
        optimized=True,
        provider="cuda",
    )
    for index in range(layers):
        out = graph.buffer(f"out{index}", (rows * heads * depth,), "float32")
        graph(
            "attention",
            p,
            "q",
            "kc",
            "vc",
            "ks",
            "vs",
            out,
            "positions",
            "enabled",
            label=f"attention{index}",
        )
    return graph


def test_staging_is_a_searchable_pipeline_with_one_shared_scratch_pair():
    from akas.ops.attention.definition import lower, search_policies
    from akas.kernels.registry import graph_outputs

    graph = reference_graph()
    assert dict(splits=64, decoded_kv=True) in search_policies(
        graph, None, provider="cuda", target="sm_89"
    )
    graph, commit = lower(
        graph, None, dict(splits=64, decoded_kv=True), provider="cuda", target="sm_89"
    )
    assert commit is None
    assert [c.kind for c in graph.calls] == [
        "attention_cache_extent",
        "kv_stage",
        "attention",
        "attention_cache_extent",
        "kv_stage",
        "attention",
    ]
    first, second = graph.calls[1], graph.calls[4]
    assert first.arguments[-4:] == second.arguments[-4:]
    assert graph_outputs(first, contract="reference") == first.arguments[-3:]
    assert all(graph.buffers[n].role == "scratch" for n in first.arguments[-4:])
    assert graph.buffers[first.arguments[-3]].shape == (2, 2, 4096, 64)
    assert graph.calls[2].parameters["kv_dtype"] == "float16"
    assert graph.calls[2].parameters["unit_scales"]
    assert graph.calls[2].label == "attention0"
    assert graph.calls[-1].arguments[5] == "out1"
    assert all(graph.buffers[n].role == "state" for n in ("kc", "vc", "ks", "vs"))


def test_staging_preserves_small_chunks_and_rejects_unqualified_provider():
    from akas.ops.attention.definition import lower

    graph = reference_graph(chunk=32)
    original = graph.metadata()
    lower(graph, None, dict(decoded_kv=True), provider="cuda", target="sm_89")
    assert graph.metadata() == original
    with pytest.raises(ValueError, match="qualified on CUDA"):
        lower(
            reference_graph(),
            None,
            dict(decoded_kv=True),
            provider="webgpu",
            target="webgpu",
        )


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_conversion_and_extent_are_registered_for_both_providers(provider):
    from akas.kernels.registry import select

    assert (
        select("kv_stage", {}, contract="reference", provider=provider).factory
        == "stage"
    )
    assert (
        select(
            "attention_cache_extent", {}, contract="reference", provider=provider
        ).factory
        == "cache_extent"
    )


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("depth", [64, 129])
def test_standalone_staging_and_extent(tmp_path, provider, depth):
    if (
        os.environ.get(
            "AKAS_LFM_MOE_CUDA" if provider == "cuda" else "AKAS_LFM_MOE_WEBGPU"
        )
        != "1"
    ):
        pytest.skip("opt-in cache staging qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.kernels.registry import validation

    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    parameters = {
        "kv_stage": dict(slots=3, kv_heads=2, capacity=257, d=depth),
        "attention_cache_extent": dict(slots=3, chunk=33, capacity=257),
    }
    for kind, p in parameters.items():
        entry, artifact = tmp_path / f"{kind}.py", tmp_path / f"{kind}.tbin"
        entry.write_text(
            source(
                kind, p, contract="reference", provider=provider, target=target
            ).render()
        )
        build_artifact(entry, artifact, provider=provider, target=target)
        metadata, _ = read_artifact(artifact)
        case = validation(kind, p, metadata["arguments"])
        with tensor.Device(provider=provider) as device, ExitStack() as stack:
            buffers = {n: device.from_numpy(a) for n, a in case.arguments.items()}
            for buffer in buffers.values():
                stack.callback(buffer.release)
            module = stack.enter_context(device.load(artifact))
            module.launch(*[buffers[a["name"]] for a in metadata["arguments"]])
            for name, expected in case.expected.items():
                np.testing.assert_array_equal(buffers[name].to_numpy(), expected)


def test_mapped_staging_preserves_unselected_request_slots(tmp_path):
    if os.environ.get("AKAS_LFM_MOE_CUDA") != "1":
        pytest.skip("opt-in mapped cache staging qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.compilation.state import mapped_source
    from akas.kernels.contracts import mapped_validation

    p = dict(slots=2, kv_heads=2, capacity=257, d=64)
    mapping = ((0, 1, 2, 3), 2, 3)
    declaration = source(
        "kv_stage", p, contract="reference", provider="cuda", target="sm_89"
    )
    entry, artifact = tmp_path / "mapped.py", tmp_path / "mapped.tbin"
    entry.write_text(mapped_source(declaration, mapping).render())
    build_artifact(entry, artifact, provider="cuda", target="sm_89")
    metadata, _ = read_artifact(artifact)
    case = mapped_validation("kv_stage", p, metadata["arguments"], mapping)
    with tensor.Device(provider="cuda") as device, ExitStack() as stack:
        buffers = {n: device.from_numpy(a) for n, a in case.arguments.items()}
        for buffer in buffers.values():
            stack.callback(buffer.release)
        module = stack.enter_context(device.load(artifact))
        module.launch(*[buffers[a["name"]] for a in metadata["arguments"]])
        for name, expected in case.expected.items():
            np.testing.assert_array_equal(buffers[name].to_numpy(), expected)


def test_staged_attention_matches_fp8_oracle_with_poisoned_padding(tmp_path):
    if os.environ.get("AKAS_LFM_MOE_CUDA") != "1":
        pytest.skip("opt-in staged attention pipeline qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.execution.plan import ExecutionPlan
    from akas.memory.transfer import write
    from akas.ops.attention.validation import canonical_validation

    p = dict(
        r=66,
        h=8,
        kh=2,
        d=64,
        capacity=512,
        chunk=33,
        kv_dtype="fp8",
        provider="cuda",
        optimized=True,
        prefill_precision="float16",
        key_rows=64,
    )
    parameters = {
        "attention_cache_extent": dict(slots=2, chunk=33, capacity=512),
        "kv_stage": dict(slots=2, kv_heads=2, capacity=512, d=64),
        "attention": dict(
            p,
            kv_dtype="float16",
            unit_scales=True,
            staged_kv=True,
            query_rows=32,
            async_loads=True,
            stages=1,
            threads=128,
        ),
    }
    artifacts = {}
    for kind, params in parameters.items():
        entry, artifact = tmp_path / f"{kind}.py", tmp_path / f"{kind}.tbin"
        entry.write_text(
            source(
                kind, params, contract="reference", provider="cuda", target="sm_89"
            ).render()
        )
        build_artifact(entry, artifact, provider="cuda", target="sm_89")
        artifacts[kind] = artifact
    signature, _ = read_artifact(artifacts["attention"])
    signature = [
        dict(a, dtype="uint8") if a["name"] in ("kc", "vc") else a
        for a in signature["arguments"]
    ]
    case = canonical_validation("attention", p, signature)
    clean = {name: value.copy() for name, value in case.arguments.items()}
    with tensor.Device(provider="cuda") as device, ExitStack() as stack:
        buffers = {n: device.from_numpy(a) for n, a in case.arguments.items()}
        lengths = device.empty((2,), "int32")
        keys = device.empty((2, 2, 512, 64), "float16")
        values = device.empty(keys.shape, "float16")
        scales = device.empty((2, 2, 512, 1), "float32")
        for buffer in (*buffers.values(), lengths, keys, values, scales):
            stack.callback(buffer.release)
        modules = {
            kind: stack.enter_context(device.load(path))
            for kind, path in artifacts.items()
        }
        plan = ExecutionPlan(device)
        for kind, bindings in (
            ("attention_cache_extent", dict(buffers, lengths=lengths)),
            (
                "kv_stage",
                dict(buffers, lengths=lengths, keys=keys, values=values, scales=scales),
            ),
            ("attention", dict(buffers, kc=keys, vc=values, ks=scales, vs=scales)),
        ):
            module = modules[kind]
            plan.bind(
                module, tuple(bindings[a["name"]] for a in module.manifest["arguments"])
            )
        prepared = plan.capture(
            resources=tuple(buffers.values())
            + tuple(modules.values())
            + (lengths, keys, values, scales)
        )
        stack.callback(prepared.close)
        for poisoned in (True, False):
            initial = {name: value.copy() for name, value in clean.items()}
            expected = case.expected["out"].copy().reshape(66, 8, 64)
            if poisoned:
                initial["enabled"][32:] = 0
                initial["q"][32:] = np.nan
                expected[32:] = 0
                for slot in range(2):
                    enabled = initial["enabled"][slot * 33 : (slot + 1) * 33] != 0
                    positions = initial["positions"][slot * 33 : (slot + 1) * 33]
                    end = int(positions[enabled].max()) + 1 if enabled.any() else 0
                    for name in ("kc", "vc"):
                        initial[name][slot, :, end:] = 127
                    for name in ("ks", "vs"):
                        initial[name][slot, :, end:] = np.nan
            initial["out"].fill(np.nan)
            for name, value in initial.items():
                write(device, buffers[name], value)
            prepared.launch()
            np.testing.assert_allclose(
                buffers["out"].to_numpy().reshape(66, 8, 64),
                expected,
                rtol=case.rtol,
                atol=case.atol,
            )


def test_staging_passes_the_unified_mapped_graph_compiler(tmp_path):
    if os.environ.get("AKAS_LFM_MOE_CUDA") != "1":
        pytest.skip("opt-in mapped graph compiler qualification")
    import tensor
    from akas.compilation.compile import build, plan
    from akas.compilation.tuning import SearchSettings
    from akas.ops.attention.definition import lower

    graph, _ = lower(
        reference_graph(layers=1),
        None,
        dict(decoded_kv=True),
        provider="cuda",
        target="sm_89",
    )
    bundle = plan(graph, provider="cuda", target="sm_89")
    stage = next(k for k in bundle.kernels if k.kind == "kv_stage")
    assert stage.mapping == ((0, 1, 2, 3), 2, 3)
    with tensor.Device(provider="cuda") as device:
        manifest = build(
            bundle,
            tmp_path,
            device=device,
            search_settings=SearchSettings(
                budget=1, width=1, warmup=0, samples=1, repeats=1
            ),
        )
    operations = manifest["tuning"]["operations"]
    assert operations and all(
        any(candidate["status"] == "passed" for candidate in op["candidates"])
        for op in operations
    )
    assert {k["kind"] for k in manifest["kernels"].values()} == {
        "attention_cache_extent",
        "kv_stage",
        "attention",
    }
