"""Packed LFM2 MoE routing, model composition and independent device checks."""

from contextlib import ExitStack
import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from akas.models.lfm2 import LFM2, LFM2MoE
from akas.checkpoints.gguf import GGUFError
from support.lfm_moe import fixture, Oracle


@pytest.mark.parametrize("dense_layers", [0, 1])
def test_packed_moe_declaration_and_backend_graphs(tmp_path, dense_layers):
    from akas.optimization.context import discover

    path = fixture(tmp_path / "moe.gguf", dense_layers=dense_layers)
    checkpoint = LFM2MoE.checkpoint_type(path, tokenizer=False)
    assert discover(path).endswith(":LFM2MoE")
    assert checkpoint.config.experts == 4 and checkpoint.config.top_k == 2
    with pytest.raises(GGUFError, match="architecture"):
        LFM2.checkpoint_type(path, tokenizer=False)
    for provider in ("cuda", "webgpu"):
        for chunk in (1, 4):
            graph = LFM2MoE.compose(
                checkpoint,
                rows=2,
                chunk=chunk,
                max_requests=4,
                max_context=64,
                provider=provider,
            ).graph
            kinds = {call.kind for call in graph.calls}
            assert {"biased_router", "quantized_experts", "routed_combine"} <= kinds
            assert graph.buffers["blk.1.ffn_gate_exps.weight"].dtype == (
                "uint8" if provider == "cuda" else "uint32"
            )
            assert graph.buffers["positions"].shape == (4,)
            caches = [
                spec
                for name, spec in graph.buffers.items()
                if name.endswith((".kc", ".vc"))
            ]
            assert caches and all(spec.dtype == "float16" for spec in caches)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m, t: m.update({"lfm2moe.expert_used_count": 5}),
        lambda m, t: m.update({"lfm2moe.expert_gating_func": 1}),
        lambda m, t: t.pop("blk.1.exp_probs_b.bias"),
    ],
)
def test_invalid_moe_geometry_and_missing_bias_rejected(tmp_path, mutation):
    with pytest.raises(GGUFError):
        LFM2MoE.checkpoint_type(
            fixture(tmp_path / "moe.gguf", mutate=mutation), tokenizer=False
        )


@pytest.mark.skipif(
    not Path("/proc/self/status").is_file(), reason="Linux RSS accounting"
)
def test_real_expert_geometry_validation_has_bounded_host_memory():
    """The 64-expert bank must stay packed while its reference uses small tiles."""
    script = """
import json
from akas.ops.moe.validation import encoded_validation
p = dict(r=1, top=4, experts=64, k=2048, o=1536,
         weight_parts=[dict(options=dict(encoding=2))])
signature = [dict(name=name, shape=shape, dtype=dtype) for name, shape, dtype in [
    ('x', [1,2048], 'float32'),
    ('w', [64*1536*2048//32*18], 'uint8'),
    ('ids', [1,4], 'int32'), ('counts', [64], 'int32'),
    ('out', [1,4,1536], 'float32')]]
case = encoded_validation('quantized_experts', p, signature)
with open('/proc/self/status') as stream:
    fields = dict(line.split(':',1) for line in stream if ':' in line)
print(json.dumps(dict(peak_bytes=int(fields['VmHWM'].split()[0])*1024,
                     output=list(case.expected['out'].shape))))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        timeout=60,
    )
    record = json.loads(result.stdout)
    assert record["output"] == [1, 4, 1536]
    assert record["peak_bytes"] < 512 * 1024**2, record


CASES = [
    ("biased_router", dict(r=3, experts=8, top=4)),
    ("routed_combine", dict(r=3, top=4, c=128)),
    ("expert_tile_map", dict(r=33, experts=64, top=4)),
    (
        "attention",
        dict(
            r=8,
            h=4,
            kh=1,
            d=64,
            capacity=64,
            chunk=4,
            kv_dtype="float16",
            optimized=True,
        ),
    ),
    (
        "attention",
        dict(
            r=8, h=4, kh=1, d=64, capacity=64, chunk=4, kv_dtype="fp8", optimized=True
        ),
    ),
    ("rope", dict(r=4, h=32, d=64, theta=10000000.0, position_limit=131072)),
] + [
    (
        "quantized_experts",
        dict(
            r=rows,
            top=2,
            experts=4,
            k=256,
            o=128,
            input_dtype="float32",
            routed_input=routed,
            weight_parts=({"rows": 512, "format": "ggml", "options": {"encoding": 2}},),
        ),
    )
    for rows in (1, 5, 33)
    for routed in (False, True)
]


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("kind,parameters", CASES)
def test_moe_kernel_device_qualification(tmp_path, provider, kind, parameters):
    if os.environ.get("AKAS_LFM_MOE_" + provider.upper()) != "1":
        pytest.skip("opt-in LFM2 MoE device qualification")
    import tensor
    from akas.compilation.selection import source
    from akas.compilation.emit import build_artifact
    from akas.kernels.registry import validation, select
    from tensor.artifacts.format import read_artifact

    parameters = {**parameters, "round_inputs": provider == "cuda"}
    if kind in ("rope", "attention"):
        parameters["provider"] = provider
    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    implementation = select(
        kind, parameters, contract="reference", provider=provider, target=target
    )
    parameters = {**dict(implementation.defaults), **parameters}
    entry, artifact = tmp_path / "kernel.py", tmp_path / "kernel.tbin"
    entry.write_text(
        source(
            kind, parameters, contract="reference", provider=provider, target=target
        ).render()
    )
    build_artifact(entry, artifact, provider=provider, target=target)
    meta, _ = read_artifact(artifact)
    case = validation(kind, parameters, meta["arguments"])
    if kind == "rope":
        case.arguments["positions"][:] = [0, 8191, 65535, 131071]
        x = case.arguments["x"].reshape(4, 32, 64).astype("float64")
        angles = case.arguments["positions"][:, None, None] * 10000000.0 ** (
            -2 * np.arange(32)[None, None, :] / 64
        )
        left, right = x[:, :, :32], x[:, :, 32:]
        expected = np.concatenate(
            (
                left * np.cos(angles) - right * np.sin(angles),
                left * np.sin(angles) + right * np.cos(angles),
            ),
            axis=-1,
        )
        case.expected["out"][:] = expected.reshape(case.expected["out"].shape)
    with tensor.Device(provider=provider) as device, ExitStack() as stack:
        buffers = []
        for argument in meta["arguments"]:
            buffer = device.from_numpy(case.arguments[argument["name"]])
            stack.callback(buffer.release)
            buffers.append(buffer)
        kernel = stack.enter_context(device.load(artifact))
        kernel.launch(*buffers)
        for argument, buffer in zip(meta["arguments"], buffers):
            np.testing.assert_allclose(
                buffer.to_numpy(),
                case.expected[argument["name"]],
                rtol=case.rtol,
                atol=case.atol,
            )
        if kind == "expert_tile_map":
            from akas.memory.transfer import write

            named = {
                argument["name"]: buffer
                for argument, buffer in zip(meta["arguments"], buffers)
            }
            write(device, named["counts"], np.zeros_like(case.arguments["counts"]))
            kernel.launch(*buffers)
            np.testing.assert_array_equal(named["tile_experts"].to_numpy(), -1)
            np.testing.assert_array_equal(named["tile_rows"].to_numpy(), 0)


@pytest.mark.parametrize("columns,block_k", [(32, 64), (64, 128), (128, 128)])
@pytest.mark.parametrize("routed", [False, True])
def test_cuda_prefill_expert_tile_shapes(tmp_path, columns, block_k, routed):
    test_moe_kernel_device_qualification(
        tmp_path,
        "cuda",
        "quantized_experts",
        dict(
            r=33,
            top=2,
            experts=4,
            k=256,
            o=128,
            input_dtype="float32",
            routed_input=routed,
            tile_map=True,
            columns=columns,
            block_k=block_k,
            block_m=32,
            weight_parts=({"rows": 512, "format": "ggml", "options": {"encoding": 2}},),
        ),
    )


@pytest.mark.parametrize(
    "block_m,columns,block_k,pad", [(4, 32, 32, 1), (32, 64, 32, 1), (32, 64, 64, 0)]
)
@pytest.mark.parametrize("routed", [False, True])
def test_webgpu_prefill_expert_tile_shapes(
    tmp_path, block_m, columns, block_k, pad, routed
):
    test_moe_kernel_device_qualification(
        tmp_path,
        "webgpu",
        "quantized_experts",
        dict(
            r=33,
            top=2,
            experts=4,
            k=256,
            o=128,
            input_dtype="float32",
            routed_input=routed,
            tile_map=True,
            columns=columns,
            block_k=block_k,
            block_m=block_m,
            rhs_pad=pad,
            weight_parts=({"rows": 512, "format": "ggml", "options": {"encoding": 2}},),
        ),
    )


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_moe_model_prefill_decode_and_slot_reuse(tmp_path, provider):
    if os.environ.get("AKAS_LFM_MOE_" + provider.upper()) != "1":
        pytest.skip("opt-in LFM2 MoE model qualification")
    import tensor
    from akas.compilation.compile import compile_program
    from akas.compilation.tuning import SearchSettings
    from akas.execution.executor import Executor

    path = fixture(tmp_path / "moe.gguf", dense_layers=0)
    loaded = LFM2MoE.checkpoint_type(path, tokenizer=False)
    out = tmp_path / "program"
    compile_program(
        loaded,
        out,
        definition=LFM2MoE.compose,
        geometries=[(1, 1), (1, 4)],
        max_requests=2,
        max_context=64,
        provider=provider,
        search_settings=SearchSettings(budget=1, samples=1, repeats=1),
    )
    with tensor.Device(provider=provider) as device:
        with Executor.from_program(loaded, out, device) as executor:
            slot = executor.admit_at(1)
            oracle = Oracle(path, provider=provider)
            for sequence, chunk in (([1, 2, 3, 4], 4), ([5], 1), ([6], 1)):
                expected = oracle._chunk(
                    np.array(sequence, dtype="int32"),
                    projection_half=False,
                    head_half=False,
                )
                executor.run_batch([slot], [sequence], chunk=chunk)
                actual = (
                    executor.batches.select(1, chunk)
                    .forward.buffers["logits"]
                    .to_numpy()
                    .reshape(-1)
                )
                np.testing.assert_allclose(actual, expected, rtol=0.02, atol=0.003)
            executor.release(slot)
            assert executor.admit_at(slot) == slot and executor.positions[slot] == 0
            np.testing.assert_array_equal(
                executor.batches.states["blk.0.conv_state"].to_numpy()[slot], 0
            )


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_expert_search_reuses_reference_across_tiles(tmp_path, monkeypatch, provider):
    if os.environ.get("AKAS_LFM_MOE_" + provider.upper()) != "1":
        pytest.skip("opt-in LFM2 MoE search qualification")
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings
    from akas.kernels import registry

    rows, top, experts, depth, columns = 33, 2, 4, 256, 128
    parameters = dict(
        r=rows,
        top=top,
        experts=experts,
        k=depth,
        o=columns,
        tile_map=True,
        input_dtype="float32",
        routed_input=False,
        round_inputs=provider == "cuda",
        weight_parts=(
            {"rows": experts * columns, "format": "ggml", "options": {"encoding": 2}},
        ),
    )
    raw_bytes = experts * columns * depth // 32 * 18
    capacity = (rows * top + 3) // 4 + experts
    layouts = [
        ("x", (rows, depth), "float32", "input"),
        (
            "w",
            (raw_bytes if provider == "cuda" else raw_bytes // 4,),
            "uint8" if provider == "cuda" else "uint32",
            "weight",
        ),
        ("counts", (experts,), "int32", "input"),
        ("routes", (experts, rows), "int32", "input"),
        ("tile_experts", (5, capacity), "int32", "input"),
        ("tile_rows", (5, capacity), "int32", "input"),
        ("out", (rows, top, columns), "float32", "output"),
    ]
    graph = Graph(contract="reference", phase="prefill")
    buffers = [
        graph.buffer(name, shape, dtype, role=role)
        for name, shape, dtype, role in layouts
    ]
    graph("quantized_experts", parameters, *buffers)
    reference = registry.validation
    constructions = []

    def counted(kind, p, signature):
        constructions.append((kind, dict(p)))
        return reference(kind, p, signature)

    monkeypatch.setattr(registry, "validation", counted)
    result = produce(
        graph,
        tmp_path,
        provider=provider,
        search_settings=SearchSettings(
            budget=2, width=1, warmup=0, samples=1, repeats=1
        ),
    )
    operation = result["tuning"]["operations"][0]
    assert operation["selected"]["trials"] == 2
    assert len(operation["candidates"]) >= 3
    assert all(
        row["timing"]["execution"] == "prepared-plan"
        for row in operation["candidates"]
        if row["status"] == "passed"
    )
    assert len(constructions) == 1, [
        (p.get("block_m"), p.get("columns")) for _, p in constructions
    ]


def test_attention_search_keeps_precision_references_distinct(tmp_path, monkeypatch):
    if os.environ.get("AKAS_LFM_MOE_CUDA") != "1":
        pytest.skip("opt-in CUDA attention search qualification")
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings
    from akas.kernels import registry

    p = dict(
        r=8,
        h=4,
        kh=1,
        d=64,
        capacity=64,
        chunk=4,
        kv_dtype="float16",
        provider="cuda",
        optimized=True,
    )
    graph = Graph(contract="reference", phase="prefill")
    layouts = [
        ("q", (8, 256), "float32"),
        ("kc", (2, 1, 64, 64), "float16"),
        ("vc", (2, 1, 64, 64), "float16"),
        ("ks", (2, 1, 64, 1), "float32"),
        ("vs", (2, 1, 64, 1), "float32"),
        ("out", (2048,), "float32"),
        ("positions", (8,), "int32"),
        ("enabled", (8,), "int32"),
    ]
    args = [
        graph.buffer(n, s, d, role="output" if n == "out" else "input")
        for n, s, d in layouts
    ]
    graph("attention", p, *args)
    reference = registry.validation
    precision = []

    def counted(kind, parameters, signature):
        precision.append(parameters.get("prefill_precision", "float32"))
        return reference(kind, parameters, signature)

    monkeypatch.setattr(registry, "validation", counted)
    produce(
        graph,
        tmp_path,
        provider="cuda",
        search_settings=SearchSettings(
            budget=2, width=1, warmup=0, samples=1, repeats=1
        ),
    )
    assert {"float16", "float32"} <= set(precision)


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_floating_attention_split_policy_resizes_producer_and_consumer(provider):
    from akas.graph import Graph
    from akas.ops.attention.definition import floating_attention, lower, search_policies

    graph = Graph(contract="reference", phase="decode")
    p = dict(
        r=1,
        h=4,
        kh=1,
        d=64,
        capacity=128,
        kv_dtype="float16",
        optimized=True,
        provider=provider,
    )
    q = graph.buffer("q", (1, 256), "float32", role="input")
    kc = graph.buffer("kc", (1, 1, 128, 64), "float16", role="state")
    vc = graph.buffer("vc", (1, 1, 128, 64), "float16", role="state")
    ks = graph.buffer("ks", (1, 1, 128, 1), "float32", role="state")
    vs = graph.buffer("vs", (1, 1, 128, 1), "float32", role="state")
    out = graph.buffer("out", (256,), "float32", role="output")
    pos = graph.buffer("positions", (1,), "int32", role="input")
    enabled = graph.buffer("enabled", (1,), "int32", role="input")
    floating_attention(graph, p, q, kc, vc, ks, vs, out, pos, enabled)
    assert {"splits": 64} in search_policies(
        graph, None, provider=provider, target="sm_89"
    )
    graph, _ = lower(graph, None, {"splits": 64}, provider=provider, target="sm_89")
    assert [call.parameters["splits"] for call in graph.calls] == [64, 64]
    assert graph.buffers["out.parts"].shape == (4 * 64 * 66,)
    assert graph.buffers["out"].shape == (256,)


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("cap,enabled", [(19, True), (257, True), (31, False)])
def test_grouped_decode_partial_tail_scales_and_inactive(
    tmp_path, provider, cap, enabled
):
    if os.environ.get("AKAS_LFM_MOE_" + provider.upper()) != "1":
        pytest.skip("opt-in grouped decode qualification")
    import tensor
    from akas.compilation.selection import source
    from akas.compilation.emit import build_artifact
    from akas.kernels.registry import validation
    from tensor.artifacts.format import read_artifact

    p = dict(
        r=2,
        h=4,
        kh=1,
        d=64,
        capacity=cap,
        splits=16,
        chunk=1,
        kv_dtype="float16",
        provider=provider,
        optimized=True,
    )
    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    entry, artifact = tmp_path / "kernel.py", tmp_path / "kernel.tbin"
    entry.write_text(
        source(
            "attention_partial",
            p,
            contract="reference",
            provider=provider,
            target=target,
            preferred="attention.grouped_decode",
        ).render()
    )
    build_artifact(entry, artifact, provider=provider, target=target)
    meta, _ = read_artifact(artifact)
    case = validation("attention_partial", p, meta["arguments"])
    if not enabled:
        case.arguments["enabled"][:] = 0
        parts = case.expected["out"].reshape(2, 4, 16, 66)
        parts[:] = 0
        parts[..., 64] = -np.inf
    with tensor.Device(provider=provider) as device, ExitStack() as stack:
        buffers = {n: device.from_numpy(a) for n, a in case.arguments.items()}
        for buffer in buffers.values():
            stack.callback(buffer.release)
        module = stack.enter_context(device.load(artifact))
        module.launch(*(buffers[a["name"]] for a in meta["arguments"]))
        for name, buffer in buffers.items():
            expected = case.expected[name] if name == "out" else case.arguments[name]
            np.testing.assert_allclose(
                buffer.to_numpy(), expected, rtol=case.rtol, atol=case.atol
            )


@pytest.mark.parametrize(
    "chunk,cap,query_rows,key_rows,stages",
    [(4, 19, 32, 64, 2), (33, 97, 32, 64, 2), (33, 129, 64, 32, 3)],
)
def test_cuda_pipelined_attention_tail_and_inactive(
    tmp_path, chunk, cap, query_rows, key_rows, stages
):
    if os.environ.get("AKAS_LFM_MOE_CUDA") != "1":
        pytest.skip("opt-in pipelined attention qualification")
    import tensor
    from akas.compilation.selection import source
    from akas.compilation.emit import build_artifact
    from akas.kernels.registry import validation
    from tensor.artifacts.format import read_artifact

    p = dict(
        r=2 * chunk,
        h=4,
        kh=1,
        d=64,
        capacity=cap,
        chunk=chunk,
        kv_dtype="float16",
        provider="cuda",
        optimized=True,
        unit_scales=True,
        prefill_precision="float16",
        query_rows=query_rows,
        key_rows=key_rows,
        stages=stages,
        async_loads=True,
    )
    entry, artifact = tmp_path / "kernel.py", tmp_path / "kernel.tbin"
    entry.write_text(
        source(
            "attention",
            p,
            contract="reference",
            provider="cuda",
            target="sm_89",
            preferred="attention.cuda.pipelined",
        ).render()
    )
    build_artifact(entry, artifact, provider="cuda", target="sm_89")
    meta, _ = read_artifact(artifact)
    case = validation("attention", p, meta["arguments"])
    with tensor.Device(provider="cuda") as device, ExitStack() as stack:
        buffers = {n: device.from_numpy(a) for n, a in case.arguments.items()}
        for buffer in buffers.values():
            stack.callback(buffer.release)
        module = stack.enter_context(device.load(artifact))
        module.launch(*(buffers[a["name"]] for a in meta["arguments"]))
        for name, buffer in buffers.items():
            np.testing.assert_allclose(
                buffer.to_numpy(), case.expected[name], rtol=case.rtol, atol=case.atol
            )


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_prepared_kernel_timing_restores_mutable_state(tmp_path, provider):
    if os.environ.get("AKAS_LFM_MOE_" + provider.upper()) != "1":
        pytest.skip("opt-in prepared timing and state reset qualification")
    from akas.graph import Graph
    from akas.compilation.compile import produce
    from akas.compilation.tuning import SearchSettings

    graph = Graph(contract="reference", phase="prefill")
    layouts = [
        ("x", (2 * 4 * 3 * 64,), "float32", "input"),
        ("w", (64 * 3,), "float32", "weight"),
        ("state", (2, 2, 64), "float32", "state"),
        ("out", (2 * 4 * 64,), "float32", "output"),
        ("enabled", (2,), "int32", "input"),
    ]
    buffers = [graph.buffer(n, s, d, role=r) for n, s, d, r in layouts]
    graph("conv", dict(r=2, c=64, length=3, chunk=4), *buffers)
    result = produce(
        graph,
        tmp_path,
        provider=provider,
        search_settings=SearchSettings(budget=1, warmup=2, samples=2, repeats=2),
    )
    # Qualification checks both the returned output and the convolution state
    # after repeated replays; accumulated state would fail that reference.
    trials = result["tuning"]["operations"][0]["candidates"]
    assert trials and all(t["status"] == "passed" for t in trials)
    assert all(t["timing"]["execution"] == "prepared-plan" for t in trials)
