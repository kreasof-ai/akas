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
    from akas.kernels.registry import validation
    from tensor.artifacts.format import read_artifact

    parameters = {**parameters, "round_inputs": provider == "cuda"}
    if kind == "rope":
        parameters["provider"] = provider
    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
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
            columns=columns,
            block_k=block_k,
            block_m=32,
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
