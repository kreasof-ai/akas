"""Compressed staging preserves routed projection semantics across GGML formats."""

from contextlib import ExitStack
import os
import numpy as np
import pytest


def parameters(encoding, routed=False):
    return dict(
        r=5,
        top=2,
        experts=4,
        k=512,
        o=64,
        input_dtype="float32",
        round_inputs=True,
        routed_input=routed,
        tile_map=True,
        weight_parts=[dict(rows=256, format="ggml", options=dict(encoding=encoding))],
    )


def test_staging_capability_follows_storage_not_model():
    from akas.quantization import get
    from akas.quantization.weights import supports_parts

    for encoding in (0, 1, 2, 3, 8, 12, 13, 14):
        p = parameters(encoding)
        assert supports_parts(p["weight_parts"], 512, "stage_tile", provider="cuda")
    assert not get("weights", "ggml").supports(
        "stage_tile", (256, 512), provider="webgpu", encoding=2
    )
    assert not get("weights", "ggml").supports(
        "stage_tile", (256, 512), provider="cuda", encoding=2, layout="planar"
    )


@pytest.mark.parametrize("encoding", [0, 1, 2, 3, 8, 12, 13, 14])
@pytest.mark.parametrize("routed", [False, True])
def test_cuda_staged_experts_match_independent_reference(tmp_path, encoding, routed):
    if os.environ.get("AKAS_LFM_MOE_CUDA") != "1":
        pytest.skip("opt-in staged expert GPU qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.execution.plan import ExecutionPlan
    from akas.memory.transfer import write
    from akas.ops.moe.validation import encoded_validation

    p = parameters(encoding, routed)
    spec = dict(
        kind="quantized_experts", parameters=p, contract="reference", phase="verify"
    )
    axes = dict(block_m=16, columns=32, block_k=256, threads=128, stages=1)
    entry, artifact = tmp_path / "staged.py", tmp_path / "staged.tbin"
    entry.write_text(
        "from akas.ops.moe.cuda import staged_experts\n"
        "from akas.kernels.definition import KernelSpec\n"
        "def tensor_export():\n"
        f" return {{'kernel': staged_experts(KernelSpec(**{spec!r}), {axes!r})}}\n"
    )
    build_artifact(entry, artifact, provider="cuda", target="sm_89")
    metadata, _ = read_artifact(artifact)
    case = encoded_validation("quantized_experts", p, metadata["arguments"])
    with tensor.Device(provider="cuda") as device, ExitStack() as owned:
        buffers = {
            name: device.from_numpy(value) for name, value in case.arguments.items()
        }
        for buffer in buffers.values():
            owned.callback(buffer.release)
        module = owned.enter_context(device.load(artifact))
        plan = ExecutionPlan(device)
        plan.bind(module, tuple(buffers[a["name"]] for a in metadata["arguments"]))
        graph = owned.enter_context(
            plan.capture(resources=tuple(buffers.values()) + (module,))
        )
        graph.launch()
        np.testing.assert_allclose(
            buffers["out"].to_numpy(),
            case.expected["out"],
            rtol=case.rtol,
            atol=case.atol,
        )
        sentinel = np.full_like(case.arguments["out"], 1.25)
        write(device, buffers["out"], sentinel)
        write(device, buffers["counts"], np.zeros_like(case.arguments["counts"]))
        graph.launch()
        np.testing.assert_array_equal(buffers["out"].to_numpy(), sentinel)
