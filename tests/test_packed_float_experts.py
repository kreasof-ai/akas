"""Packed expert products retain declared F32 operands on both providers."""

import os
import numpy as np
import pytest


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_f32_dot_capability_is_codec_owned_and_geometry_checked(provider):
    from akas.quantization.formats import get
    from akas.quantization.weights import supports_parts

    codec = get("weights", "ggml")
    assert codec.supports("dot_float", (222, 256), provider=provider, encoding=2)
    assert not codec.supports("dot_float", (222, 128), provider=provider, encoding=2)
    assert not codec.supports("dot_float", (222, 256), provider=provider, encoding=3)
    part = dict(rows=222, format="ggml", options=dict(encoding=2))
    assert supports_parts((part,), 256, "dot_float", provider=provider)
    assert not supports_parts((part, part), 256, "dot_float", provider=provider)
    assert not supports_parts(
        ({**part, "row_order": (0, 1)},), 256, "dot_float", provider=provider
    )


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
@pytest.mark.parametrize("routed", [False, True])
def test_registered_packed_f32_operation_against_independent_reference(
    tmp_path, provider, routed
):
    if os.environ.get("AKAS_LFM_MOE_" + provider.upper()) != "1":
        pytest.skip("opt-in packed F32 GPU qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.selection import source
    from akas.compilation.emit import build_artifact
    from akas.kernels.registry import select, validation
    from akas.memory.transfer import write

    parameters = dict(
        r=1,
        top=4,
        experts=6,
        k=256,
        o=37,
        input_dtype="float32",
        round_inputs=False,
        routed_input=routed,
        weight_parts=({"rows": 222, "format": "ggml", "options": {"encoding": 2}},),
    )
    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    selected = select(
        "quantized_experts",
        parameters,
        contract="reference",
        provider=provider,
        target=target,
    )
    assert selected.factory == "packed_float_experts"
    entry, artifact = tmp_path / "expert.py", tmp_path / "expert.tbin"
    entry.write_text(
        source(
            "quantized_experts",
            parameters,
            contract="reference",
            provider=provider,
            target=target,
        ).render()
    )
    build_artifact(entry, artifact, provider=provider, target=target)
    signature, _ = read_artifact(artifact)
    case = validation("quantized_experts", parameters, signature["arguments"])
    with tensor.Device(provider=provider) as device:
        buffers = {
            name: device.from_numpy(value) for name, value in case.arguments.items()
        }
        try:
            with device.load(artifact) as module:
                args = tuple(buffers[a["name"]] for a in module.manifest["arguments"])
                module.launch(*args)
                np.testing.assert_allclose(
                    buffers["out"].to_numpy(),
                    case.expected["out"],
                    rtol=2e-5,
                    atol=2e-6,
                )
                for name, expected in case.arguments.items():
                    if name != "out":
                        np.testing.assert_array_equal(
                            buffers[name].to_numpy(), expected
                        )
                inactive = case.arguments["counts"].copy()
                inactive[case.arguments["ids"][0, -1]] = 0
                write(device, buffers["counts"], inactive)
                module.launch(*args)
                np.testing.assert_array_equal(
                    buffers["out"].to_numpy()[0, -1], np.zeros(37, dtype="float32")
                )
                write(device, buffers["counts"], np.zeros_like(inactive))
                module.launch(*args)
                np.testing.assert_array_equal(
                    buffers["out"].to_numpy(), np.zeros_like(case.expected["out"])
                )
        finally:
            for buffer in buffers.values():
                buffer.release()
