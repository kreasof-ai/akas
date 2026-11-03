"""Warp batching preserves the independent block-128 quantization reference."""

import os
import numpy as np
import pytest


@pytest.mark.skipif(
    os.getenv("AKAS_LFM_FP8_CUDA") != "1", reason="native FP8 CUDA qualification"
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("top", [None, 4])
def test_warp_quantization_preserves_codes_and_scales(tmp_path, dtype, top):
    import tensor
    from akas.kernels.definition import definition, define
    from akas.kernels.registry import validation
    from akas.compilation.selection import selected_source
    from tensor.compiler.frontend.signature import frontend_arguments

    parameters = dict(r=3, k=384, input_dtype=dtype, **({"top": top} if top else {}))
    spec = dict(kind="quantize", parameters=parameters, contract="reference")
    caps = dict(provider="cuda", target="sm_89")
    impl, schedule = definition(
        spec, dict(implementation="fp8.cuda.warp_quantize", groups_per_block=4), caps
    )
    primitive = define(spec, schedule, caps, impl.declaration())
    case = validation("quantize", parameters, frontend_arguments(primitive, {}))
    source, artifact = tmp_path / "kernel.py", tmp_path / "kernel.tbin"
    source.write_text(selected_source(spec, schedule, caps, impl).render())
    tensor.build(source, artifact, target="sm_89")
    with tensor.Device() as device, device.load(artifact) as executable:
        values = {
            a["name"]: device.from_numpy(case.arguments[a["name"]], dtype=a["dtype"])
            for a in executable.manifest["arguments"]
        }
        executable.launch(
            *(values[a["name"]] for a in executable.manifest["arguments"])
        )
        for name, expected in case.expected.items():
            np.testing.assert_array_equal(values[name].to_numpy(), expected)


@pytest.mark.parametrize("rows", [8192, 32768, 65536])
def test_warp_quantization_accepts_inputs_larger_than_256_mib(rows):
    pytest.importorskip("tilelang")
    from akas.kernels.definition import definition, define

    spec = dict(
        kind="quantize",
        parameters=dict(r=rows, k=11776, input_dtype="float32"),
        contract="reference",
    )
    caps = dict(provider="cuda", target="sm_90a")
    impl, schedule = definition(
        spec, dict(implementation="fp8.cuda.warp_quantize", groups_per_block=4), caps
    )
    primitive = define(spec, schedule, caps, impl.declaration())
    assert tuple(int(v) for v in primitive.buffer_map[primitive.params[0]].shape) == (
        rows,
        11776,
    )
