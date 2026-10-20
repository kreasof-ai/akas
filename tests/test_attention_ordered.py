"""Warp-local softmax preserves the established reduction and cache contracts."""

import os
import numpy as np
import pytest


def parameters():
    return dict(
        r=514,
        h=8,
        kh=2,
        d=64,
        capacity=512,
        chunk=257,
        kv_dtype="float16",
        unit_scales=True,
        staged_kv=True,
        optimized=True,
        provider="cuda",
        prefill_precision="float16",
    )


def test_ordered_attention_selection_requires_qualified_staging_and_geometry():
    from akas.kernels.registry import select, candidates

    p = parameters()
    assert (
        select(
            "attention", p, contract="reference", provider="cuda", target="sm_89"
        ).factory
        == "floating_ordered"
    )
    for changed in (
        dict(staged_kv=False),
        dict(unit_scales=False),
        dict(d=128),
        dict(chunk=32),
        dict(capacity=513),
        dict(kv_dtype="fp8"),
    ):
        assert not any(
            i.factory == "floating_ordered"
            for i in candidates(
                "attention",
                {**p, **changed},
                contract="reference",
                provider="cuda",
                target="sm_89",
            )
        )
    assert not any(
        i.factory == "floating_ordered"
        for i in candidates(
            "attention", p, contract="reference", provider="cuda", target="sm_80"
        )
    )
    assert not any(
        i.factory == "floating_ordered"
        for i in candidates(
            "attention",
            p,
            contract="reference",
            provider="webgpu",
            target="webgpu-portable-v1",
        )
    )


def test_ordered_attention_schedule_cannot_change_reduction_geometry():
    from akas.kernels.definition import definition

    spec = dict(
        kind="attention", parameters=parameters(), contract="reference", phase="prefill"
    )
    implementation, schedule = definition(
        spec,
        dict(query_rows=64, key_rows=64, threads=128, stages=1),
        dict(provider="cuda", target="sm_89"),
    )
    assert implementation.factory == "floating_ordered"
    assert schedule["query_rows"] == 64
    with pytest.raises(ValueError, match="undeclared kernel axes"):
        definition(spec, dict(unknown=True), dict(provider="cuda", target="sm_89"))


def test_registered_ordered_attention_gpu_f64_reference_bitwise_tails_and_masks(
    tmp_path,
):
    if os.environ.get("AKAS_LFM_MOE_CUDA") != "1":
        pytest.skip("opt-in ordered CUDA attention qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import selected_source
    from akas.kernels.definition import definition
    from akas.kernels.registry import validation
    from akas.memory.transfer import write

    p = parameters()
    artifacts = {}
    for name, preferred, axes in (
        ("ordered", "attention.cuda.ordered", {}),
        (
            "reference",
            "attention.cuda.pipelined",
            dict(query_rows=32, key_rows=64, threads=128, stages=1),
        ),
    ):
        spec = dict(
            kind="attention", parameters=p, contract="reference", phase="prefill"
        )
        capabilities = dict(provider="cuda", target="sm_89")
        implementation, schedule = definition(
            spec, axes, capabilities, preferred=preferred
        )
        source = tmp_path / (name + ".py")
        artifact = source.with_suffix(".tbin")
        source.write_text(
            selected_source(spec, schedule, capabilities, implementation).render()
        )
        build_artifact(source, artifact, provider="cuda", target="sm_89")
        artifacts[name] = artifact
    signature, _ = read_artifact(artifacts["ordered"])
    case = validation("attention", p, signature["arguments"])
    # First slot has a one-row tail, now inactive. The second slot is empty.
    case.arguments["enabled"][256:] = 0
    case.arguments["q"].reshape(514, -1)[256:] = np.nan
    case.expected["out"].reshape(514, -1)[256:] = 0
    for name in ("kc", "vc"):
        case.arguments[name][0, :, 511:] = 0  # staging owns finite future padding
        case.arguments[name][1] = np.nan  # an empty slot must never read its cache
    with tensor.Device(provider="cuda") as device:
        buffers = {
            name: device.from_numpy(value) for name, value in case.arguments.items()
        }
        try:
            results = {}
            for name, artifact in artifacts.items():
                with device.load(artifact) as module:
                    write(
                        device,
                        buffers["out"],
                        np.full_like(case.arguments["out"], np.nan),
                    )
                    args = tuple(
                        buffers[a["name"]] for a in module.manifest["arguments"]
                    )
                    module.launch(*args)
                    actual = buffers["out"].to_numpy()
                    np.testing.assert_allclose(
                        actual, case.expected["out"], rtol=0.002, atol=0.0002
                    )
                    for field, original in case.arguments.items():
                        if field != "out":
                            np.testing.assert_array_equal(
                                buffers[field].to_numpy(), original
                            )
                    results[name] = actual.copy()
                    write(
                        device,
                        buffers["enabled"],
                        np.zeros_like(case.arguments["enabled"]),
                    )
                    module.launch(*args)
                    np.testing.assert_array_equal(
                        buffers["out"].to_numpy(), np.zeros_like(actual)
                    )
                    write(device, buffers["enabled"], case.arguments["enabled"])
            np.testing.assert_array_equal(
                results["ordered"].view("uint32"), results["reference"].view("uint32")
            )
        finally:
            for buffer in buffers.values():
                buffer.release()
