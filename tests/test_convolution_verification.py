"""Speculative convolution retains every prefix and commits only accepted inputs."""

from contextlib import ExitStack
import os
import numpy as np
import pytest


def signature(rows=2, channels=7, chunk=3):
    p = dict(r=rows, c=channels, length=3, chunk=chunk)
    fields = [
        ("x", [rows * chunk * 3 * channels], "float32"),
        ("w", [channels * 3], "float32"),
        ("state", [rows, 2, channels], "float32"),
        ("journal", [rows, chunk, 2, channels], "float32"),
        ("out", [rows * chunk * channels], "float32"),
        ("enabled", [rows], "int32"),
    ]
    return p, [dict(name=n, shape=s, dtype=d) for n, s, d in fields]


def test_convolution_verification_reference_retains_state_and_every_history_prefix():
    from akas.ops.convolution.validation import verification_validation

    p, fields = signature()
    case = verification_validation("conv_verify", p, fields)
    np.testing.assert_array_equal(case.expected["state"], case.arguments["state"])
    x = case.arguments["x"].reshape(2, 3, 3, 7)
    expected = case.expected["journal"]
    # The first history slot shifts once, then is entirely replaced by inputs.
    np.testing.assert_array_equal(expected[0, 0, 0], case.arguments["state"][0, 1])
    np.testing.assert_array_equal(expected[0, 1, 0], (x[0, 0, 0] * x[0, 0, 2]))
    np.testing.assert_array_equal(expected[0, 2, 1], (x[0, 2, 0] * x[0, 2, 2]))
    # An inactive suffix retains the last live prefix and emits zero output.
    np.testing.assert_array_equal(expected[1, 2], expected[1, 1])
    np.testing.assert_array_equal(case.expected["out"].reshape(2, 3, 7)[1, 2], 0)


def test_convolution_commit_reference_covers_full_empty_and_partial_acceptance():
    from akas.ops.convolution.validation import verification_validation

    p, _ = signature(rows=3)
    fields = [
        dict(name=n, shape=s, dtype=d)
        for n, s, d in [
            ("state", [3, 2, 7], "float32"),
            ("journal", [3, 3, 2, 7], "float32"),
            ("accepted", [3], "int32"),
        ]
    ]
    case = verification_validation("conv_commit", p, fields)
    np.testing.assert_array_equal(case.arguments["accepted"], [3, 0, 2])
    np.testing.assert_array_equal(
        case.expected["state"][0], case.arguments["journal"][0, 2]
    )
    np.testing.assert_array_equal(case.expected["state"][1], case.arguments["state"][1])
    np.testing.assert_array_equal(
        case.expected["state"][2], case.arguments["journal"][2, 1]
    )


@pytest.mark.parametrize("provider", ["cuda", "webgpu"])
def test_convolution_verification_gpu_bitwise_prefix_commit_and_poisoned_inactive(
    tmp_path, provider
):
    if os.environ.get("AKAS_LFM_MOE_" + provider.upper()) != "1":
        pytest.skip("opt-in convolution verification qualification")
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import selected_source
    from akas.kernels.definition import definition
    from akas.memory.transfer import write
    from akas.ops.convolution.validation import verification_validation

    p, _ = signature()
    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    artifacts = {}
    for kind in ("conv", "conv_verify", "conv_commit"):
        spec = dict(kind=kind, parameters=p, contract="reference", phase="verify")
        capabilities = dict(provider=provider, target=target)
        implementation, schedule = definition(spec, {}, capabilities)
        source = tmp_path / (kind + ".py")
        source.write_text(
            selected_source(spec, schedule, capabilities, implementation).render()
        )
        build_artifact(
            source, source.with_suffix(".tbin"), provider=provider, target=target
        )
        artifacts[kind] = source.with_suffix(".tbin")
    fields, _ = read_artifact(artifacts["conv_verify"])
    case = verification_validation("conv_verify", p, fields["arguments"])
    case.arguments["enabled"][1] = 0
    case.arguments["x"].reshape(2, 3, 3, 7)[1] = np.nan
    case.expected["out"].reshape(2, 3, 7)[1] = 0
    case.expected["journal"][1] = case.arguments["state"][1]
    with tensor.Device(provider=provider) as device, ExitStack() as owned:
        buffers = {n: device.from_numpy(a) for n, a in case.arguments.items()}
        original = device.from_numpy(case.arguments["state"])
        old_state = device.from_numpy(case.arguments["state"])
        old_out = device.empty((2 * 3 * 7,), "float32")
        accepted = device.empty((2,), "int32")
        for b in [*buffers.values(), original, old_state, old_out, accepted]:
            owned.callback(b.release)
        modules = {n: device.load(path) for n, path in artifacts.items()}
        for module in modules.values():
            owned.callback(module.release)
        module = modules["conv_verify"]
        module.launch(*(buffers[a["name"]] for a in module.manifest["arguments"]))
        actual = buffers["out"].to_numpy()
        journal = buffers["journal"].to_numpy()
        np.testing.assert_allclose(
            actual, case.expected["out"], rtol=0.002, atol=0.0002
        )
        np.testing.assert_array_equal(
            buffers["state"].to_numpy(), case.arguments["state"]
        )
        np.testing.assert_array_equal(
            journal[:, :, 1], case.expected["journal"][:, :, 1]
        )
        for count in range(4):
            old_state.copy_from(original)
            buffers["state"].copy_from(original)
            write(device, buffers["enabled"], np.array([count, 0], dtype="int32"))
            mapping = dict(buffers, state=old_state, out=old_out)
            module = modules["conv"]
            module.launch(*(mapping[a["name"]] for a in module.manifest["arguments"]))
            write(device, accepted, np.array([count, 0], dtype="int32"))
            mapping = dict(buffers, accepted=accepted)
            module = modules["conv_commit"]
            module.launch(*(mapping[a["name"]] for a in module.manifest["arguments"]))
            np.testing.assert_array_equal(
                buffers["state"].to_numpy().view("uint32"),
                old_state.to_numpy().view("uint32"),
            )
            if count:
                np.testing.assert_array_equal(
                    actual.reshape(2, 3, 7)[0, :count].view("uint32"),
                    old_out.to_numpy().reshape(2, 3, 7)[0, :count].view("uint32"),
                )
        np.testing.assert_array_equal(buffers["journal"].to_numpy(), journal)
