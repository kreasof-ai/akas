from support.migration import historical_source
"""Consolidated normalization retains each numerical and reduction policy."""

import importlib
import os
import sys

import numpy as np
import pytest

from normalization_parity import source


@pytest.mark.parametrize("provider,contract,kind,old,parameters", [
    ("cuda", "packed", "rms", "cuda/packed_baseline.py", dict(r=4,c=2048,eps=1e-5)),
    ("cuda", "packed", "add_rms", "cuda/packed.py", dict(r=4,c=2048,eps=1e-5)),
    ("cuda", "bf16", "rms", "cuda/decode.py", dict(r=4,c=2048,eps=1e-6)),
    ("cuda", "bf16", "add_rms", "cuda/decode.py", dict(r=4,c=2048,eps=1e-6)),
    ("cuda", "reference", "rms", "reference.py", dict(r=3,c=257,eps=1e-5)),
    ("webgpu", "reference", "rms", "reference.py", dict(r=3,c=257,eps=1e-5)),
    ("webgpu", "packed", "add_rms", "webgpu/packed.py", dict(r=4,c=2048,eps=1e-5)),
    ("cuda", "packed", "kvnorm", "cuda/packed.py", dict(r=4,h=4,kh=2,d=64,cap=32,eps=1e-5,theta=1e6)),
    ("webgpu", "packed", "kvnorm", "webgpu/packed.py", dict(r=4,h=4,kh=2,d=64,cap=32,eps=1e-5,theta=1e6)),
])
def test_normalization_before_after(tmp_path, monkeypatch, provider, contract, kind, old, parameters):
    if os.environ.get("AKAS_FORWARD_" + provider.upper()) != "1":
        pytest.skip("enable hardware normalization parity")
    import tensor
    from support.migration import source as export_source
    from tensor.compiler.tuning import measure
    from akas.compilation.emit import build_artifact
    from akas.kernels.registry import select
    from akas.kernels.contracts import sample_arguments

    helper = "normalization_baseline_helper"
    module = "normalization_baseline_" + old.replace("/", "_").removesuffix(".py")
    (tmp_path / (helper + ".py")).write_text(source("kernels.py"))
    (tmp_path / (module + ".py")).write_text(source(old).replace("akas.ops.normalization.kernels", helper))
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in (helper, module):
        sys.modules.pop(name, None)
    importlib.import_module(module)
    target = "sm_89" if provider == "cuda" else "webgpu-portable-v1"
    variant = select(kind, parameters, provider=provider, target=target, contract=contract)
    definitions = [(module, (kind, parameters), (helper,)),
                   (historical_source(variant, parameters)["module"], historical_source(variant, parameters)["args"], ())]
    artifacts = []
    for index, (name, args, dependencies) in enumerate(definitions):
        path = tmp_path / f"kernel-{index}.py"
        path.write_text(export_source(name, "make_kernel", *args, dependencies=dependencies))
        artifact = path.with_suffix(".tbin")
        build_artifact(path, artifact, provider=provider, target=target)
        artifacts.append(artifact)
    with tensor.Device(provider=provider) as device:
        kernels = [device.load(path) for path in artifacts]
        assert kernels[0].manifest["arguments"] == kernels[1].manifest["arguments"]
        assert kernels[0].manifest["launch"]["grid"] == kernels[1].manifest["launch"]["grid"]
        initial = sample_arguments(kernels[0].manifest["arguments"])
        if "pos" in initial:
            initial["pos"][:] = [3, 3]
        buffers = [[device.from_numpy(initial[spec["name"]], dtype="bfloat16")
                    if spec["dtype"] == "bfloat16" else device.from_numpy(initial[spec["name"]])
                    for spec in kernel.manifest["arguments"]] for kernel in kernels]
        for kernel, arguments in zip(kernels, buffers):
            kernel.launch(*arguments)
        for before, after in zip(*buffers):
            np.testing.assert_array_equal(before.to_numpy(), after.to_numpy())
        timings = [measure(device, lambda kernel=kernel, arguments=arguments: kernel.launch(*arguments),
                           warmup=3, samples=7, repeats=10)
                   for kernel, arguments in zip(kernels, buffers)]
        (tmp_path / "timings.json").write_text(__import__("json").dumps(timings, indent=2))
        assert all(row["median_seconds"] > 0 for row in timings)
