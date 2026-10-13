"""Consolidated activations retain the old kernels' numerical and launch ABI."""

import importlib
import os
import sys
import numpy as np
import pytest

from activation_parity import source


@pytest.mark.parametrize("provider,contract,kind", [
    ("cuda","packed","swiglu"),
    ("cuda","bf16","swiglu"),
    ("cuda","block-fp8","swiglu"),
    ("cuda","block-fp8","swiglu_experts"),
    ("webgpu","packed","swiglu"),
])
def test_before_after_activation_storage_and_math(tmp_path, monkeypatch, provider, contract, kind):
    if os.environ.get("AKAS_FORWARD_"+provider.upper()) != "1":
        pytest.skip("enable provider for native activation parity")
    import tensor
    from tensor.compiler.frontend.source import export_source
    from tensor.compiler.tuning import measure
    from akas.ops.registry import select
    from akas.compilation.emit import build_artifact

    helper = tmp_path/'activation_baseline_helper.py'
    helper.write_text(source('activation.py'))
    old_name = 'bf16_batch' if contract=='bf16' else 'decode' if contract=='block-fp8' else 'packed_baseline'
    module_name = 'activation_baseline_'+old_name
    (tmp_path/(module_name+'.py')).write_text(source('cuda/'+old_name+'.py').replace('akas.ops.ffn.activation','activation_baseline_helper'))
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in (module_name,'activation_baseline_helper'):
        sys.modules.pop(name,None)
    importlib.import_module(module_name)
    params = dict(r=3,c=257,top=2)
    if contract=='bf16': params = dict(slots=1,chunk=3,c=257)
    variant = select(kind,params,contract=contract,phase='decode' if contract=='block-fp8' else 'forward',provider=provider,target='sm_89' if provider=='cuda' else 'webgpu-portable-v1')
    definitions = [(module_name,kind,params,('activation_baseline_helper',)),
                   ('akas.ops.'+variant.module,*variant.source_arguments(params),())]
    artifacts = []
    for index,(module,operation,parameters,dependencies) in enumerate(definitions):
        path = tmp_path/f'kernel-{index}.py'
        path.write_text(export_source(module,'make_kernel',operation,parameters,dependencies=dependencies))
        image = path.with_suffix('.tbin')
        build_artifact(path,image,target='sm_89' if provider=='cuda' else 'webgpu-portable-v1',provider=provider)
        artifacts.append(image)
    with tensor.Device(provider=provider) as device:
        kernels = [device.load(path) for path in artifacts]
        assert kernels[0].manifest['arguments'] == kernels[1].manifest['arguments']
        assert kernels[0].manifest['launch']['grid'] == kernels[1].manifest['launch']['grid']
        rng = np.random.default_rng(81)
        inputs = [device.from_numpy(rng.normal(0,1,tuple(spec['shape'])).astype('float32')) for spec in kernels[0].manifest['arguments'][:-1]]
        output = kernels[0].manifest['arguments'][-1]
        outputs = [device.empty(output['shape'],output['dtype']) for _ in kernels]
        for kernel,out in zip(kernels,outputs):
            kernel.launch(*inputs,out)
        np.testing.assert_array_equal(outputs[0].to_numpy(),outputs[1].to_numpy())
        timings = [measure(device,lambda kernel=k,out=o:kernel.launch(*inputs,out),warmup=3,samples=7,repeats=10) for k,o in zip(kernels,outputs)]
        assert all(item['median_seconds'] > 0 for item in timings)
