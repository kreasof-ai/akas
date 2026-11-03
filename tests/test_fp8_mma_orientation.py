"""Both GEMM orientations preserve independent FP8 operation references."""
import os
import shutil

import numpy as np
import pytest


@pytest.mark.skipif(os.getenv('AKAS_LFM_FP8_CUDA') != '1', reason='CUDA FP8 qualification')
@pytest.mark.parametrize('kind', ['expert_partial', 'expert_pair_partial',
                                  'fp8_expert_swiglu', 'projection_partial'])
def test_weight_major_fp8_matches_independent_reference(tmp_path, kind):
    import tensor
    from akas.kernels.definition import definition, define
    from akas.kernels.registry import validation
    from akas.compilation.selection import selected_source
    from tensor.compiler.frontend.signature import frontend_arguments

    p = dict(r=3, k=256, o=256, partitions=1)
    implementation = 'linear.pipeline:reference'
    schedule = dict(block_m=16, columns=128, threads=128, stages=2,
                    mma_dtype='float8_e4m3fn', mma_orientation='weight')
    if kind != 'projection_partial':
        p.update(rows=3, top=3, experts=7, compact=True, tile_map=True,
                 block_m=32, max_block_m=128)
        implementation = 'moe.cuda.fp8-bucketed:mapped'
    if kind == 'expert_pair_partial':
        p['partitions'] = 2
        implementation = 'moe.cuda.fp8-paired-partial:reference'
        schedule['pair_layout'] = 'wide'
    if kind == 'fp8_expert_swiglu':
        p.update(product_dtype='bfloat16', activation_dtype='bfloat16')
        implementation = 'moe.cuda.fp8-swiglu:mapped'
        schedule.pop('columns')
        schedule.update(pair_layout='wide')
    schedule['implementation'] = implementation
    spec = dict(kind=kind, parameters=p, contract='reference', phase='decode')
    caps = dict(provider='cuda', target='sm_89')
    impl, selected = definition(spec, schedule, caps)
    primitive = define(spec, selected, caps, impl.declaration())
    case = validation(kind, p, frontend_arguments(primitive, {}))
    source, artifact = tmp_path/'kernel.py', tmp_path/'kernel.tbin'
    source.write_text(selected_source(spec, selected, caps, impl).render())
    tensor.build(source, artifact, target='sm_89')
    with tensor.Device() as device, device.load(artifact) as executable:
        values = {a['name']: device.from_numpy(case.arguments[a['name']], dtype=a['dtype'])
                  for a in executable.manifest['arguments']}
        executable.launch(*(values[a['name']] for a in executable.manifest['arguments']))
        for name, expected in case.expected.items():
            actual = values[name].to_numpy()
            if name in case.comparators:
                case.comparators[name](actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, rtol=case.rtol, atol=case.atol)


@pytest.mark.skipif(os.getenv('TENSOR_NVRTC') != '1' or not shutil.which('nvdisasm'),
                    reason='GPU-free Hopper machine-code check')
def test_hopper_small_batch_weight_orientation_emits_native_fp8(tmp_path):
    import subprocess
    import zipfile
    import tensor
    from akas.kernels.definition import definition
    from akas.compilation.selection import selected_source

    spec = dict(kind='expert_partial', contract='reference', phase='decode',
        parameters=dict(r=1, rows=1, k=256, o=128, top=4, experts=64,
                        partitions=1, compact=True, tile_map=True, block_m=16))
    caps = dict(provider='cuda', target='sm_90a')
    impl, schedule = definition(spec, dict(implementation='moe.cuda.fp8-bucketed:mapped',
        block_m=16, columns=64, threads=128, stages=2, mma_orientation='weight'), caps)
    source, artifact = tmp_path/'kernel.py', tmp_path/'kernel.tbin'
    source.write_text(selected_source(spec, schedule, caps, impl).render())
    tensor.build(source, artifact, target='sm_90a')
    with zipfile.ZipFile(artifact) as archive:
        cubin = tmp_path/'kernel.cubin'
        cubin.write_bytes(archive.read('kernel.cubin'))
    sass = subprocess.check_output(['nvdisasm', str(cubin)], text=True)
    assert 'QGMMA.64x16x32.F32.E4M3.E4M3' in sass
    assert 'F2FP.F16.E4M3.UNPACK' not in sass
