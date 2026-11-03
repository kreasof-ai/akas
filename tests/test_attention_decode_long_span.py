"""Single-split decode must preserve the online softmax across full tile groups."""
from contextlib import ExitStack
import os
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('AKAS_LFM_MOE_CUDA')!='1',reason='CUDA qualification')
@pytest.mark.parametrize('stages',[0,2])
@pytest.mark.parametrize('mapped',[False,True])
def test_decode_full_groups(tmp_path,stages,mapped):
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.compilation.state import mapped_source
    from akas.kernels.registry import validation
    from akas.kernels.contracts import mapped_validation
    p=dict(r=2,h=8,kh=2,d=64,capacity=2176,splits=1,kv_dtype='float16',
           unit_scales=True,provider='cuda',optimized=True,key_rows=128,stages=stages,
           decode_precision='float16',copy_complete_tiles=True)
    selected=source('attention_partial',p,contract='reference',target='sm_89',
                    preferred='attention.cuda.mma_decode')
    mapping=((1,2),2,4)
    if mapped:selected=mapped_source(selected,mapping)
    entry=tmp_path/'kernel.py';artifact=tmp_path/'kernel.tbin';entry.write_text(selected.render())
    build_artifact(entry,artifact,target='sm_89')
    manifest,_=read_artifact(artifact)
    case=(mapped_validation('attention_partial',p,manifest['arguments'],mapping) if mapped
          else validation('attention_partial',p,manifest['arguments']))
    with tensor.Device() as device,ExitStack() as stack:
        buffers={name:device.from_numpy(a) for name,a in case.arguments.items()}
        for buffer in buffers.values():stack.callback(buffer.release)
        module=stack.enter_context(device.load(artifact))
        module.launch(*(buffers[a['name']] for a in manifest['arguments']))
        np.testing.assert_allclose(buffers['out'].to_numpy(),case.expected['out'],
                                   rtol=case.rtol,atol=case.atol)


@pytest.mark.skipif(os.environ.get('AKAS_LFM_MOE_CUDA')!='1',reason='CUDA qualification')
def test_rotary_short_context_precision(tmp_path):
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.compilation.emit import build_artifact
    from akas.compilation.selection import source
    from akas.kernels.registry import validation
    p=dict(r=256,h=32,d=64,theta=10000000.,provider='cuda',position_limit=2176)
    entry=tmp_path/'rope.py';artifact=tmp_path/'rope.tbin'
    entry.write_text(source('rope',p,contract='reference',target='sm_89').render())
    build_artifact(entry,artifact,target='sm_89');manifest,_=read_artifact(artifact)
    case=validation('rope',p,manifest['arguments'])
    with tensor.Device() as device,ExitStack() as stack:
        buffers={name:device.from_numpy(a) for name,a in case.arguments.items()}
        for buffer in buffers.values():stack.callback(buffer.release)
        module=stack.enter_context(device.load(artifact))
        module.launch(*(buffers[a['name']] for a in manifest['arguments']))
        np.testing.assert_allclose(buffers['out'].to_numpy(),case.expected['out'],
                                   rtol=case.rtol,atol=case.atol)
