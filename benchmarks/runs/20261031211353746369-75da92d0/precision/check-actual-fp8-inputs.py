from pathlib import Path
import numpy as np
from akas.models.lfm2.weights import LFM2MoECheckpoint
from support.native_fp8_reference import NativeFP8Reference
from akas.quantization._shared.e4m3 import encode_storage,decode_storage
root=Path('build/lfm-serving');g=np.load(root/'gpu-current-trace.npz')
cp=LFM2MoECheckpoint('build/checkpoints/lfm2.5-24b-a2b-fp8',tokenizer=False);ref=NativeFP8Reference(cp,projection_half=True)
for part,sourcename in [('gate','ffn_normalized.view.603'),('up','ffn_normalized.view.606'),('down','activated.view.610')]:
 key='pipeline.blk.0.'+sourcename
 bits=g[key+'.bits'];scales=g[key+'.scales']
 x=g['blk.0.'+('activated' if part=='down' else 'ffn_normalized')].reshape(bits.shape)
 blocks=x.reshape(len(x),-1,128)
 expected_scale=np.maximum(np.max(np.abs(blocks),axis=-1),np.float32(1e-12))/np.float32(448)
 expected_bits=encode_storage(blocks/expected_scale[...,None]).reshape(bits.shape)
 print(part,'bit differences',np.count_nonzero(bits!=expected_bits),'scale maxerr',np.max(np.abs(scales-expected_scale)),flush=True)
 source=decode_storage(bits)*np.repeat(scales,128,axis=-1)
 expected=(source.astype('float64')@ref.weights[f'blk.0.ffn_{part}.weight'].T).astype('float32')
 actual=g['blk.0.'+('ffn_output' if part=='down' else part)].reshape(expected.shape)
 print(part,'actual input matmul relative',np.linalg.norm(expected-actual)/np.linalg.norm(expected),'max',np.max(np.abs(expected-actual)),flush=True)
print('GATE_UP_SAME_BITS',np.array_equal(g['pipeline.blk.0.ffn_normalized.view.603.bits'],g['pipeline.blk.0.ffn_normalized.view.606.bits']))
