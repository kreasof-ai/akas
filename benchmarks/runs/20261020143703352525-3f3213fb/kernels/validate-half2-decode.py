import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from akas.kernels.registry import validation
from akas.quantization.weights.ggml.validation import decode_rows
from akas.benchmarking.history import Recorder
root=Path('build/lfm-moe/half2-decode')
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-correctness-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),weight_format='q4_0',operand_dtype='float16',accumulator_dtype='float32',qualification='Independent per-lane continuous F32 sum oracle, odd output tail and bitwise qualified GPU reference for full expert shapes; no performance claim')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Decode Q4 expert coefficients with adjacent half2 arithmetic while preserving continuous F32 dot order')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
try:
 with tensor.Device(provider='cuda') as device:
  for depth,columns,routed in ((2048,7,False),(2048,1536,False),(1536,2048,True)):
   p=dict(r=1,top=4,experts=64,k=depth,o=columns,input_dtype='float32',routed_input=routed,round_inputs=True,weight_parts=({'rows':64*columns,'format':'ggml','options':{'encoding':2}},))
   metadata,_=read_artifact(root/f'{depth}-{columns}-128.tbin');case=validation('quantized_experts',p,metadata['arguments'])
   buffers={n:device.from_numpy(a) for n,a in case.arguments.items()}
   expected=None
   if columns==7:
    expected=np.zeros((1,4,columns),dtype='float32');lane=np.arange(32)
    for rank,expert in enumerate(case.arguments['ids'][0]):
     weight=decode_rows(case.arguments['w'],2,depth,int(expert)*columns,(int(expert)+1)*columns).astype('float16').astype('float32')
     activation=case.arguments['x'][0].astype('float16').astype('float32');total=np.zeros((columns,32),dtype='float32')
     for tile in range(depth//256):
      base=tile*256+lane//4*32+lane%4*4
      for offset in (0,1,2,3,16,17,18,19):
       index=base+offset;total+=activation[index]*weight[:,index]
     for step in (16,8,4,2,1):total[:,:step]+=total[:,step:step*2]
     expected[0,rank]=total[:,0]
   else:
    with device.load(f'build/lfm-moe/packed-view/{depth}-128.tbin') as kernel:
     kernel.launch(*[buffers[a['name']] for a in kernel.manifest['arguments']]);expected=buffers['out'].to_numpy()
   for threads in (128,256):
    path=root/f'{depth}-{columns}-{threads}.tbin'
    with device.load(path) as kernel:
     kernel.launch(*[buffers[a['name']] for a in kernel.manifest['arguments']]);actual=buffers['out'].to_numpy()
     np.testing.assert_array_equal(actual,expected)
    checks.append(dict(depth=depth,columns=columns,routed_input=routed,threads=threads,bitwise=True));print('BITWISE_PASS',checks[-1],flush=True)
   for buffer in buffers.values():buffer.release()
 passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 for path in (__file__,'build/lfm-moe/half2_decode.py','build/lfm-moe/q4_word_tile.py','build/lfm-moe/compile-half2-decode.py'):
  rec.artifact(path,'kernels/'+Path(path).name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
