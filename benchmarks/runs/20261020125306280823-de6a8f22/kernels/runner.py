"""Independent FP8-to-half tensor-core attention check and prepared-graph latency."""
import json,time,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from akas.benchmarking.history import Recorder,write
from akas.quantization._shared.e4m3 import decode_storage
root=Path('build/lfm-moe/fp8-mma')
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),kv_dtype='fp8',dequant_dtype='float16',query_dtype='float16',probability_dtype='float16',accumulator_dtype='float32',qualification='Independent NumPy FP8 cache decoding, per-position scales, FP16 staging and probability rounding; short/tail/full context and disabled requests; latency is hot-cache microbenchmark, not model throughput')
rec=Recorder('benchmarks',meta,kind='diagnostic',change='Prototype FP8 KV unpack-to-half grouped tensor-core decode')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
try:
 metadata,_=read_artifact(root/'64-1.tbin');signature=metadata['arguments'];rng=np.random.default_rng(123)
 arrays={a['name']:np.zeros(a['shape'],dtype=a['dtype']) for a in signature}
 arrays['q'][:]=rng.normal(0,.2,arrays['q'].shape)
 for name in ('kc','vc'):
  period=rng.integers(0,80,1048573,dtype='uint8');period|=rng.integers(0,2,1048573,dtype='uint8')<<7
  arrays[name][:]=np.resize(period,arrays[name].shape)
 for name in ('ks','vs'):arrays[name][:]=rng.uniform(.005,.05,arrays[name].shape)
 def reference(length,bk,enabled):
  out=np.zeros((32,16,66),dtype='float64');out[:,:,64]=-np.inf
  if not enabled:return out.astype('float32').ravel()
  span=((length+bk-1)//bk+15)//16*bk
  for head in range(8):
   keys=(decode_storage(arrays['kc'][0,head,:length])*arrays['ks'][0,head,:length]).astype('float16').astype('float64')
   vals=(decode_storage(arrays['vc'][0,head,:length])*arrays['vs'][0,head,:length]).astype('float16').astype('float64')
   queries=arrays['q'][0].reshape(32,64)[head*4:head*4+4].astype('float16').astype('float64')
   for part in range(16):
    maximum=np.full(4,-np.inf);norm=np.zeros(4);acc=np.zeros((4,64))
    for begin in range(part*span,min(length,(part+1)*span),bk):
     end=min(length,begin+bk);scores=(queries@keys[begin:end].T)*.125
     nextmax=np.maximum(maximum,scores.max(-1));correction=np.exp(maximum-nextmax)
     prob=np.exp(scores-nextmax[:,None]);acc=acc*correction[:,None]+prob.astype('float16').astype('float64')@vals[begin:end]
     norm=norm*correction+prob.sum(-1);maximum=nextmax
    out[head*4:head*4+4,part,:64]=acc;out[head*4:head*4+4,part,64]=maximum;out[head*4:head*4+4,part,65]=norm
  return out.astype('float32').ravel()
 with tensor.Device(provider='cuda') as device:
  buffers={name:device.from_numpy(a) for name,a in arrays.items()}
  from akas.memory.transfer import write as upload
  for bk,stages in ((64,1),(64,2),(128,1),(128,2)):
   path=root/f'{bk}-{stages}.tbin';metadata,_=read_artifact(path)
   with device.load(path) as kernel:
    for length,enabled in ((32,1),(65,1),(8191,1),(131072,1),(131072,0)):
     arrays['positions'][:]=length-1;arrays['enabled'][:]=enabled
     upload(device,buffers['positions'],arrays['positions']);upload(device,buffers['enabled'],arrays['enabled'])
     expected=reference(length,bk,enabled)
     kernel.launch(*[buffers[a['name']] for a in metadata['arguments']]);actual=buffers['out'].to_numpy().ravel()
     np.testing.assert_allclose(actual,expected,rtol=.002,atol=.0002)
     # Time existing graph API, identical buffers; resets and validation excluded.
     check=dict(key_rows=bk,stages=stages,length=length,enabled=enabled,correct=True,max_abs_error=float(np.max(np.abs(actual[np.isfinite(expected)]-expected[np.isfinite(expected)]))))
     if enabled:
      from akas.execution.plan import ExecutionPlan
      from tensor.compiler.tuning import measure
      plan=ExecutionPlan(device);plan.bind(kernel,tuple(buffers[a['name']] for a in metadata['arguments']))
      prepared=plan.capture(resources=(kernel,*buffers.values()))
      try:check['timing']=measure(device,prepared.launch,warmup=2,samples=5,repeats=5)
      finally:prepared.close()
     checks.append(check);print('PASS',json.dumps(check),flush=True)
   rec.artifact(root/f'{bk}-{stages}.py',f'kernels/{bk}-{stages}.py')
  for buffer in buffers.values():buffer.release()
 meta['checks']=checks;passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 rec.artifact(__file__,'kernels/runner.py');rec.artifact('build/lfm-moe/fp8_mma.py','kernels/fp8_mma.py');rec.artifact('build/lfm-moe/compile-fp8-mma.py','kernels/compiler.py')
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
