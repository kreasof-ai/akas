import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.quantization._shared.e4m3 import decode_storage
from akas.benchmarking.history import Recorder
from akas.execution.plan import ExecutionPlan
from akas.memory.transfer import write
root=Path('build/lfm-moe/attention-groups')
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',device=dict(name='NVIDIA L4',provider='cuda'),qualification='Independent NumPy decoded cache, FP16 operands and probabilities; prepared graph latency; kernel microbenchmark only, precision labelled per case')
rec=Recorder('benchmarks',meta,kind='configuration',change='Compare fixed CUDA pipeline group lengths for grouped tensor-core decode')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False

def oracle(arrays,dtype,length,splits):
 bk=64;out=np.zeros((32,splits,66),dtype='float64');out[:,:,64]=-np.inf;span=((length+bk-1)//bk+splits-1)//splits*bk
 for head in range(8):
  def cache(name,scale):
   values=arrays[name][0,head,:length]
   if dtype=='fp8':values=decode_storage(values)*arrays[scale][0,head,:length]
   return values.astype('float16').astype('float64')
  keys=cache('kc','ks');vals=cache('vc','vs');q=arrays['q'][0].reshape(32,64)[head*4:head*4+4].astype('float16').astype('float64')
  for part in range(splits):
   maximum=np.full(4,-np.inf);norm=np.zeros(4);acc=np.zeros((4,64))
   for start in range(part*span,min(length,(part+1)*span),bk):
    stop=min(length,start+bk);scores=q@keys[start:stop].T*.125;nextmax=np.maximum(maximum,scores.max(-1));correction=np.exp(maximum-nextmax)
    prob=np.exp(scores-nextmax[:,None]);acc=acc*correction[:,None]+prob.astype('float16').astype('float64')@vals[start:stop];norm=norm*correction+prob.sum(-1);maximum=nextmax
   out[head*4:head*4+4,part,:64]=acc;out[head*4:head*4+4,part,64]=maximum;out[head*4:head*4+4,part,65]=norm
 return out.astype('float32').ravel()
try:
 with tensor.Device(provider='cuda') as device:
  for dtype in ('float16','fp8'):
   rng=np.random.default_rng(321)
   metadata,_=read_artifact(root/f'{dtype}-64-8.tbin');arrays={a['name']:np.zeros(a['shape'],dtype=a['dtype']) for a in metadata['arguments']}
   arrays['q'][:]=rng.normal(0,.2,arrays['q'].shape)
   for name in ('kc','vc'):
    if dtype=='fp8':
     tile=rng.integers(0,80,1048573,dtype='uint8');tile|=rng.integers(0,2,1048573,dtype='uint8')<<7
    else:tile=rng.normal(0,.2,1048573).astype('float16')
    arrays[name][:]=np.resize(tile,arrays[name].shape)
   for name in ('ks','vs'):arrays[name][:]=1 if dtype=='float16' else rng.uniform(.005,.05,arrays[name].shape)
   arrays['enabled'][:]=1
   buffers={name:device.from_numpy(value) for name,value in arrays.items()}
   for splits,group in ((16,8),(64,8),(16,16),(64,16)):
    path=root/f'{dtype}-{splits}-{group}.tbin';metadata,_=read_artifact(path)
    with device.load(path) as kernel:
     for length in (32,8192,131072):
      arrays['positions'][:]=length-1;write(device,buffers['positions'],arrays['positions']);expected=oracle(arrays,dtype,length,splits)
      output=device.from_numpy(np.zeros(next(a['shape'] for a in metadata['arguments'] if a['name']=='out'),dtype='float32'))
      args=tuple(output if a['name']=='out' else buffers[a['name']] for a in metadata['arguments'])
      kernel.launch(*args);actual=output.to_numpy().ravel()
      # Match the operation's existing raw-partial tolerance, then check
      # the normalized attention result at a much smaller absolute tolerance.
      np.testing.assert_allclose(actual,expected,rtol=.002,atol=.002)
      def merge(parts):
       parts=parts.reshape(32,splits,66).astype('float64');m=parts[:,:,64];weight=np.exp(m-m.max(-1,keepdims=True))
       return (parts[:,:,:64]*weight[:,:,None]).sum(1)/(parts[:,:,65]*weight).sum(1)[:,None]
      np.testing.assert_allclose(merge(actual),merge(expected),rtol=.002,atol=.000002)
      plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
      try:timing=measure(device,prepared.launch,warmup=3,samples=5,repeats=10)
      finally:prepared.close()
      output.release()
      check=dict(kv_dtype=dtype,splits=splits,pipeline_group=group,length=length,correct=True,timing=timing);checks.append(check);print('PASS',json.dumps(check),flush=True)
    rec.artifact(root/f'{dtype}-{splits}-{group}.py',f'kernels/{dtype}-{splits}-{group}.py')
   for b in buffers.values():b.release()
 meta['checks']=checks;passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 rec.artifact(__file__,'kernels/runner.py');rec.artifact('build/lfm-moe/fp8_mma.py','kernels/fp8_mma.py')
 for path in Path('build/lfm-moe').glob('group_*.py'):rec.artifact(path,'kernels/'+path.name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
