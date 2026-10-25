"""Compare actual FP8 prefill against packed staging, then refine long KV decode."""
import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.compilation.emit import build_artifact
from akas.benchmarking.history import Recorder
from akas.execution.plan import ExecutionPlan
from akas.kernels.contracts import sample_arguments
from akas.quantization._shared.e4m3 import decode_storage
from akas.memory.transfer import write
root=Path('build/lfm-moe/fp8-serving-kernels');root.mkdir(exist_ok=True)
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),kv_dtype='fp8',qualification='Same-precision kernel latency; prefill checked against independently decoded F16 cache and qualified per-head oracle; decode against independent NumPy F16 operands/probabilities, canonical partial tolerances and tighter normalized output check')
rec=Recorder('benchmarks',meta,kind='configuration',change='Measure packed versus scalar FP8 prefill and refine long-context decode geometry')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
try:
 p=dict(r=4096,h=32,kh=8,d=64,capacity=131072,chunk=4096,kv_dtype='fp8',provider='cuda',unit_scales=False,key_rows=64,query_rows=32,stages=0,async_loads=False)
 path=root/'prefill-scalar.py';path.write_text('from akas.ops.attention.cuda import floating\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":floating(KernelSpec("attention",'+repr(p)+'),{})}\n')
 build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89')
 for bk,stages in ((64,1),(128,1),(128,2)):
  p=dict(r=1,h=32,kh=8,d=64,capacity=131072,kv_dtype='fp8',provider='cuda',splits=64,key_rows=bk,stages=stages,optimized=True,decode_precision='float16')
  path=root/f'decode-{bk}-{stages}.py';path.write_text('from akas.ops.attention.cuda import mma_partial\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":mma_partial(KernelSpec("attention_partial",'+repr(p)+'),{})}\n')
  build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89')
 print('COMPILED',flush=True)
 with tensor.Device(provider='cuda') as device:
  metadata,_=read_artifact(root/'prefill-scalar.tbin');arrays=sample_arguments(metadata['arguments']);rng=np.random.default_rng(44)
  arrays['positions'][:]=np.arange(131072-4096,131072,dtype='int32');arrays['enabled'][:]=1;arrays['enabled'][-1]=0
  for name in ('ks','vs'):arrays[name][:]=np.abs(arrays[name])+.01
  for name in ('kc','vc'):
   tile=rng.integers(0,100,1048573,dtype='uint8');tile|=rng.integers(0,2,1048573,dtype='uint8')<<7;arrays[name][:]=np.resize(tile,arrays[name].shape)
  buffers={n:device.from_numpy(v) for n,v in arrays.items()};reference=dict(buffers)
  for cache,scale in (('kc','ks'),('vc','vs')):
   reference[cache]=device.from_numpy((decode_storage(arrays[cache])*arrays[scale]).astype('float16'));reference[scale]=device.from_numpy(np.ones_like(arrays[scale]))
  with device.load('build/lfm-moe/grouped-prefill/reference-64.tbin') as kernel:
   kernel.launch(*[reference[a['name']] for a in kernel.manifest['arguments']]);expected=buffers['out'].to_numpy()
  for label,path in (('scalar',root/'prefill-scalar.tbin'),('packed-staged',Path('build/lfm-moe/prefill-occupancy/fp8-g64t128.tbin'))):
   with device.load(path) as kernel:
    args=tuple(buffers[a['name']] for a in kernel.manifest['arguments']);kernel.launch(*args);np.testing.assert_allclose(buffers['out'].to_numpy(),expected,rtol=.002,atol=.0002)
    plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
    try:timing=measure(device,prepared.launch,warmup=1,samples=3,repeats=1)
    finally:prepared.close()
    check=dict(phase='prefill',implementation=label,correct=True,timing=timing);checks.append(check);print('PASS',json.dumps(check),flush=True)
  for name,buffer in reference.items():
   if buffer is not buffers[name]:buffer.release()
  for buffer in buffers.values():buffer.release()
  # Reuse the independently written bounded CPU oracle, including stage rounding.
  text=Path('build/lfm-moe/validate-attention-splits-v2.py').read_text();text=text[text.index('def oracle('):text.index('\ntry:')];text=text.replace('def oracle(arrays,dtype,length,splits):','def oracle(arrays,dtype,length,splits,bk):').replace(' bk=64;out=',' out=');exec(text)
  metadata,_=read_artifact(root/'decode-64-1.tbin');arrays=sample_arguments(metadata['arguments']);rng=np.random.default_rng(321)
  arrays['q'][:]=rng.normal(0,.2,arrays['q'].shape);arrays['enabled'][:]=1
  for name in ('kc','vc'):
   tile=rng.integers(0,80,1048573,dtype='uint8');tile|=rng.integers(0,2,1048573,dtype='uint8')<<7;arrays[name][:]=np.resize(tile,arrays[name].shape)
  for name in ('ks','vs'):arrays[name][:]=rng.uniform(.005,.05,arrays[name].shape)
  buffers={n:device.from_numpy(v) for n,v in arrays.items()}
  for bk,stages in ((64,1),(128,1),(128,2)):
   with device.load(root/f'decode-{bk}-{stages}.tbin') as kernel:
    args=tuple(buffers[a['name']] for a in kernel.manifest['arguments'])
    for length in (32,8192,131072):
     arrays['positions'][:]=length-1;write(device,buffers['positions'],arrays['positions']);expected=oracle(arrays,'fp8',length,64,bk)
     kernel.launch(*args);actual=buffers['out'].to_numpy().ravel();np.testing.assert_allclose(actual,expected,rtol=.002,atol=.002)
     def merged(parts):
      parts=parts.reshape(32,64,66).astype('float64');m=parts[:,:,64];w=np.exp(m-m.max(-1,keepdims=True));return (parts[:,:,:64]*w[:,:,None]).sum(1)/(parts[:,:,65]*w).sum(1)[:,None]
     np.testing.assert_allclose(merged(actual),merged(expected),rtol=.002,atol=.000002)
     plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
     try:timing=measure(device,prepared.launch,warmup=2,samples=5,repeats=10)
     finally:prepared.close()
     check=dict(phase='decode',key_rows=bk,stages=stages,splits=64,length=length,correct=True,timing=timing);checks.append(check);print('PASS',json.dumps(check),flush=True)
  for buffer in buffers.values():buffer.release()
 passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks;rec.artifact(__file__,'kernels/runner.py');rec.artifact('build/lfm-moe/validate-attention-splits-v2.py','kernels/oracle.py');rec.artifact('build/lfm-moe/prefill_occupancy.py','kernels/prefill_occupancy.py')
 for path in root.glob('*.py'):rec.artifact(path,'kernels/'+path.name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
