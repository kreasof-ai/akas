import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.ops.attention.validation import canonical_validation
from akas.kernels.contracts import sample_arguments
from akas.quantization._shared.e4m3 import decode_storage
from akas.compilation.emit import build_artifact
from akas.benchmarking.history import Recorder
from akas.execution.plan import ExecutionPlan
from akas.memory.transfer import write
root=Path('build/lfm-moe/grouped-prefill')
# These reference kernels use the already-qualified per-query-head F16 path.
for bk in (64,128):
 p=dict(r=4096,h=32,kh=8,d=64,capacity=131072,chunk=4096,kv_dtype='float16',unit_scales=True,query_rows=32,key_rows=bk,stages=2,threads=128,async_loads=True)
 path=root/f'reference-{bk}.py';path.write_text('from akas.ops.attention.cuda import floating\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":floating(KernelSpec("attention",'+repr(p)+'),{})}\n')
 build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89')
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',device=dict(name='NVIDIA L4',provider='cuda'),qualification='Independent NumPy full small/tail/inactive-row fixtures; full 4096-query/131K comparison to the independently-qualified per-head tensor-core path using identically decoded F16 KV; kernel latency only')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Share K/V staging across grouped query heads during CUDA prefill')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
try:
 with tensor.Device(provider='cuda') as device:
  for dtype in ('float16','fp8'):
   # Independent CPU expectation covers all small-fixture outputs.
   for qm,bk in ((64,64),(128,64),(64,128)):
    path=root/f'{dtype}-{qm}-{bk}-small.tbin';metadata,_=read_artifact(path)
    p=dict(r=33,h=8,kh=2,d=64,capacity=4096,chunk=33,kv_dtype=dtype,unit_scales=dtype=='float16',provider='cuda',prefill_precision='float16',key_rows=bk)
    case=canonical_validation('attention',p,metadata['arguments']);buffers={n:device.from_numpy(v) for n,v in case.arguments.items()}
    with device.load(path) as kernel:
     kernel.launch(*[buffers[a['name']] for a in metadata['arguments']]);actual=buffers['out'].to_numpy()
     np.testing.assert_allclose(actual,case.expected['out'],rtol=case.rtol,atol=case.atol)
    for buffer in buffers.values():buffer.release()
    check=dict(kv_dtype=dtype,query_rows=qm,key_rows=bk,fixture='independent-small',correct=True);checks.append(check);print('PASS',json.dumps(check),flush=True)
   metadata,_=read_artifact(root/f'{dtype}-64-64.tbin');arrays=sample_arguments(metadata['arguments']);rng=np.random.default_rng(44)
   arrays['positions'][:]=np.arange(131072-4096,131072,dtype='int32');arrays['enabled'][:]=1;arrays['enabled'][-1]=0
   for name in ('ks','vs'):arrays[name][:]=1 if dtype=='float16' else np.abs(arrays[name])+.01
   if dtype=='fp8':
    for name in ('kc','vc'):
     tile=rng.integers(0,100,1048573,dtype='uint8');tile|=rng.integers(0,2,1048573,dtype='uint8')<<7;arrays[name][:]=np.resize(tile,arrays[name].shape)
   buffers={n:device.from_numpy(v) for n,v in arrays.items()}
   # Decode with the independent host codec, then stage with identical F16 rounding.
   reference_buffers=dict(buffers)
   if dtype=='fp8':
    for cache,scale in (('kc','ks'),('vc','vs')):
     reference_buffers[cache]=device.from_numpy((decode_storage(arrays[cache])*arrays[scale]).astype('float16'))
     reference_buffers[scale]=device.from_numpy(np.ones_like(arrays[scale]))
   references={}
   for bk in (64,128):
    path=root/f'reference-{bk}.tbin';metadata,_=read_artifact(path)
    with device.load(path) as kernel:
     args=tuple(reference_buffers[a['name']] for a in metadata['arguments']);kernel.launch(*args);references[bk]=buffers['out'].to_numpy().copy()
     plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
     try:timing=measure(device,prepared.launch,warmup=1,samples=3,repeats=1)
     finally:prepared.close()
     check=dict(kv_dtype=dtype,key_rows=bk,fixture='full-reference',timing=timing);checks.append(check);print('REFERENCE',json.dumps(check),flush=True)
   for qm,bk in ((64,64),(128,64),(64,128)):
    path=root/f'{dtype}-{qm}-{bk}.tbin';metadata,_=read_artifact(path)
    with device.load(path) as kernel:
     args=tuple(buffers[a['name']] for a in metadata['arguments']);kernel.launch(*args);actual=buffers['out'].to_numpy()
     np.testing.assert_allclose(actual,references[bk],rtol=.002,atol=.0002)
     plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
     try:timing=measure(device,prepared.launch,warmup=1,samples=3,repeats=1)
     finally:prepared.close()
     check=dict(kv_dtype=dtype,query_rows=qm,key_rows=bk,fixture='full',correct=True,timing=timing);checks.append(check);print('PASS',json.dumps(check),flush=True)
    rec.artifact(root/f'{dtype}-{qm}-{bk}.py',f'kernels/{dtype}-{qm}-{bk}.py')
   for name,buffer in reference_buffers.items():
    if buffer is not buffers[name]:buffer.release()
   for buffer in buffers.values():buffer.release()
 meta['checks']=checks;passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 rec.artifact(__file__,'kernels/runner.py');rec.artifact('build/lfm-moe/grouped_prefill.py','kernels/grouped_prefill.py');rec.artifact('build/lfm-moe/compile-grouped-prefill.py','kernels/compiler.py')
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
