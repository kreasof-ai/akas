import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.ops.attention.validation import canonical_validation
from akas.kernels.contracts import sample_arguments
from akas.artifacts.read import read_program,artifact_path
from akas.execution.plan import ExecutionPlan
from akas.benchmarking.history import Recorder
root=Path('build/lfm-moe/wide-query-attention');program=Path('build/lfm-moe/codec-model/cuda/program');top=read_program(program)
manifest=json.loads((program/next(r['path'] for r in top['phases'] if r['phase']=='prefill' and r['chunk']==4096)).read_text());record=next(k for k in manifest['kernels'].values() if k['kind']=='attention');baseline=artifact_path(program,record)
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),kv_dtype='fp8',operand_dtype='float16',accumulator_dtype='float32',qualification='Independent small NumPy online-softmax expectation, poisoned future cache, inactive slots and query tails; full 4096-query outputs at 8K and 131K against independently qualified current AOT GQA implementation. Query geometry changes only; kernel timings only.')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Increase prefill KV reuse with larger query tiles and 512-thread MMA blocks')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
try:
 with tensor.Device(provider='cuda') as device:
  for qm in (128,256):
   path=root/f'{qm}-small.tbin';metadata,_=read_artifact(path);p=dict(r=66,h=8,kh=2,d=64,capacity=4096,chunk=33,kv_dtype='fp8',provider='cuda',prefill_precision='float16',key_rows=64)
   case=canonical_validation('attention',p,metadata['arguments']);a=case.arguments;a['enabled'][32:]=0;a['q'][32:]=np.nan;case.expected['out'].reshape(66,8,64)[32:]=0
   for slot in range(2):
    enabled=a['enabled'][slot*33:(slot+1)*33]!=0;pos=a['positions'][slot*33:(slot+1)*33];cutoff=int(pos[enabled].max())+1 if enabled.any() else 0
    for name in ('kc','vc'):a[name][slot,:,cutoff:]=127
    for name in ('ks','vs'):a[name][slot,:,cutoff:]=np.nan
   buffers={n:device.from_numpy(v) for n,v in a.items()}
   with device.load(path) as kernel:
    kernel.launch(*[buffers[arg['name']] for arg in kernel.manifest['arguments']]);np.testing.assert_allclose(buffers['out'].to_numpy(),case.expected['out'],rtol=case.rtol,atol=case.atol)
   for b in buffers.values():b.release()
   checks.append(dict(phase='independent-small',query_rows=qm,correct=True,poisoned_future=True,inactive_slots=True));print('SMALL_PASS',qm,flush=True)
  metadata,_=read_artifact(baseline)
  for length in (8192,131072):
   arrays=sample_arguments(metadata['arguments']);rng=np.random.default_rng(44);arrays['positions'][:]=np.arange(length-4096,length,dtype='int32');arrays['enabled'][:]=1;arrays['enabled'][-1]=0;arrays['q'][-1]=np.nan
   if 'mapping' in arrays:arrays['mapping'][:]=0
   for name in ('ks','vs'):arrays[name][:]=np.abs(arrays[name])+.01;arrays[name][:,:,length-1:]=np.nan
   for name in ('kc','vc'):
    tile=rng.integers(0,100,1048573,dtype='uint8');tile|=rng.integers(0,2,1048573,dtype='uint8')<<7;arrays[name][:]=np.resize(tile,arrays[name].shape);arrays[name][:,:,length-1:]=127
   buffers={n:device.from_numpy(a) for n,a in arrays.items()};expected=None
   for label,path in [('baseline',baseline)]+[(str(qm),root/f'{qm}-64-512.tbin') for qm in (128,256)]:
    with device.load(path) as kernel:
     args=tuple(buffers[a['name']] for a in kernel.manifest['arguments']);kernel.launch(*args);actual=buffers['out'].to_numpy()
     if expected is None:expected=actual.copy()
     np.testing.assert_allclose(actual,expected,rtol=.002,atol=.0002)
     plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
     try:timing=measure(device,prepared.launch,warmup=5,samples=5,repeats=1)
     finally:prepared.close()
     check=dict(phase='full',prefix=length,query_rows=label,correct=True,bitwise=bool(np.array_equal(actual,expected)),max_absolute_error=float(np.max(np.abs(actual-expected))),launch=kernel.manifest['launch'],timing=timing);checks.append(check);print('PASS',length,label,'ms',timing['median_seconds']*1000,'max_error',check['max_absolute_error'],flush=True)
   for b in buffers.values():b.release()
 passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 for path in (__file__,'build/lfm-moe/wide_query_attention.py','build/lfm-moe/compile-wide-query-attention.py','build/lfm-moe/compile-wide-query-small.py'):rec.artifact(path,'kernels/'+Path(path).name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
