"""Qualify factored FP8 attention and measure its kernel, separately from serving."""
import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.compiler.tuning import measure
from akas.artifacts.read import read_program,artifact_path
from akas.execution.plan import ExecutionPlan
from akas.memory.transfer import write
from akas.benchmarking.history import Recorder
from factored_reference import inputs,merge
root=Path('build/lfm-moe/factored-scaled-attention');program=Path('build/lfm-moe/fp8-model/cuda/program')
top=read_program(program);manifest=json.loads((program/next(r['path'] for r in top['phases'] if r['phase']=='decode')).read_text())
record=next(k for k in manifest['kernels'].values() if k['kind']=='attention_partial');baseline=artifact_path(program,record)
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),kv_dtype='fp8',mode='ar',operand_dtype='float16',accumulator_dtype='float32',probability_scale=128,precision_policy='Decode unscaled FP8 values exactly to F16; apply per-position K scale after QK and V scale before probability staging, with exact power-of-two compensation. Different rounding points from staged scaled-F16 cache.',qualification='Independent NumPy FP8 bit equation and factored online-softmax reference; strict merged absolute tolerance 2e-6; poisoned future cache and inactive query checks. Kernel timing only; no model quality claim.')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Factor per-position FP8 KV scales out of tensor-core operand decoding')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
try:
 with tensor.Device(provider='cuda') as device,np.load(root/'expected.npz',allow_pickle=False) as references:
  for length in (8192,131055):
   arrays=inputs();arrays['mapping']=np.array([0],dtype='int32');arrays['positions'][0]=length-1
   for name in ('kc','vc'):arrays[name][:,:,length:]=127
   for name in ('ks','vs'):arrays[name][:,:,length:]=np.nan
   buffers={n:device.from_numpy(a) for n,a in arrays.items()};baseline_values=None
   for label,path,bk in [('staged-reference',baseline,64)]+[(f'factored-{bk}',root/f'131072-{bk}.tbin',bk) for bk in (64,128)]+[(f'masked-softmax-{bk}',Path('build/lfm-moe/masked-softmax-attention')/f'131072-{bk}.tbin',bk) for bk in (64,128)]:
    with device.load(path) as kernel:
     args=tuple(buffers[a['name']] for a in kernel.manifest['arguments']);kernel.launch(*args);actual=buffers['out'].to_numpy();normalized=merge(actual)
     if not np.isfinite(normalized).all():raise ValueError('nonfinite merged output')
     error=None
     if label=='staged-reference':baseline_values=normalized.copy()
     else:
      expected=references[f'p{length}-bk{bk}'];np.testing.assert_allclose(actual,expected,rtol=.002,atol=.002);error=float(np.max(np.abs(normalized-merge(expected))))
      np.testing.assert_allclose(normalized,merge(expected),rtol=0,atol=2e-6)
     plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
     try:timing=measure(device,prepared.launch,warmup=50,samples=7,repeats=10)
     finally:prepared.close()
     write(device,buffers['q'],np.full_like(arrays['q'],np.nan));write(device,buffers['enabled'],np.zeros_like(arrays['enabled']));kernel.launch(*args)
     inactive=buffers['out'].to_numpy().reshape(32,64,66);np.testing.assert_array_equal(inactive[:,:,:64],0);np.testing.assert_array_equal(inactive[:,:,65],0);assert np.isneginf(inactive[:,:,64]).all()
     write(device,buffers['q'],arrays['q']);write(device,buffers['enabled'],arrays['enabled'])
     check=dict(prefix=length,implementation=label,key_rows=bk,correct=True,poisoned_future=True,inactive_query=True,merged_max_absolute_error=error,relative_output_difference_from_staged=float(np.linalg.norm(normalized-baseline_values)/max(np.linalg.norm(baseline_values),1e-30)),launch=kernel.manifest['launch'],timing=timing);checks.append(check);print('PASS',length,label,'ms',timing['median_seconds']*1000,'error',error,flush=True)
   for buffer in buffers.values():buffer.release()
 passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 for path in (__file__,'build/lfm-moe/factored_reference.py','build/lfm-moe/prepare-factored-reference.py','build/lfm-moe/factored_scaled_attention.py','build/lfm-moe/compile-factored-scaled-attention.py','build/lfm-moe/masked_softmax_attention.py','build/lfm-moe/compile-masked-softmax-attention.py'):
  rec.artifact(path,'kernels/'+Path(path).name)
 rec.artifact(root/'expected.npz','reference/expected.npz')
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
