import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.artifacts.read import read_program,artifact_path
from akas.kernels.contracts import sample_arguments
from akas.execution.plan import ExecutionPlan
from akas.benchmarking.history import Recorder
root=Path('build/lfm-moe/router-columns');program=Path('build/lfm-moe/fp8-model/cuda/program');top=read_program(program)
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),operand_dtype='float32',accumulator_dtype='float32',qualification='Independent F64 dot reference for small odd output tail; bitwise full-size comparison with qualified continuous F32 warp projection, including unchanged per-lane FMA order; kernel timing only')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Reuse each F32 router activation across several warp output columns')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
try:
 with tensor.Device(provider='cuda') as device:
  path=root/'3-128-7-4.tbin';metadata,_=read_artifact(path);arrays=sample_arguments(metadata['arguments']);expected=(arrays['x'].reshape(3,128).astype('float64')@arrays['w'].reshape(7,128).astype('float64').T).astype('float32').ravel()
  buffers={n:device.from_numpy(a) for n,a in arrays.items()}
  with device.load(path) as kernel:
   kernel.launch(*[buffers[a['name']] for a in kernel.manifest['arguments']]);np.testing.assert_allclose(buffers['out'].to_numpy(),expected,rtol=1e-6,atol=1e-7)
  for buffer in buffers.values():buffer.release()
  checks.append(dict(phase='small-f64-oracle',correct=True));print('SMALL_ORACLE_OK',flush=True)
  for rows in (32,4096):
   manifest=json.loads((program/next(r['path'] for r in top['phases'] if r['phase']=='prefill' and r['chunk']==rows)).read_text())
   record=next(k for k in manifest['kernels'].values() if k['kind']=='linear' and 'type' not in k['parameters']);baseline=artifact_path(program,record)
   metadata,_=read_artifact(baseline);arrays=sample_arguments(metadata['arguments']);buffers={n:device.from_numpy(a) for n,a in arrays.items()};expected=None
   for label,path in [('baseline',baseline)]+[(str(micro),root/f'{rows}-2048-64-{micro}.tbin') for micro in (2,4,8)]:
    with device.load(path) as kernel:
     args=tuple(buffers[a['name']] for a in kernel.manifest['arguments']);kernel.launch(*args);actual=buffers['out'].to_numpy()
     if expected is None:expected=actual.copy()
     np.testing.assert_array_equal(actual,expected)
     plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
     try:timing=measure(device,prepared.launch,warmup=50,samples=7,repeats=10)
     finally:prepared.close()
     check=dict(rows=rows,columns_per_warp=label,bitwise=True,launch=kernel.manifest['launch'],timing=timing);checks.append(check);print('PASS',rows,label,'ms',timing['median_seconds']*1000,flush=True)
   for buffer in buffers.values():buffer.release()
 passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 for path in (__file__,'build/lfm-moe/router_columns.py','build/lfm-moe/compile-router-columns.py'):rec.artifact(path,'kernels/'+Path(path).name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
