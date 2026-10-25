"""Compare fused codec-tile projections with their retained AOT implementations."""
import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.artifacts.read import read_program,artifact_path
from akas.kernels.contracts import sample_arguments
from akas.quantization.weights.ggml.validation import sample_weight
from akas.execution.plan import ExecutionPlan
from akas.benchmarking.history import Recorder
root=Path('build/lfm-moe/codec-linear-production');program=Path('build/lfm-moe/fp8-model/cuda/program');top=read_program(program)
manifest=json.loads((program/next(r['path'] for r in top['phases'] if r['phase']=='prefill' and r['chunk']==4096)).read_text())
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),weight_format='q4_0',operand_dtype='float16',accumulator_dtype='float32',qualification='Independent small linear/paired FFN/residual codec references; full-size outputs compared bitwise against retained scalar-decode AOT kernels; kernel timings only')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Share optimized quantization tile access across linear, paired FFN and expert prefill')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
try:
 with tensor.Device(provider='cuda') as device:
  for kind,depth,columns in (('linear',2048,6144),('linear',11776,2048),('ffn',2048,11776)):
   record=next(k for k in manifest['kernels'].values() if k['kind']==kind and k['parameters']['k']==depth and k['parameters']['o']==columns)
   baseline=artifact_path(program,record);metadata,_=read_artifact(baseline);arrays=sample_arguments(metadata['arguments'])
   for name in ('w','w2'):
    if name in arrays:sample_weight(arrays[name],2,decode=False)
   buffers={n:device.from_numpy(a) for n,a in arrays.items()};expected=None
   paths=[('baseline',baseline)]+[(f'{bm}-{bn}-{bk}-{threads}',root/f'{kind}-{depth}-{columns}-{bm}-{bn}-{bk}-{threads}.tbin') for bm,bn,bk,threads in ((64,64,64,128),(64,128,128,256))]
   for label,path in paths:
    with device.load(path) as kernel:
     args=tuple(buffers[a['name']] for a in kernel.manifest['arguments']);kernel.launch(*args);actual=buffers['out'].to_numpy()
     if expected is None:expected=actual.copy()
     np.testing.assert_array_equal(actual,expected)
     plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
     try:timing=measure(device,prepared.launch,warmup=50,samples=7,repeats=10)
     finally:prepared.close()
     check=dict(kind=kind,depth=depth,columns=columns,geometry=label,bitwise=True,launch=kernel.manifest['launch'],timing=timing);checks.append(check);print('PASS',kind,depth,columns,label,'ms',timing['median_seconds']*1000,flush=True)
    if label!='baseline':rec.artifact(path.with_suffix('.py'),'kernels/'+path.with_suffix('.py').name)
   for buffer in buffers.values():buffer.release()
 passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks;rec.artifact(__file__,'kernels/validator.py');rec.artifact('build/lfm-moe/compile-codec-linear-production.py','kernels/compiler.py')
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
