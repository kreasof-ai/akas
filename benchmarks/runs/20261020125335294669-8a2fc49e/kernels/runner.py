import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.kernels.registry import validation
from akas.kernels.definition import KernelSpec
from akas.compilation.emit import build_artifact
from akas.benchmarking.history import Recorder
from akas.execution.plan import ExecutionPlan
root=Path('build/lfm-moe/parallel-routes')
for rows in (33,512,4096):
 path=root/f'{rows}-serial.py'
 path.write_text('from akas.ops.moe.kernel import quantized_routes\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":quantized_routes(KernelSpec("quantized_routes",'+repr(dict(r=rows,top=4,experts=64))+'),{})}\n')
 build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89')
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',device=dict(name='NVIDIA L4',provider='cuda'),qualification='Bitwise independent CPU stable route compaction, all/inactive rows, non-warp-aligned tails; captured graph latency')
rec=Recorder('benchmarks',meta,kind='diagnostic',change='Parallelize expert route compaction while retaining stable token/rank order')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
try:
 with tensor.Device(provider='cuda') as device:
  from akas.memory.transfer import write
  for rows in (33,512,4096):
   p=dict(r=rows,top=4,experts=64)
   metadata,_=read_artifact(root/f'{rows}.tbin');case=validation('quantized_routes',p,metadata['arguments'])
   # Random routing and inactive rows, exact expected compaction recomputed below.
   rng=np.random.default_rng(19+rows);case.arguments['ids'][:]=np.argsort(rng.random((rows,64)),axis=1)[:,:4]
   buffers={name:device.from_numpy(a) for name,a in case.arguments.items()}
   for all_inactive in (False,True):
    active=rng.integers(0,2,rows,dtype='int32') if not all_inactive else np.zeros(rows,dtype='int32')
    write(device,buffers['active'],active)
    expected_routes=np.full((64,rows),-1,dtype='int32');expected_counts=np.zeros(64,dtype='int32')
    for row in range(rows):
     if active[row]:
      for rank,expert in enumerate(case.arguments['ids'][row]):
       expected_routes[expert,expected_counts[expert]]=row*4+rank;expected_counts[expert]+=1
    for implementation,suffix in (('serial','-serial'),('parallel','')):
     path=root/f'{rows}{suffix}.tbin';metadata,_=read_artifact(path)
     with device.load(path) as kernel:
      args=tuple(buffers[a['name']] for a in metadata['arguments']);kernel.launch(*args)
      np.testing.assert_array_equal(buffers['routes'].to_numpy(),expected_routes);np.testing.assert_array_equal(buffers['counts'].to_numpy(),expected_counts)
      plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
      try:timing=measure(device,prepared.launch,warmup=2,samples=5,repeats=20)
      finally:prepared.close()
      check=dict(rows=rows,all_inactive=all_inactive,implementation=implementation,correct=True,timing=timing);checks.append(check);print('PASS',json.dumps(check),flush=True)
   for b in buffers.values():b.release()
 meta['checks']=checks;passed=True
except BaseException as error:
 meta['error']=str(error);print(traceback.format_exc(),flush=True)
finally:
 meta['checks']=checks
 rec.artifact(__file__,'kernels/runner.py');rec.artifact('build/lfm-moe/parallel_routes.py','kernels/parallel_routes.py')
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
