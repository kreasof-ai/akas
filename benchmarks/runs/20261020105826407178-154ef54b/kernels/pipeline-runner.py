import json,sys
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.compilation.emit import build_artifact
from akas.compilation.selection import source
from akas.kernels.registry import validation
from akas.execution.plan import ExecutionPlan
root=Path('build/lfm-moe/attention-cuda-async');root.mkdir(exist_ok=True)
for cap in (8192,):
 p=dict(r=512,h=32,kh=8,d=64,capacity=cap,chunk=512,kv_dtype='float16',provider='cuda',optimized=True,unit_scales=True,prefill_precision='float16')
 variants=[('serial',dict(stages=0,async_loads=False)),('async2',dict(stages=2,async_loads=True)),('async3',dict(stages=3,async_loads=True)),('asyncq64',dict(query_rows=64,stages=2,async_loads=True))]
 paths=[]
 for name,schedule in variants:
  entry=root/f'{cap}-{name}.py';artifact=entry.with_suffix('.tbin');entry.write_text(source('attention',{**p,**schedule},contract='reference',provider='cuda',target='sm_89',preferred='attention.cuda.pipelined').render())
  try:
   artifact.unlink(missing_ok=True)
   build_artifact(entry,artifact,provider='cuda',target='sm_89');paths.append((name,schedule,artifact))
  except Exception as e:print(json.dumps(dict(cap=cap,name=name,error=str(e))),flush=True)
 meta,_=read_artifact(paths[0][2]);case=validation('attention',p,meta['arguments'])
 with tensor.Device(provider='cuda') as device:
  buffers={name:device.from_numpy(array) for name,array in case.arguments.items()}
  try:
   for name,schedule,path in [*paths,('serial-repeat',paths[0][1],paths[0][2])]:
    meta,_=read_artifact(path);args=[buffers[a['name']] for a in meta['arguments']]
    with device.load(path) as module:
     module.launch(*args);actual=buffers['out'].to_numpy()
     np.testing.assert_allclose(actual,case.expected['out'],rtol=case.rtol,atol=case.atol)
     plan=ExecutionPlan(device);plan.bind(module,args)
     prepared=plan.capture(resources=(module,*args))
     try:
      timing=measure(device,prepared.launch,warmup=1,samples=3,repeats=3)
     finally:prepared.close()
     timing['execution']='prepared-plan'
     print(json.dumps(dict(cap=cap,name=name,schedule=schedule,timing=timing,correct=True)),flush=True)
  finally:
   for b in buffers.values():b.release()
