"""Conditioned convolution schedule measurements at serving prefill geometries."""
from pathlib import Path
from contextlib import ExitStack
import json
import numpy as np,tensor
from akas.benchmarking.history import Recorder
from akas.compilation.selection import source
from akas.kernels.definition import define
from akas.kernels.registry import validation
from tensor.compiler.frontend.signature import frontend_arguments
from tensor.compiler.tuning import cuda_working_set,measure
root=Path.cwd()/'build/lfm-serving/convolution-time-tiles';root.mkdir(parents=True,exist_ok=True)
metadata=dict(runtime='akas',purpose='kernel-benchmark',provider='cuda',target='sm_89',qualification_scope='Independent convolution state equations and serving-shaped inputs; component latency, not model throughput')
rec=Recorder(Path.cwd()/'benchmarks',metadata,source_root=Path.cwd(),kind='benchmark',change='Measure time tiling within the shared channel-owned convolution',dependency_sources={'tensor':Path('/home/sagemaker-user/tensor')})
rec.artifact(Path(__file__),'benchmark.py');print('STARTED',rec.id,flush=True)
results=[]
try:
 with tensor.Device() as device,ExitStack() as lifetime:
  prepare,conditioning=lifetime.enter_context(cuda_working_set(device,256<<20,out=root/'working-set',target='sm_89'))
  for rows in (1,4):
   p=dict(r=rows,c=2048,chunk=2048,length=3)
   case=validation('conv',p,frontend_arguments(define(dict(kind='conv',parameters=p,contract='reference'),capabilities=dict(provider='cuda',target='sm_89')),{}))
   # Large arrays are deliberately nonconstant, including nonzero initial history.
   with ExitStack() as buffers:
    values={name:device.from_numpy(v) for name,v in case.arguments.items()}
    for value in values.values():buffers.callback(value.release)
    candidates=[(2048,8,256),(32,16,128),(64,16,128),(128,16,128),(32,32,128),(64,32,128),(128,32,128),(256,32,128),(64,64,128),(128,64,128),(64,128,256),(128,128,256)]
    for index,(time_tile,columns,threads) in enumerate(candidates):
     candidate=dict(p,time_tile=time_tile,columns=columns,threads=threads)
     entry=root/f'r{rows}-candidate{index}.py';artifact=entry.with_suffix('.tbin')
     entry.write_text(source('conv',candidate,contract='reference',target='sm_89').render());tensor.build(entry,artifact,target='sm_89')
     with device.load(artifact) as executable:
      # Reset the recurrent state before each independent qualification.
      values['state'].release();values['state']=device.from_numpy(case.arguments['state']);buffers.callback(values['state'].release)
      call=executable.bind(*(values[a['name']] for a in executable.manifest['arguments']))
      call.launch()
      for name in ('state','out'):np.testing.assert_allclose(values[name].to_numpy(),case.expected[name],rtol=case.rtol,atol=case.atol)
      from tensor.providers.cuda.graph import CudaGraph
      with CudaGraph(device,call.launch,resources=tuple(values.values())) as graph:
       timing=measure(device,graph.launch,warmup=2,samples=7,repeats=5,reset=prepare)
      result=dict(rows=rows,time_tile=time_tile,columns=columns,threads=threads,timing=timing,independent_reference=True)
      results.append(result);path=root/f'r{rows}-candidate{index}.json';path.write_text(json.dumps(result,indent=2)+'\n');rec.artifact(path,path.name)
      print('RESULT',rows,time_tile,columns,threads,timing['median_seconds'],flush=True)
     rec.artifact(entry,entry.name);rec.artifact(artifact,artifact.name)
  path=root/'results.json';path.write_text(json.dumps(dict(results=results,cache_conditioning=conditioning),indent=2)+'\n');rec.artifact(path,'results.json')
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
