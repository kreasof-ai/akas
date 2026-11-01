"""Compare existing normalization graphs at serving geometries."""
from pathlib import Path
from contextlib import ExitStack
import ctypes as c,json
import numpy as np,tensor
from akas.graph import Graph
from akas.compilation.compile import produce
from akas.compilation.policy import lower
from akas.execution.forward import Forward
from akas.benchmarking.history import Recorder
from tensor.compiler.tuning import measure,cuda_working_set
root=Path('build/lfm-serving/normalization-sequence');root.mkdir(parents=True,exist_ok=True)
metadata=dict(runtime='akas',purpose='kernel-sequence-benchmark',provider='cuda',qualification_scope='Residual and RMS sequence, including useful outputs; not model throughput')
rec=Recorder('/home/sagemaker-user/akas/benchmarks',metadata,kind='benchmark',change='Measure serial-tree fusion after eliminating replicated row staging',dependency_sources={'tensor':Path('/home/sagemaker-user/tensor')})
rec.artifact(Path(__file__),'benchmark-batched-normalization.py');print('STARTED',rec.id,flush=True)
results=[]
try:
 with tensor.Device() as device,ExitStack() as resources:
  prepare,conditioning=resources.enter_context(cuda_working_set(device,256<<20,out=root/'working-set',target='sm_89'))
  for rows in (2048,8192):
   width=2048;rng=np.random.default_rng(19)
   with ExitStack() as case:
    buffers={name:device.from_numpy(rng.normal(0,.1,(rows*width if name!='w' else width,)).astype('float32')) for name in ('x','mixed','w')}
    buffers.update({name:device.empty((rows*width,),'float32') for name in ('residual','out')})
    for value in buffers.values():case.callback(value.release)
    baseline=None
    for fused in (False,True):
     graph=Graph(contract='reference')
     for name,value in buffers.items():graph.buffer(name,value.shape,str(value.dtype),role='output' if name in ('residual','out') else 'input')
     graph('add',dict(r=rows,c=width),'x','mixed','residual')
     graph('rms',dict(r=rows,c=width,eps=1e-6),'residual','w','out')
     if fused:graph,_=lower(graph,policy={'normalization':{'fused_residual':True}},target='sm_89')
     path=root/f'r{rows}-fused{fused}';produce(graph,path,target='sm_89',search=False)
     rec.artifact(path/'forward.json',f'r{rows}-fused{fused}.json')
     with Forward(path,device,buffers,graphs=True) as forward:
      forward.run();actual={name:buffers[name].to_numpy() for name in ('residual','out')}
      if baseline is None:baseline=actual
      else:
       for name,array in actual.items():np.testing.assert_array_equal(array,baseline[name])
      stats=[]
      device.driver.lib.cuFuncGetAttribute.argtypes=[c.POINTER(c.c_int),c.c_int,c.c_void_p]
      for executable in forward.kernels.values():
       values={}
       for name,attr in (('registers',4),('local_bytes',3)):
        v=c.c_int();device.driver.call('cuFuncGetAttribute',c.byref(v),attr,executable.function);values[name]=v.value
       stats.append(values)
      times={label:measure(device,forward.run,warmup=2,samples=7,repeats=5,reset=reset) for label,reset in (('warm',None),('working-set',prepare))}
      row=dict(rows=rows,width=width,fused=fused,bitwise=True,resources=stats,timing=times,cache_conditioning=conditioning)
      results.append(row);print('RESULT',rows,fused,stats,{key:value['median_seconds'] for key,value in times.items()},flush=True)
 out=root/'results.json';out.write_text(json.dumps(dict(results=results),indent=2)+'\n');rec.artifact(out,'results.json')
 for path in (root/'working-set').glob('*'):
  if path.is_file():rec.artifact(path,'working-set/'+path.name)
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
