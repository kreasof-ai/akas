"""Complete decode attention sequences at the serving context length."""
from pathlib import Path
from contextlib import ExitStack
import json
import numpy as np
import tensor
from akas.graph import Graph
from akas.compilation.compile import produce
from akas.execution.forward import Forward
from akas.benchmarking.history import Recorder
from tensor.compiler.search import ScheduleProfile
from tensor.compiler.tuning import cuda_working_set, measure

main=Path('/home/sagemaker-user/akas');work=Path.cwd()
root=work/'build/lfm-serving/attention-decode-sequences';root.mkdir(parents=True,exist_ok=True)
metadata=dict(runtime='akas',purpose='attention-sequence-benchmark',provider='cuda',
 qualification_scope='Complete partial and merge attention; FP16 KV with original scalar reference and independent tensor-core tests; not model throughput')
rec=Recorder(main/'benchmarks',metadata,source_root=work,kind='benchmark',
 change='Measure existing GQA tensor-core attention with short-context complete-tile copies',
 dependency_sources={'tensor':main.parent/'tensor'})
rec.artifact(Path(__file__),'benchmark.py');print('STARTED',rec.id,flush=True)
results=[]
try:
 with tensor.Device() as device,ExitStack() as lifetime:
  prepare,conditioning=lifetime.enter_context(cuda_working_set(device,256<<20,out=root/'working-set',target='sm_89'))
  for rows in (4,256,1,16,64):
   rng=np.random.default_rng(814+rows)
   host=dict(q=rng.normal(0,.5,(rows,2048)).astype('float32'),
             kc=rng.normal(0,.5,(rows,8,2176,64)).astype('float16'),
             vc=rng.normal(0,.5,(rows,8,2176,64)).astype('float16'),
             ks=np.ones((rows,8,2176,1),'float32'),vs=np.ones((rows,8,2176,1),'float32'),
             positions=np.full(rows,2048,'int32'),enabled=np.ones(rows,'int32'))
   host['kc'][:,:,2049:]=np.nan;host['vc'][:,:,2049:]=np.nan
   with ExitStack() as case:
    buffers={name:device.from_numpy(array) for name,array in host.items()}
    buffers['out']=device.zeros((rows,2048),'float32')
    for value in buffers.values():case.callback(value.release)
    del host
    candidates=[('attention.cuda.floating',16,64,False)]
    candidates += [('attention.cuda.mma_decode',16,64,False)]
    candidates += [('attention.cuda.mma_decode',splits,bk,True) for bk in (64,128) for splits in (1,2,4,8,16,32)]
    reference=None
    for index,(implementation,splits,bk,copy) in enumerate(candidates):
     graph=Graph(contract='reference')
     for name,buffer in buffers.items():graph.buffer(name,buffer.shape,buffer.dtype,role='output' if name=='out' else 'input')
     partial=graph.buffer('partial',(rows*32*splits*66,),'float32')
     p=dict(r=rows,h=32,kh=8,d=64,capacity=2176,splits=splits,kv_dtype='float16',unit_scales=True,provider='cuda',optimized=True)
     pp={**p,**(dict(decode_precision='float16') if implementation.endswith('mma_decode') else {})}
     graph('attention_partial',pp,graph.buffers['q'],graph.buffers['kc'],graph.buffers['vc'],graph.buffers['ks'],graph.buffers['vs'],partial,graph.buffers['positions'],graph.buffers['enabled'])
     graph('attention_merge',p,partial,graph.buffers['out'],graph.buffers['enabled'])
     schedules=ScheduleProfile(dict(entries=[dict(operation='attention_partial',parameters=pp,
       schedule=dict(implementation=implementation,**(dict(key_rows=bk,stages=2,copy_complete_tiles=copy) if implementation.endswith('mma_decode') else {}))),
       dict(operation='attention_merge',parameters=p,schedule=dict(implementation='attention.cuda.floating'))]))
     destination=root/f'r{rows}-candidate{index}'
     produce(graph,destination,search=False,target='sm_89',schedules=schedules)
     with Forward(destination,device,buffers,graphs=True) as forward:
      forward.run();actual=buffers['out'].to_numpy()
      if reference is None:reference=actual.copy()
      else:np.testing.assert_allclose(actual,reference,rtol=.002,atol=.002)
      timing=measure(device,forward.run,warmup=2,samples=7,repeats=5,reset=prepare)
     row=dict(rows=rows,implementation=implementation,splits=splits,key_rows=bk,copy_complete_tiles=copy,
       maximum_absolute_error=float(np.max(np.abs(actual-reference))),timing=timing)
     results.append(row);path=root/f'r{rows}-candidate{index}-result.json';path.write_text(json.dumps(row,indent=2)+'\n');rec.artifact(path,path.name)
     rec.artifact(destination/'forward.json',f'r{rows}-candidate{index}.json')
     print('RESULT',rows,implementation,splits,bk,copy,timing['median_seconds'],row['maximum_absolute_error'],flush=True)
  path=root/'results.json';path.write_text(json.dumps(dict(results=results,cache_conditioning=conditioning),indent=2)+'\n');rec.artifact(path,'results.json')
  for path in root.rglob('*.tbin'):rec.artifact(path,'producer/'+str(path.relative_to(root)))
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
