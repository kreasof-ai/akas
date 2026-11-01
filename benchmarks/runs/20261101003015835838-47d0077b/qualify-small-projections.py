"""Qualify existing mapped experts and prequantized dense GEMMs at small batches."""
from pathlib import Path
import json
from akas.graph import Graph
from akas.kernels.definition import define
from akas.compilation.compile import plan
from akas.compilation.tuning import qualify,SearchSettings
from akas.benchmarking.history import Recorder
from tensor.compiler.frontend.signature import frontend_arguments
from tensor.compiler.search import ScheduleProfile

root=Path('build/lfm-serving/small-projection-tuning')
root.mkdir(parents=True,exist_ok=True)
graph=Graph(contract='reference',phase='forward')
entries=[]

def add(kind,p,schedule):
 primitive=define(dict(kind=kind,parameters=p,contract='reference'))
 prefix=f'op{len(graph.calls)}'
 bindings={a['name']:graph.buffer(f'{prefix}.{a["name"]}',a['shape'],a['dtype'],
            role='output' if a['name'] in ('out','output_scales') else 'input')
            for a in frontend_arguments(primitive,{})}
 graph(kind,p,**bindings)
 entries.append(dict(operation=kind,parameters=p,schedule=schedule))

for rows in (1,4):
 for kind,columns,depth,routed in [('fp8_expert_swiglu',1536,2048,False),('expert_partial',2048,1536,True)]:
  p=dict(r=rows,rows=rows,top=4,experts=64,o=columns,k=depth,routed_input=routed,
         partitions=1,block_m=32,compact=True,tile_map=True)
  schedule=dict(block_m=16,threads=128,stages=2,activation_vector=16,
                implementation='moe.cuda.fp8-swiglu:mapped' if kind=='fp8_expert_swiglu'
                               else 'moe.cuda.fp8-bucketed:mapped')
  if kind=='expert_partial':schedule['columns']=64
  add(kind,p,schedule)

for columns,depth in ((11776,2048),(2048,11776),(2048,2048),(512,2048)):
 for parts in (1,2,4,8):
  if depth%(128*parts):continue
  p=dict(r=4,k=depth,o=columns,input_dtype='float32',partitions=parts,
         columns=128,block_m=32,threads=256,stages=2,packed_copy=True)
  add('projection_partial',p,dict(columns=64,block_m=16,threads=128,stages=1,
       packed_copy=True,implementation='linear.pipeline:reference'))

seed=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=entries))
metadata=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',protocol='operation-tuning-v1',
 qualification_scope='Small-batch FP8 expert and dense projection operations; not model throughput')
record=Recorder('benchmarks',metadata,kind='benchmark',change='Qualify existing small-batch projection paths and legal K partitions')
record.artifact(Path(__file__),'qualify-small-projections.py')
print('STARTED',record.id,flush=True)
try:
 _,report=qualify(plan(graph,target='sm_89',schedules=seed),root,
   settings=SearchSettings(budget=12,width=4,warmup=2,samples=5,repeats=3))
 for filename,key in (('profile.json','profile'),('report.json',None)):
  path=root/filename
  path.write_text(json.dumps(report if key is None else report[key],indent=2)+'\n')
  record.artifact(path,filename)
 record.update(dict(metadata=metadata,results=[]),final=True,status='passed')
 print('RUN',record.id,flush=True)
 for row in report['operations']:print(row['kind'],row['parameters'],row['selected'],flush=True)
except BaseException as error:
 if record.record['completed_at'] is None:
  record.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
