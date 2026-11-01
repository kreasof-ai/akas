from pathlib import Path
from dataclasses import replace
import json
from akas.graph import Graph
from akas.kernels.definition import define
from akas.kernels.registry import candidates
from akas.compilation.compile import plan
from akas.compilation.tuning import qualify, SearchSettings
from akas.benchmarking.history import Recorder
from tensor.compiler.frontend.signature import frontend_arguments
from tensor.compiler.search import ScheduleProfile

root=Path('build/lfm-serving/native-half-copy-tuning')
root.mkdir(parents=True,exist_ok=True)
graph=Graph(contract='reference',phase='prefill')
seed=json.loads(Path('build/lfm-serving/dense-projection-qualified/profile.json').read_text())
entries=[]
for rows in (2048,8192):
 for columns in (6144,2048):
  p=dict(r=rows,k=2048,o=columns,optimized=True,weight_dtype='float16',input_dtype='float16')
  primitive=define(dict(kind='linear',parameters=p,contract='reference'))
  bindings={a['name']:graph.buffer(f'{rows}.{columns}.{a["name"]}',a['shape'],a['dtype'],
            role='output' if a['name']=='out' else 'input')
            for a in frontend_arguments(primitive,{})}
  graph('linear',p,**bindings)
  incumbent=next(e for e in seed['entries'] if e['operation']=='linear' and e['parameters']['o']==columns)
  entries.append(dict(operation='linear',parameters=p,schedule={k:v for k,v in incumbent['schedule'].items()
             if k not in ('minimum_mma_groups','maximum_gemv_rows')}))
profile=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=entries))
implementation=next(v for v in candidates('linear',p,contract='reference',provider='cuda',phase='prefill',target='sm_89')
                    if v.identifier=='linear.cuda.floating')
# The workload already selects MMA; spend its producer budget on physical tiles.
catalog=(replace(implementation,accepts=None,space=tuple((k,v) for k,v in implementation.space
             if k in ('block_m','block_n','block_k','threads','stages'))),)
metadata=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',protocol='operation-tuning-v1',
              qualification_scope='Native FP16 projections; not model throughput')
record=Recorder('benchmarks',metadata,kind='benchmark',change='Qualify shared typed weight tile copies and projection schedules')
try:
 _,report=qualify(plan(graph,target='sm_89',schedules=profile,catalog=catalog),root,
  settings=SearchSettings(budget=16,width=4,warmup=2,samples=5,repeats=3))
 (root/'profile.json').write_text(json.dumps(report['profile'],indent=2)+'\n')
 (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 record.artifact(Path(__file__),'tune-native-half.py')
 record.artifact(root/'profile.json','profile.json')
 record.artifact(root/'report.json','qualification.json')
 record.update(dict(metadata=metadata,results=[]),final=True,status='passed')
 print('RUN',record.id,flush=True)
 for row in report['operations']:print(row['parameters'],row['selected'],flush=True)
except BaseException as error:
 if record.record['completed_at'] is None:
  record.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
