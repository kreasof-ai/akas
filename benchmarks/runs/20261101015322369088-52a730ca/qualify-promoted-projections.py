"""Requalify selected small dense schedules with exact FP16 operand promotion."""
from pathlib import Path
import argparse,json
from akas.graph import Graph
from akas.kernels.definition import define
from akas.compilation.compile import plan
from akas.compilation.tuning import qualify,SearchSettings
from akas.benchmarking.history import Recorder
from tensor.compiler.frontend.signature import frontend_arguments
from tensor.compiler.search import ScheduleProfile

base=Path('build/lfm-serving')
parser=argparse.ArgumentParser()
parser.add_argument('--rows',type=int,default=4)
parser.add_argument('--out',type=Path,default=base/'promoted-projection-qualification')
parser.add_argument('--cache-working-set-bytes',type=int,default=0)
args=parser.parse_args()
root=args.out;root.mkdir(parents=True,exist_ok=True)
entries=[{**e,'parameters':{**e['parameters'],'r':args.rows},'schedule':{**e['schedule'],'mma_dtype':'float16'}}
         for e in json.loads((base/'small-projection-tuning/profile.json').read_text())['entries']
         if e['operation']=='projection_partial']
graph=Graph(contract='reference',phase='forward')
for i,entry in enumerate(entries):
 p=entry['parameters'];kind=entry['operation']
 primitive=define(dict(kind=kind,parameters=p,contract='reference'),schedule=entry['schedule'])
 bindings={a['name']:graph.buffer(f'op{i}.{a["name"]}',a['shape'],a['dtype'],
           role='output' if a['name']=='out' else 'input') for a in frontend_arguments(primitive,{})}
 graph(kind,p,**bindings)
seed=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=entries))
metadata=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',protocol='operation-tuning-v1',
 qualification_scope='Selected dense FP8 schedules with exact FP16 operands; not model throughput')
record=Recorder('benchmarks',metadata,kind='benchmark',change='Qualify tighter FP8 projection accumulation before model measurements',dependency_sources={'tensor':Path('../tensor')})
record.artifact(Path(__file__),'qualify-promoted-projections.py')
print('STARTED',record.id,flush=True)
try:
 _,report=qualify(plan(graph,target='sm_89',schedules=seed),root,
   settings=SearchSettings(budget=1,width=1,warmup=2,samples=5,repeats=3,cache_working_set_bytes=args.cache_working_set_bytes))
 for filename,key in (('profile.json','profile'),('report.json',None)):
  path=root/filename;path.write_text(json.dumps(report if key is None else report[key],indent=2)+'\n')
  record.artifact(path,filename)
 for path in (root/'measurement-working-set').glob('*'):
  if path.is_file():record.artifact(path,f'measurement-working-set/{path.name}')
 record.update(dict(metadata=metadata,results=[]),final=True,status='passed')
 print('RUN',record.id,flush=True)
 for row in report['operations']:print(row['parameters'],row['selected'],flush=True)
except BaseException as error:
 if record.record['completed_at'] is None:
  record.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
