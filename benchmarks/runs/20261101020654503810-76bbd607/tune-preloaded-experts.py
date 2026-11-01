from pathlib import Path
import argparse,json
from akas.benchmarking.history import Recorder
from akas.graph import Graph
from akas.kernels.definition import define
from akas.compilation.compile import plan
from akas.compilation.tuning import qualify,SearchSettings
from tensor.compiler.frontend.signature import frontend_arguments
from tensor.compiler.search import ScheduleProfile

parser=argparse.ArgumentParser()
parser.add_argument('--rows',type=int,nargs='+',default=[1,4,16,64,256,2048,8192])
parser.add_argument('--out',type=Path,default=Path('build/lfm-serving/mapped-expert-tuning'))
parser.add_argument('--budget',type=int,default=8)
parser.add_argument('--seed-profile',type=Path,required=True)
parser.add_argument('--preload-scales',action='store_true')
args=parser.parse_args()
graph=Graph(contract='reference',phase='forward')
entries=[]
for rows in args.rows:
 for kind,columns,depth,routed in [('fp8_expert_swiglu',1536,2048,False),('expert_partial',2048,1536,True)]:
  p=dict(r=rows,rows=rows,top=4,experts=64,o=columns,k=depth,routed_input=routed,
         partitions=1,block_m=32,compact=True,tile_map=True)
  primitive=define(dict(kind=kind,parameters=p,contract='reference'))
  bindings={a['name']:graph.buffer(f'{kind}.{rows}.{a["name"]}',a['shape'],a['dtype'],
             role='output' if a['name'] in ('out','output_scales') else 'input')
            for a in frontend_arguments(primitive,{})}
  graph(kind,p,**bindings)
  schedule=dict(block_m=16 if rows<=256 else 32,threads=128 if rows<=4 else 256,
                stages=2 if rows<=4 else 1,activation_vector=16,
                implementation='moe.cuda.fp8-swiglu:mapped' if kind=='fp8_expert_swiglu'
                               else 'moe.cuda.fp8-bucketed:mapped')
  if kind=='expert_partial':schedule['columns']=64
  entries.append(dict(operation=kind,parameters=p,schedule=schedule))
previous=json.loads(args.seed_profile.read_text())['entries']
for entry in entries:
 matched=[row for row in previous if row['operation']==entry['operation'] and row['parameters']==entry['parameters']]
 if matched:entry['schedule']=dict(matched[-1]['schedule'])
 if args.preload_scales:entry['schedule']['preload_scales']=True
metadata=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',protocol='operation-tuning-v1',qualification_scope='Shared expert scale preloading; independent numerical gates, no model throughput claim')
record=Recorder('/home/sagemaker-user/akas/benchmarks',metadata,kind='qualification',change='Measure reusable expert scale staging against qualified prefill schedules',dependency_sources={'tensor':Path('/home/sagemaker-user/tensor')})
record.artifact(Path(__file__),'tune-preloaded-experts.py')
print('STARTED',record.id,flush=True)
seed=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=entries))
try:
 profile,report=qualify(plan(graph,target='sm_89',schedules=seed),args.out,
  settings=SearchSettings(budget=args.budget,width=4,warmup=2,samples=5,repeats=3))
 for filename,key in [('profile.json','profile'),('report.json',None)]:
  path=args.out/filename;path.write_text(json.dumps(report if key is None else report[key],indent=2)+'\n')
  record.artifact(path,filename)
 record.update(dict(metadata=metadata,results=[]),final=True,status='passed')
 print('RUN',record.id,flush=True)
 for row in report['operations']:print(row['parameters'],row['selected'],flush=True)
except BaseException as error:
 if record.record['completed_at'] is None:record.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
