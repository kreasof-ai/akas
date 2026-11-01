"""Qualify existing expert schedules with explicit BF16 intermediate boundaries."""
from pathlib import Path
import argparse,json
from akas.graph import Graph
from akas.kernels.definition import define
from akas.compilation.compile import plan
from akas.compilation.tuning import qualify,SearchSettings
from akas.benchmarking.history import Recorder
from tensor.compiler.frontend.signature import frontend_arguments
from tensor.compiler.search import ScheduleProfile

parser=argparse.ArgumentParser()
parser.add_argument('--rows',type=int,nargs='+',default=[2048,8192])
parser.add_argument('--budget',type=int,default=16)
args=parser.parse_args()
main=Path('/home/sagemaker-user/akas');root=main/'build/lfm-serving/bf16-boundary-expert-tuning'
root.mkdir(parents=True,exist_ok=True)
meta=dict(runtime='akas',purpose='expert-schedule-search',provider='cuda',qualification_scope='Independent operation qualification and conditioned kernel timings; not model throughput')
rec=Recorder(main/'benchmarks',meta,source_root=main,kind='benchmark',change='Tune reusable native FP8 experts at declared BF16 precision boundaries',dependency_sources={'tensor':main.parent/'tensor'})
rec.artifact(Path(__file__),'tune.py');print('STARTED',rec.id,flush=True)
try:
 graph=Graph(contract='reference',phase='forward');entries=[]
 for rows in args.rows:
  for kind,o,k,routed in [('fp8_expert_swiglu',1536,2048,False),('expert_partial',2048,1536,True)]:
   p=dict(r=rows,rows=rows,top=4,experts=64,o=o,k=k,routed_input=routed,partitions=1,block_m=32,compact=True,tile_map=True)
   if routed:p['output_dtype']='bfloat16'
   else:p.update(product_dtype='bfloat16',activation_dtype='bfloat16')
   primitive=define(dict(kind=kind,parameters=p,contract='reference'))
   buffers={a['name']:graph.buffer(f'{rows}.{kind}.{a["name"]}',a['shape'],a['dtype'],role='output' if a['name'] in ('out','output_scales') else 'input') for a in frontend_arguments(primitive,{})}
   graph(kind,p,**buffers)
   schedule=dict(block_m=32,threads=256,stages=1,activation_vector=16,mma_dtype='float8_e4m3fn',implementation='moe.cuda.fp8-bucketed:mapped' if routed else 'moe.cuda.fp8-swiglu:mapped')
   if routed:schedule['columns']=64
   entries.append(dict(operation=kind,parameters=p,schedule=schedule))
 baseline=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=entries))
 _,report=qualify(plan(graph,target='sm_89',schedules=baseline),root,settings=SearchSettings(budget=args.budget,width=4,warmup=2,samples=5,repeats=3,cache_working_set_bytes=256<<20))
 for name,value in [('profile.json',report['profile']),('report.json',report)]:
  path=root/name;path.write_text(json.dumps(value,indent=2)+'\n');rec.artifact(path,name)
 for path in root.rglob('*'):
  if path.is_file() and path.suffix in ('.tbin','.py'):
   rec.artifact(path,'producer/'+str(path.relative_to(root)))
 for row in report['operations']:print('SELECTED',row['kind'],row['parameters'],row['selected'],flush=True)
 rec.update(dict(metadata=meta,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=meta,results=[],error=repr(error)),final=True,status='failed')
 raise
