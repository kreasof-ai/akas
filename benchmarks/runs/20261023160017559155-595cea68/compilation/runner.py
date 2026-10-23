"""Compile only the missing FP8 target with DSpark feature bindings, reusing qualified caches."""
import json,os,resource,shutil,time,traceback
from pathlib import Path
import tensor
from tensor.compiler.search import ScheduleProfile
from akas.artifacts.read import read_program
from akas.benchmarking.history import Recorder,write
from akas.compilation.compile import compile_program
from akas.compilation.policy import lower
from akas.compilation.tuning import SearchSettings
from akas.models.lfm2.model import LFM2MoE
ROOT=Path('/home/sagemaker-user/akas');OUT=ROOT/'build/lfm-moe/c4-final/target-fp8';OUT.mkdir(exist_ok=True)
base=ROOT/'build/lfm-moe/c4-final/target-float16';fp8=ROOT/'build/lfm-moe/cuda-ordered-full/cuda/program';staged=ROOT/'build/lfm-moe/dspark-staged-public/target';old=ROOT/'build/lfm-moe/dspark-short-optimized/target'
for src in (fp8,old,staged):
 for name in ('compiler-cache','search-cache','search-kernels'):
  if (src/name).is_dir():
   def link(a,b):
    try:os.link(a,b)
    except FileExistsError:pass
   shutil.copytree(src/name,OUT/name,copy_function=link,dirs_exist_ok=True)
entries=[]
for directory in (fp8,base):
 for phase in read_program(directory)['phases']:entries.extend(json.loads((directory/phase['path']).read_text())['tuning']['profile']['entries'])
def key(p):
 p=json.loads(json.dumps(p));p.pop('staged_kv',None)
 if 'kv_dtype' in p:p['kv_dtype']='float16'
 if 'unit_scales' in p:p['unit_scales']=True
 return json.dumps(p,sort_keys=True)
lookup={(e['operation'],key(e['parameters'])):e['schedule'] for e in entries}
reader=LFM2MoE.checkpoint_type(ROOT/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf');options=dict(kv_dtype='fp8',auxiliary_layers=(2,10,18,26,34));policy=dict(attention=dict(splits=64,decoded_kv=True),moe=dict(fused_swiglu=True));geometries=[(1,c) for c in (1,32,4096)]+[(1,c,'verify') for c in (1,2,4,10)];seed={}
for rows,chunk,*purpose in geometries:
 result=LFM2MoE.compose(reader,rows=rows,chunk=chunk,max_requests=1,max_context=131072,provider='cuda',phase=purpose[0] if purpose else None,**options);graph,_=lower(result.graph,result.commit,policy=policy,provider='cuda',target='sm_89')
 for call in graph.calls:
  axes=lookup.get((call.kind,key(call.parameters)))
  if call.kind=='attention':axes=dict(implementation='attention.cuda.ordered' if call.parameters.get('staged_kv') else 'attention.cuda.packed_prefill')
  if axes is not None:seed[call.key]=dict(operation=call.kind,parameters=call.parameters,schedule=axes)
profile=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=list(seed.values())));write(OUT.parent/'fp8-seed.json',profile.data)
meta=dict(runtime='akas',purpose='compilation',provider='cuda',target='sm_89',model='akas.models.lfm2.model:LFM2MoE',mode='dspark',weight_format='q4_0',kv_dtype='fp8',max_context=131072,policy=policy,model_options=options,qualification='Public compiler with independent operation references and source-bound cache reuse. Existing measured schedules; no new geometry search campaign.')
rec=Recorder(ROOT/'benchmarks',meta,kind='configuration',parent='20261023145424043306-e22b91f8',change='Prepare missing FP8 target feature/verification phases for C4 context sweep');report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);write(OUT.parent/'fp8-compile-run-id.json',dict(run_id=rec.id));print('COMPILE',rec.id,flush=True);passed=False
try:
 with tensor.Device(provider='cuda') as dev:
  start=time.monotonic();compile_program(reader,OUT,definition=LFM2MoE.compose,geometries=geometries,max_requests=1,max_context=131072,provider='cuda',target='sm_89',model_options=options,policy=policy,schedules=profile,search_settings=SearchSettings(budget=1,width=1,warmup=0,samples=1,repeats=1),device=dev);meta['seconds']=time.monotonic()-start
 rec.program(OUT);passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;rec.artifact(__file__,'compilation/runner.py');rec.artifact(OUT.parent/'fp8-seed.json','compilation/seed.json');rec.update(report,final=True,status='passed' if passed else 'failed');print('COMPILED',passed,meta.get('seconds'),flush=True)
if not passed:raise SystemExit(1)
