"""Compile short-serving target phases through public codec staging."""
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
ROOT=Path('/home/sagemaker-user/akas');OLD=ROOT/'build/lfm-moe/dspark-short-optimized/target';OUT=ROOT/'build/lfm-moe/dspark-staged-public/target';OUT.mkdir(parents=True,exist_ok=True)
meta=dict(runtime='akas',purpose='compilation',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4'),
 model='akas.models.lfm2.model:LFM2MoE',mode='dspark',weight_format='q4_0',kv_dtype='float16',draft_weight_dtype='q8_0',
 max_context=131072,verify_chunks=[1,2,4,10],shape_timeout_seconds=120,
 qualification='Public operation qualification with codec-owned compressed staging. All target phases, including original single-token fusion and floating KV, pass ordinary independent references. Source-bound caches are reused only when valid. Short-serving geometries; this preparation does not measure long-context throughput.')
rec=Recorder(ROOT/'benchmarks',meta,kind='algorithm',parent='20261023145202380878-9c040e07',change='Compile shared compressed staging for short speculative target verification')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);write(OUT.parent/'run-id.json',dict(run_id=rec.id));print('RUN',rec.id,flush=True);passed=False
try:
 def cached(source,dest):
  if not Path(dest).exists():os.link(source,dest)
 for name in ('compiler-cache','search-cache','search-kernels'):
  if (OLD/name).is_dir():shutil.copytree(OLD/name,OUT/name,copy_function=cached,dirs_exist_ok=True)
 entries=[]
 for phase in read_program(OLD)['phases']:entries.extend(json.loads((OLD/phase['path']).read_text())['tuning']['profile']['entries'])
 def key(p):return json.dumps(p,sort_keys=True)
 exact={(e['operation'],key(e['parameters'])):e['schedule'] for e in entries}
 reader=LFM2MoE.checkpoint_type(ROOT/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf')
 options=dict(kv_dtype='float16',auxiliary_layers=(2,10,18,26,34))
 geometries=[(1,c) for c in (1,32,512)]+[(1,c,'verify') for c in (1,2,4,10)]
 policy=dict(attention=dict(splits=64,decoded_kv=False,native_tail=True),moe=dict(fused_swiglu=True))
 seed={}
 for rows,chunk,*phase in geometries:
  phase=phase[0] if phase else None
  result=LFM2MoE.compose(reader,rows=rows,chunk=chunk,max_requests=1,max_context=131072,provider='cuda',phase=phase,**options)
  graph,_=lower(result.graph,result.commit,policy=policy,provider='cuda',target='sm_89')
  for call in graph.calls:
   p=call.parameters;axes=exact.get((call.kind,key(p)))
   if call.kind=='quantized_experts' and 1<p['r']<=10:
    axes=dict(implementation='moe.cuda.staged',block_m=16,columns=64,block_k=256,threads=256,stages=1)
   if axes is not None:seed[call.key]=dict(operation=call.kind,parameters=p,schedule=axes)
 profile=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=list(seed.values())));write(OUT.parent/'seed.json',profile.data)
 with tensor.Device(provider='cuda') as dev:
  start=time.monotonic();compile_program(reader,OUT,definition=LFM2MoE.compose,geometries=geometries,max_requests=1,max_context=131072,provider='cuda',target='sm_89',model_options=options,policy=policy,schedules=profile,search_settings=SearchSettings(budget=1,width=1,warmup=0,samples=1,repeats=1),device=dev);meta['seconds']=time.monotonic()-start
 rec.program(OUT);passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;rec.artifact(__file__,'compilation/runner.py')
 if (OUT.parent/'seed.json').exists():rec.artifact(OUT.parent/'seed.json','compilation/seed.json')
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
