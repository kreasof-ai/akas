"""Qualify the manual kernel changes; no multi-candidate search is launched."""
import json,resource,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
import tensor
from tensor.compiler.search import ScheduleProfile
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.cases import Case
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.compilation.compile import compile_program
from akas.compilation.tuning import SearchSettings
from akas.models.lfm2.model import LFM2MoE
from akas.optimization.evaluation import diagnostic,measured_requests
from akas.engine import Engine
from akas.config import EngineConfig
provider=sys.argv[1];target='sm_89' if provider=='cuda' else 'webgpu-portable-v1'
root=Path('build/lfm-moe/manual-v2-matched')/provider;root.mkdir(parents=True,exist_ok=True)
import shutil
shutil.copytree(Path('build/lfm-moe/manual-v2')/provider/'program',root/'program',dirs_exist_ok=True)
checkpoint=Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf').resolve()
oldroot=Path('build/single-user-long-context')/f'lfm2.5-24b-a2b-q4_0-{provider}-default-ar'
oldstate=json.loads((oldroot/'campaign.json').read_text())
oldjob=next(j for j in oldstate['jobs'] if j['recipe']['id']==oldstate['winner']['recipe']) if oldstate.get('winner') else oldstate['jobs'][0]
old=Path(oldjob['directory']);oldreport=json.loads((old/'report.json').read_text())
entries={}
for phase in json.loads((old/'program/program.json').read_text())['phases']:
 forward=json.loads((old/'program'/phase['path']).read_text())
 for e in forward.get('tuning',{}).get('profile',{}).get('entries',[]):
  if e['operation'] not in ('attention','attention_partial','attention_merge'):
   entries[json.dumps([e['operation'],e['parameters']],sort_keys=True)]=e
extra=json.loads(Path('benchmarks/runs/20261019131816505070-605b9f45/preflight/seed-profile.json').read_text())
for e in extra['entries']:
 if e['operation']=='quantized_experts' and e['parameters'].get('r')==512:
  entries[json.dumps([e['operation'],e['parameters']],sort_keys=True)]=e
seed=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider=provider,target=target,entries=list(entries.values())))
meta=dict(runtime='akas',purpose='preflight',protocol='native-token-step-v2',definition='akas.models.lfm2.model:LFM2MoE',model='akas.models.lfm2.model:LFM2MoE',provider=provider,target=target,mode='ar',kv_dtype='float16',weight_format='q4_0',checkpoint=str(checkpoint),checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',device=dict(name='NVIDIA L4',provider=provider),shape_timeout_seconds=120,warmup=1,repeats=1,graphs=True,qualification='independent operations; diagnostics and greedy IDs versus retained campaign reference; HTTP not measured',baseline_run_id=oldjob['run_id'],baseline_profile_status=oldjob['status'],expert_seed_run_id='20261019131816505070-605b9f45')
recorder=Recorder('benchmarks',meta,kind='algorithm',change='Manually optimize pipelined CUDA and vector-reusing WebGPU attention; validate bounded measurements without restarting campaign',parent=oldjob['run_id'])
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[])
passed=False
started=time.monotonic()
try:
 loaded=LFM2MoE.checkpoint_type(checkpoint,tokenizer=False)
 with tensor.Device(provider=provider,max_buffer_size=268435456 if provider=='webgpu' else None) as device:
  compile_program(loaded,root/'program',definition=LFM2MoE.compose,geometries=[(1,1),(1,32),(1,512)],max_requests=1,max_context=131072,provider=provider,target=target,device=device,model_options=dict(LFM2MoE.model_defaults),schedules=seed,search_settings=SearchSettings(budget=1,width=1,warmup=0,samples=1,repeats=1))
  meta['compilation_resources']=dict(seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  print('COMPILED',json.dumps(meta['compilation_resources']),flush=True)
  with LFM2MoE(checkpoint,root/'program',device,graphs=True) as executor:
   recipe=dict(chunk=512,window=None,prefill_requests=1)
   previous=np.load(old/'diagnostics.npz',allow_pickle=False)
   engine=Engine(executor,config=EngineConfig(prefill_chunk=512,prefill_requests=1))
   for prefix in (32,8192):
    case=Case(1,prefix,128);print('MEASURE',prefix,flush=True)
    try:
     values=diagnostic(executor,recipe,1,prefix,timeout=120)
     expected=previous[f'b1-p{prefix}'];relative=float(np.linalg.norm(values.astype('float64')-expected)/max(np.linalg.norm(expected),1e-30))
     if relative>0.01:raise ValueError(f'logit drift {relative} at P{prefix}')
     with Session(executor,batch=1,chunk=512,timeout=120) as session:
      row=run(session,case,warmup=1,repeats=1)
     measured=measured_requests(engine,1,prefix,128,timeout=120)
     oldrow=next(x for x in oldreport['results'] if x['case']['prompt']==prefix)
     if measured['output_ids']!=oldrow['output_ids']:raise ValueError(f'greedy mismatch P{prefix}')
     row.update(correctness=dict(correct=True,reference=oldjob['run_id'],relative_logit_error=relative),output_ids=measured.pop('output_ids'),engine=measured)
     report['results'].append(row);write(root/'report.json',report)
     print('RESULT',prefix,row['summary']['prefill_tokens_per_second']['median'],row['summary']['decode_tokens_per_second']['median'],'logit',relative,flush=True)
    except TimeoutError as e:
     report['results'].append(dict(case=asdict(case),status='timeout',error=str(e),samples=[],summary={}))
     print('SHAPE_TIMEOUT',prefix,str(e),flush=True);write(root/'report.json',report)
     break
   previous.close()
   engine.close()
 passed=len(report['results'])==2 and all(r['status']=='ok' for r in report['results'])
except BaseException as error:
 meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 if (root/'program/program.json').exists():recorder.program(root/'program')
 recorder.artifact(__file__,'preflight/runner.py')
 write(root/'report.json',report)
 recorder.update(report,final=True,status='passed' if passed else 'failed')
 print('RETAINED',recorder.id,'passed',passed,flush=True)
if not passed:sys.exit(1)
