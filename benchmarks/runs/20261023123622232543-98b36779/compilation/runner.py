"""Compile qualified short-context adaptive verification and measured projections."""
from contextlib import ExitStack
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
from akas.models.dspark.model import DSparkDraft
from akas.models.dspark.weights import DSparkDraftCheckpoint
root=Path('/home/sagemaker-user/akas');old=root/'build/lfm-moe/dspark-long-qualification';out=root/'build/lfm-moe/dspark-short-optimized';out.mkdir(exist_ok=True)
checkpoint=root/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf';draft_path=checkpoint.with_name('dspark-f16.gguf')
reader=LFM2MoE.checkpoint_type(checkpoint);draft=DSparkDraftCheckpoint(draft_path,target=reader)
meta=dict(runtime='akas',purpose='compilation',provider='cuda',target='sm_89',max_context=131072,kv_dtype='float16',draft_kv_dtype='float16',verify_chunks=[1,2,4,10],checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',qualification='Ordinary public operation qualification, inherited source/device-bound caches, measured bitwise short projection geometries. Target and draft context capacity retained at its already-qualified 131072 geometry; prompts restricted to the short-only first gate. Full model measurements follow compilation.')
rec=Recorder(root/'benchmarks',meta,kind='geometry',parent='20261023122750183156-12560ca0',change='Prepare smaller target verification windows with matching draft feature contexts and qualified short projection schedules')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);write(out/'run-id.json',dict(run_id=rec.id));print('RUN',rec.id,flush=True);passed=False
policy=dict(attention=dict(splits=64,decoded_kv=False,native_tail=True),moe=dict(fused_swiglu=True))
settings=SearchSettings(budget=1,width=1,warmup=0,samples=1,repeats=1)
def key(p,drop_rows=False):
 p=dict(p)
 if drop_rows:p.pop('r',None)
 return json.dumps(p,sort_keys=True)
try:
 with tensor.Device(provider='cuda') as device:
  for label,loaded,compose,geometries,options,chosen_policy in (
   ('target',reader,LFM2MoE.compose,[(1,c) for c in (1,32,512,2048,4096)]+[(1,c,'verify') for c in (1,2,4,10)],dict(kv_dtype='float16',auxiliary_layers=draft.config.target_layers),policy),
   ('draft',draft,DSparkDraft.compose,[(1,c,'context') for c in (1,2,4,10,32,512,2048,4096)]+[(1,9,'proposal')],dict(kv_dtype='float16'),None)):
   destination=out/label;destination.mkdir(exist_ok=True)
   def cached_link(source,dest):
    if not Path(dest).exists():os.link(source,dest)
   for name in ('compiler-cache','search-cache','search-kernels'):
    if (old/label/name).is_dir():shutil.copytree(old/label/name,destination/name,copy_function=cached_link,dirs_exist_ok=True)
   entries=[]
   for phase in read_program(old/label)['phases']:
    entries.extend(json.loads((old/label/phase['path']).read_text())['tuning']['profile']['entries'])
   exact={(e['operation'],key(e['parameters'])):e['schedule'] for e in entries}
   small={(e['operation'],key(e['parameters'],True)):e['schedule'] for e in entries if e['parameters'].get('r') in (9,10)}
   seed={}
   for geometry in geometries:
    rows,chunk,*phase=geometry;phase=phase[0] if phase else None
    composition=compose(loaded,rows=rows,chunk=chunk,max_requests=1,max_context=131072,provider='cuda',phase=phase,**options)
    graph,_=lower(composition.graph,composition.commit,policy=chosen_policy,provider='cuda',target='sm_89')
    for call in graph.calls:
     p=call.parameters;axes=exact.get((call.kind,key(p)))
     if axes is None and p.get('r',0)>1:axes=small.get((call.kind,key(p,True)))
     if call.kind=='linear' and p.get('ggml') and 1<p['r']<=10:
      bm,bn,bk,stages=(16,64,64,2) if p['type']==14 else (16,64,128,1)
      axes=dict(implementation='linear.cuda.packed:prefill',block_m=bm,block_n=bn,block_k=bk,stages=stages,threads=128)
     if call.kind=='quantized_experts' and 1<p['r']<=10:
      axes=dict(implementation='akas.ops.moe.cuda:compact_experts',block_m=16,columns=64 if p.get('routed_input') else 32,block_k=128,threads=128)
     if call.kind=='attention' and p.get('chunk',1)>1:
      axes=dict(implementation='attention.cuda.ordered' if p.get('staged_kv') else 'attention.cuda.pipelined')
     if axes is not None:seed[call.key]=dict(operation=call.kind,parameters=p,schedule=axes)
   profile=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=list(seed.values())))
   write(out/(label+'-seed.json'),profile.data);print('COMPILE',label,flush=True);start=time.monotonic()
   compile_program(loaded,destination,definition=compose,geometries=geometries,max_requests=1,max_context=131072,provider='cuda',target='sm_89',model_options=options,policy=chosen_policy,schedules=profile,search_settings=settings,device=device)
   meta[label+'_seconds']=time.monotonic()-start;rec.program(destination);rec.update(report);print('COMPILED',label,meta[label+'_seconds'],flush=True)
 passed=True
except BaseException as error:meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 rec.artifact(__file__,'compilation/runner.py')
 for name in ('target','draft'):
  if (out/(name+'-seed.json')).exists():rec.artifact(out/(name+'-seed.json'),'compilation/'+name+'-seed.json')
 write(out/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
