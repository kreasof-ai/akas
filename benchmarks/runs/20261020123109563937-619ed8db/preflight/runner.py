"""Compile the integrated algorithms, then compare bounded prefill chunk paths."""
import json,resource,time,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.compiler.search import ScheduleProfile
from akas.models.lfm2.model import LFM2MoE
from akas.compilation.compile import compile_program
from akas.compilation.tuning import SearchSettings
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.benchmarking.cases import Case
from akas.optimization.evaluation import diagnostic,measured_requests
from akas.engine import Engine
from akas.config import EngineConfig

root=Path('build/lfm-moe/manual-v3/cuda');root.mkdir(parents=True,exist_ok=True)
old=Path('build/single-user-long-context/lfm2.5-24b-a2b-q4_0-cuda-default-ar/jobs/ee9b3acb399892d0/attempt-0001')
checkpoint=Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf').resolve()
entries={}
for phase in json.loads((old/'program/program.json').read_text())['phases']:
    f=json.loads((old/'program'/phase['path']).read_text())
    for e in f.get('tuning',{}).get('profile',{}).get('entries',[]):
        op,p=e['operation'],e['parameters']
        if op in ('attention','attention_partial','attention_merge'):continue
        if op=='quantized_experts' and p['r']==1:continue
        if op=='linear' and 'type' not in p:continue
        entries[json.dumps([op,p],sort_keys=True)]=e

loaded=LFM2MoE.checkpoint_type(checkpoint,tokenizer=False)
for chunk in (512,4096):
    graph=LFM2MoE.compose(loaded,rows=1,chunk=chunk,max_requests=1,max_context=131072,provider='cuda').graph
    for key,(kind,p) in graph.requirements().items():
        schedule=None
        if kind=='quantized_experts':
            schedule=dict(implementation='akas.ops.moe.cuda:compact_experts',block_m=64,
                          columns=128 if chunk==512 else 64,block_k=128 if chunk==512 else 64)
        elif chunk==4096 and kind in ('linear','ffn') and p.get('type')==2 and p.get('r',1)>1:
            schedule=dict(implementation='linear.cuda.packed:prefill',block_m=64,block_n=64,block_k=64,stages=2,threads=128)
        if schedule is not None:
            entries[json.dumps([kind,p],sort_keys=True)]=dict(operation=kind,parameters=p,schedule=schedule)
seed=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=list(entries.values())))
write(root/'seed-profile.json',seed.data)
meta=dict(runtime='akas',purpose='preflight',protocol='native-token-step-v2',
          model='akas.models.lfm2.model:LFM2MoE',definition='akas.models.lfm2.model:LFM2MoE',
          checkpoint=str(checkpoint),checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',
          provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),
          weight_format='q4_0',kv_dtype='float16',mode='ar',graphs=True,warmup=1,repeats=3,
          shape_timeout_seconds=120,cpu_reference_threads=4,
          qualification='Independent operation qualification, complete compiled model diagnostics and exact 128 greedy IDs versus retained reference; HTTP not measured',
          baseline_run_id='20261020102328416670-be712fbb',geometries=[[1,1],[1,32],[1,512],[1,4096]])
rec=Recorder('benchmarks',meta,kind='algorithm',change='Integrate packed expert dots, native GQA tensor-core decode, F32 warp routers; qualify larger prefill chunks',parent=meta['baseline_run_id'])
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[])
passed=False
try:
    started=time.monotonic();print('COMPILING',flush=True)
    with tensor.Device(provider='cuda') as device:
        compile_program(loaded,root/'program',definition=LFM2MoE.compose,
                        geometries=[(1,1),(1,32),(1,512),(1,4096)],max_requests=1,max_context=131072,
                        provider='cuda',target='sm_89',device=device,model_options=dict(LFM2MoE.model_defaults),
                        schedules=seed,search_settings=SearchSettings(budget=1,width=1,warmup=0,samples=1,repeats=1))
        meta['compilation_resources']=dict(seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        print('COMPILED',json.dumps(meta['compilation_resources']),flush=True)
        oldreport=json.loads((old/'report.json').read_text());expected=np.load(old/'diagnostics.npz',allow_pickle=False)
        with LFM2MoE(checkpoint,root/'program',device,graphs=True) as executor:
            trials=[]
            for chunk in (512,4096):
                print('PREFILL_TRIAL',chunk,flush=True)
                values=diagnostic(executor,dict(chunk=chunk),1,8192,timeout=120)
                ref=expected['b1-p8192'];relative=float(np.linalg.norm(values.astype('float64')-ref)/max(np.linalg.norm(ref),1e-30))
                if relative>.01:raise ValueError(f'chunk {chunk}: logit drift {relative}')
                with Session(executor,batch=1,chunk=chunk,timeout=120) as session:
                    row=run(session,Case(1,8192,128),warmup=1,repeats=3)
                trials.append(dict(chunk=chunk,relative_logit_error=relative,native=row))
                print('TRIAL_RESULT',chunk,row['summary']['prefill_tokens_per_second']['median'],row['summary']['decode_tokens_per_second']['median'],'relative',relative,flush=True)
            write(root/'chunk-trials.json',trials);meta['chunk_trials']=trials
            winner=max(trials,key=lambda v:v['native']['summary']['prefill_tokens_per_second']['median'])
            chunk=winner['chunk'];meta['chunk']=chunk
            engine=Engine(executor,config=EngineConfig(prefill_chunk=chunk,prefill_requests=1))
            for prefix in (32,8192,130944):
                print('MEASURE',chunk,prefix,flush=True)
                values=diagnostic(executor,dict(chunk=chunk),1,prefix,timeout=120)
                ref=expected[f'b1-p{prefix}'];relative=float(np.linalg.norm(values.astype('float64')-ref)/max(np.linalg.norm(ref),1e-30))
                if relative>.01:raise ValueError(f'logit drift {relative} at {prefix}')
                print('DIAGNOSTIC_OK',prefix,relative,flush=True)
                if prefix==8192:row=winner['native']
                else:
                    with Session(executor,batch=1,chunk=chunk,timeout=120) as session:
                        row=run(session,Case(1,prefix,128),warmup=1,repeats=3)
                print('NATIVE_OK',prefix,flush=True)
                measured=measured_requests(engine,1,prefix,128,timeout=120)
                oldrow=next(r for r in oldreport['results'] if r['case']['prompt']==prefix)
                if measured['output_ids']!=oldrow['output_ids']:raise ValueError(f'greedy mismatch at {prefix}')
                row.update(correctness=dict(correct=True,relative_logit_error=relative,reference=meta['baseline_run_id']),output_ids=measured.pop('output_ids'),engine=measured)
                report['results'].append(row);write(root/'report.json',report)
                print('RESULT',prefix,'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],flush=True)
        expected.close()
    passed=True
except BaseException as error:
    meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
    meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if (root/'program/program.json').exists():rec.program(root/'program')
    rec.artifact(__file__,'preflight/runner.py');rec.artifact(root/'seed-profile.json','preflight/seed-profile.json')
    if (root/'chunk-trials.json').exists():rec.artifact(root/'chunk-trials.json','preflight/chunk-trials.json')
    write(root/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed')
    print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
