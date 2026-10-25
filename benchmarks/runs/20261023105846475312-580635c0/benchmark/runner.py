"""Qualify an external F16 draft from compilation through portable HTTP deployment."""
import asyncio
from contextlib import ExitStack
import json
import os
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np
import tensor
from tensor.compiler.search import ScheduleProfile
from akas.artifacts.read import read_program
from akas.benchmarking.cases import Case
from akas.benchmarking.history import Recorder, write
from akas.benchmarking.runner import run
from akas.benchmarking.speculative import SpeculativeSession
from akas.compilation.compile import compile_program
from akas.compilation.policy import lower
from akas.compilation.tuning import SearchSettings
from akas.config import EngineConfig
from akas.engine import Engine
from akas.generation.speculative.decoding import SpeculativeDecoder
from akas.models.lfm2.model import LFM2MoE
from akas.models.dspark import DSparkDraft
from akas.models.dspark.proposer import load as load_head
from akas.models.dspark.weights import DSparkDraftCheckpoint
from akas.optimization.context import identity
from akas.optimization.deployment import describe, service
from akas.optimization.evaluation import http_benchmark, measured_requests
from dspark_reference import qualify

ROOT=Path('/home/sagemaker-user/akas')
OUT=ROOT/'build/lfm-moe/dspark-long-qualification'
OUT.mkdir(parents=True,exist_ok=True)
PARENT='20261023101615391936-42d4989e'
prior=json.loads((ROOT/'benchmarks/runs'/PARENT/'run.json').read_text())
if prior['status']!='passed': raise RuntimeError('The prepared F16 AR qualification must pass first')
REFERENCE=json.loads((ROOT/'benchmarks/runs'/PARENT/'report.json').read_text())
PREVIOUS=ROOT/'build/lfm-moe/cuda-f16-prepared-full/cuda/program'
CHECKPOINT=ROOT/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
DRAFT=CHECKPOINT.with_name('dspark-f16.gguf')
TARGET_PROGRAM,DRAFT_PROGRAM=OUT/'target',OUT/'draft'
PREFIXES=(32,8192,130944)
CHUNKS=(1,32,512,2048,4096)
POLICY=dict(attention=dict(splits=64,decoded_kv=False,native_tail=True),moe=dict(fused_swiglu=True))
loaded=LFM2MoE.checkpoint_type(CHECKPOINT)
sidecar=DSparkDraftCheckpoint(DRAFT,target=loaded)
meta=dict(runtime='akas',purpose='benchmark',protocol='native-token-step-v2',provider='cuda',target='sm_89',
    device=dict(name='NVIDIA L4',provider='cuda'),model='akas.models.lfm2.model:LFM2MoE',
    definition='akas.models.lfm2.model:LFM2MoE',mode='dspark',native_phase_mode='dspark',
    checkpoint=str(CHECKPOINT),checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',
    draft_checkpoint=str(DRAFT),draft_checkpoint_sha256='0cd847ca049aeabf53cb41d62091ae20eb50245cf89594c75af204784a1b1a0f',
    weight_format='q4_0',weight_dtype='gguf-q4_0-mixed',draft_weight_dtype='float16',
    kv_dtype='float16',draft_kv_dtype='float16',activation_dtype='float32',chunk=4096,
    window=10,block_size=9,confidence_threshold=0,graphs=True,round_capture=False,
    requested_prefixes=list(PREFIXES),output_tokens=128,ignore_eos=True,warmup=1,repeats=3,
    policy=POLICY,shape_timeout_seconds=120,ar_reference_run_id=PARENT,
    target_layers=list(sidecar.config.target_layers),
    qualification='Independent operation references and full-draft NumPy reference; exact native block coordinator/Engine/portable deployment HTTP output consistency. F16 target and draft KV. AR differences retained separately; not a corpus accuracy claim. Three predeclared context gates, not the full eight-shape performance matrix.')
rec=Recorder(ROOT/'benchmarks',meta,kind='algorithm',parent=PARENT,
    change='Qualify paired F16 DSpark through public composition, shared target features, actual block timing and checksummed HTTP deployment at short, 8K and maximum context')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[])
write(OUT/'run-id.json',dict(run_id=rec.id))
rec.update(report)
print('RUN',rec.id,flush=True)
settings=SearchSettings(budget=1,width=1,warmup=0,samples=1,repeats=1)

# Cache identity still checks all operation, compiler and device fingerprints.
# Hard links retain bounded storage without bypassing reference qualification.
TARGET_PROGRAM.mkdir(exist_ok=True)
def link_cached(source,destination):
    try:os.link(source,destination)
    except FileExistsError:pass
for name in ('compiler-cache','search-cache','search-kernels'):
    source=PREVIOUS/name
    if source.is_dir(): shutil.copytree(source,TARGET_PROGRAM/name,copy_function=link_cached,dirs_exist_ok=True)

entries=[]
for phase in read_program(PREVIOUS)['phases']:
    data=json.loads((PREVIOUS/phase['path']).read_text())
    entries.extend(data['tuning']['profile']['entries'])
def key(p,drop_rows=False):
    p=json.loads(json.dumps(p))
    if drop_rows:p.pop('r',None)
    return json.dumps(p,sort_keys=True)
exact={(e['operation'],key(e['parameters'])):e['schedule'] for e in entries}
short={(e['operation'],key(e['parameters'],True)):e['schedule'] for e in entries if e['parameters'].get('r')==32}
seed_entries={}
for chunk,phase in [*((chunk,None) for chunk in CHUNKS),(10,'verify')]:
    composed=LFM2MoE.compose(loaded,rows=1,chunk=chunk,max_requests=1,max_context=131072,
        provider='cuda',phase=phase,kv_dtype='float16',auxiliary_layers=sidecar.config.target_layers)
    graph,_=lower(composed.graph,composed.commit,policy=POLICY,provider='cuda',target='sm_89')
    for call in graph.calls:
        axes=exact.get((call.kind,key(call.parameters)))
        if axes is None and chunk==10 and call.kind in ('linear','quantized_experts'):
            axes=short.get((call.kind,key(call.parameters,True)))
        if call.kind=='attention':
            axes=dict(implementation='attention.cuda.ordered' if call.parameters.get('staged_kv') else 'attention.cuda.pipelined')
        if axes is not None:
            seed_entries[call.key]=dict(operation=call.kind,parameters=call.parameters,schedule=axes)
seed=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=list(seed_entries.values())))
write(OUT/'target-seed.json',seed.data)
workload=dict(id='single-user-dspark-context-gates-v1',batches=[1],prefix_tokens=list(PREFIXES),output_tokens=128)
context=dict(checkpoint=str(CHECKPOINT),definition=meta['definition'],workload=workload,
    backend='cuda',device=0,memory_bytes=None,ttft_ms=None,tpot_ms=None,kv_dtype='float16',mode='dspark')
recipe=dict(provider='cuda',target='sm_89',chunk=4096,prefill_requests=1,graphs=True,trials=1,
    policy=POLICY,model_options=dict(kv_dtype='float16',auxiliary_layers=list(sidecar.config.target_layers)),
    window=10,deferred=False,round_capture=False,head='akas.models.dspark.proposer:load',
    head_options=dict(checkpoint=str(DRAFT),bundle=str(DRAFT_PROGRAM)),mode='dspark')
recipe['id']=identity(recipe)[:16]
campaign_identity=identity(dict(source=rec.record['source']['tree_sha256'],recipe=recipe,run=rec.id))
passed=False
try:
    with tensor.Device(provider='cuda') as device:
        for label,reader,definition,geometries,options,policy,schedules in (
            ('target',loaded,LFM2MoE.compose,[*((1,c) for c in CHUNKS),(1,10,'verify')],
             recipe['model_options'],POLICY,seed),
            ('draft',sidecar,DSparkDraft.compose,[*((1,c,'context') for c in (*CHUNKS,10)),(1,9,'proposal')],
             dict(kv_dtype='float16'),None,None)):
            path=TARGET_PROGRAM if label=='target' else DRAFT_PROGRAM
            print('COMPILE',label,flush=True);start=time.monotonic()
            compile_program(reader,path,definition=definition,geometries=geometries,
                max_requests=1,max_context=131072,provider='cuda',target='sm_89',device=device,
                model_options=options,policy=policy,schedules=schedules,search_settings=settings)
            meta[label+'_compilation']=dict(seconds=time.monotonic()-start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            rec.program(path);rec.update(report)
            print('COMPILED',label,json.dumps(meta[label+'_compilation']),flush=True)
        with ExitStack() as owned:
            target=owned.enter_context(LFM2MoE(CHECKPOINT,TARGET_PROGRAM,device,graphs=True))
            head=load_head(target,checkpoint=DRAFT,bundle=DRAFT_PROGRAM,graphs=True)
            owned.callback(head.close)
            meta['full_draft_reference']=qualify(target,head.draft)
            print('DRAFT_REFERENCE',json.dumps(meta['full_draft_reference']),flush=True)
            backend=SpeculativeDecoder(target,head,window=10,capture=False)
            with Engine(backend,config=EngineConfig(prefill_chunk=4096,prefill_requests=1)) as engine:
                for prefix in PREFIXES:
                    print('MEASURE',prefix,flush=True)
                    with SpeculativeSession(backend,batch=1,chunk=4096,timeout=120) as session:
                        row=run(session,Case(1,prefix,128),warmup=1,repeats=3)
                        native=[list(r.output) for r in session.requests]
                        row['native_speculation']=session.stats
                    expected=next(r['output_ids'] for r in REFERENCE['results'] if r['case']['prompt']==prefix)
                    row['native_output_ids']=native
                    row['ar_reference']=dict(run_id=PARENT,exact=native==expected,
                        common_prefix_tokens=next((i for i,(a,b) in enumerate(zip(native[0],expected[0])) if a!=b),128))
                    row['native_phase_mode']='dspark'
                    row['mode']='dspark'
                    row['output_ids']=native
                    report['results'].append(row);rec.update(report)
                    engine_samples=[]
                    before=dict(backend.stats)
                    for iteration in range(4):
                        measured=measured_requests(engine,1,prefix,128,timeout=120)
                        ids=measured.pop('output_ids')
                        if ids!=native:raise ValueError(f'Native block/Engine outputs differ at {prefix}')
                        if iteration:engine_samples.append(measured)
                    row['engine']=dict(samples=engine_samples)
                    row['serving']=dict(samples=engine_samples)
                    row['engine_speculation']={name:value-before[name] for name,value in backend.stats.items()}
                    row['correctness']=dict(correct=True,scope='Independent operations, NumPy draft and native block/Engine outputs; HTTP gate follows')
                    print('RESULT',prefix,'pp',row['summary']['prefill_tokens_per_second']['median'],
                        'tg',row['summary']['decode_tokens_per_second']['median'],
                        'speculation',json.dumps(row['native_speculation']),flush=True)
                    rec.update(report)
    deployment=describe(context,recipe,TARGET_PROGRAM,checkpoint_sha256=meta['checkpoint_sha256'],
        campaign_identity=campaign_identity,run_id=rec.id,hardware=meta['device'])
    if deployment['recipe']['head_options']['checkpoint_sha256'] != meta['draft_checkpoint_sha256']:
        raise ValueError('Draft checkpoint differs from pinned identity')
    # describe publishes paths relative to target's parent, i.e. OUT.
    deployment_path=OUT/'deployment.json'
    write(deployment_path,deployment)
    os.environ['AKAS_SHAPE_TIMEOUT']='120'
    os.environ['AKAS_BENCHMARK_WARMUP']='1'
    os.environ['AKAS_BENCHMARK_REPEATS']='3'
    print('HTTP_PORTABLE_DEPLOYMENT',flush=True)
    meta['http']=asyncio.run(http_benchmark(lambda:service(deployment_path,candidate=True),report['results'],vocab=128000))
    for row in report['results']:row['correctness']['scope']=meta['qualification']
    deployment['status']='qualified';write(deployment_path,deployment)
    passed=True
except BaseException as error:
    meta['error']=f'{type(error).__name__}: {error}'
    meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
    meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    for program in (TARGET_PROGRAM,DRAFT_PROGRAM):
        if (program/'program.json').exists():rec.program(program)
    for path,name in ((Path(__file__),'benchmark/runner.py'),
        (Path(__file__).with_name('dspark_reference.py'),'validation/reference.py'),
        (OUT/'target-seed.json','benchmark/target-seed.json'),
        (OUT/'deployment.json','serving/deployment.json')):
        if path.exists():rec.artifact(path,name)
    write(OUT/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed')
    print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

# Publish through the same fixed report format as all previous measurements.
from tools.campaign_report import export
profile_id='lfm2.5-24b-a2b-q4_0-cuda-float16-dspark'
model=json.loads((ROOT/'benchmarks/models/lfm2.5-24b-a2b.json').read_text())
model['modes']=['dspark']
entry=dict(id=profile_id,context=context,status='complete',deployment=str(deployment_path),
    winner=dict(run_id=rec.id,recipe=recipe['id']),spent_seconds=0)
campaign_root=OUT/'campaign'
state=dict(schema='akas.matrix-campaign.v1',identity=campaign_identity,
    specification=dict(schema='akas.matrix-campaign.v1',models=['lfm2.5-24b-a2b'],
        formats=['q4_0'],backends=['cuda'],caches=['float16'],workload=workload['id'],execution='fixed-context-gates'),
    models={'lfm2.5-24b-a2b':model},source=rec.record['source'],created_at=rec.record['started_at'],
    updated_at=rec.record['completed_at'],status='complete',entries=[entry],spent_seconds=0)
write(campaign_root/profile_id/'campaign.json',dict(jobs=[dict(recipe=recipe,
    directory=str(OUT),run_id=rec.id,status='passed',elapsed_seconds=0)]))
write(campaign_root/'campaign.json',state)
destination=export(campaign_root/'campaign.json',history=ROOT/'benchmarks')
write(OUT/'publication.json',dict(report=str(destination),deployment=str(deployment_path)))
print('PUBLISHED',destination,flush=True)
