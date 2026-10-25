"""Qualify compiler-selected shared FP8 staging through native and HTTP serving."""
import asyncio,json,os,resource,time,traceback
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import tensor
from tensor.compiler.search import ScheduleProfile
from akas.models.lfm2.model import LFM2MoE
from akas.compilation.compile import compile_program
from akas.compilation.tuning import SearchSettings
from akas.artifacts.read import read_program
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.benchmarking.cases import Case
from akas.optimization.evaluation import measured_requests,http_benchmark,prompts
from akas.serving.http import Service
from akas.engine import Engine
from akas.config import EngineConfig
from kv_quality import collect
root=Path('build/lfm-moe/staged-model/cuda');root.mkdir(parents=True,exist_ok=True)
old_program=Path('build/lfm-moe/codec-model/cuda/program');program=root/'program'
parent='20261020152431734974-3777f95e';reference=json.loads((Path('benchmarks/runs')/parent/'report.json').read_text())
checkpoint=Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf').resolve();loaded=LFM2MoE.checkpoint_type(checkpoint)
entries={}
for phase in read_program(old_program)['phases']:
 manifest=json.loads((old_program/phase['path']).read_text())
 for original in manifest.get('tuning',{}).get('profile',{}).get('entries',[]):
  entry=json.loads(json.dumps(original));p=entry['parameters'];schedule=entry['schedule'];kind=entry['operation']
  if kind=='quantized_experts' and p['r']==4096:schedule.update(block_m=64,columns=128,block_k=128,threads=256)
  if kind in ('linear','ffn') and p['r']==4096 and p.get('type')==2 and ((kind=='linear' and (p['k'],p['o'])==(2048,6144)) or (kind=='ffn' and (p['k'],p['o'])==(2048,11776))):
   schedule.update(block_m=64,block_n=128,block_k=128,threads=256)
  if kind=='attention' and p['r']==4096:
   p.update(kv_dtype='float16',unit_scales=True,staged_kv=True)
   schedule.clear();schedule.update(implementation='attention.cuda.pipelined',async_loads=True,query_rows=32,key_rows=64,stages=1,threads=128)
  entries[json.dumps([kind,p],sort_keys=True)]=entry
seed=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=list(entries.values())))
write(root/'seed-profile.json',seed.data)
meta=dict(runtime='akas',purpose='preflight',protocol='native-token-step-v2',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),model='akas.models.lfm2.model:LFM2MoE',definition='akas.models.lfm2.model:LFM2MoE',checkpoint=str(checkpoint),checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',weight_format='q4_0',kv_dtype='fp8',mode='ar',chunk=4096,graphs=True,warmup=1,repeats=3,output_tokens=128,policy=dict(attention=dict(splits=64,decoded_kv=True)),shape_timeout_seconds=120,cpu_reference_threads=4,qualification='Integrated compiler-owned decoded-cache pipeline, independent operation references, paired F16 drift probe, and exact 128 FP8 token IDs across native/Engine/HTTP. One temporary FP16 KV pair per compatible phase; persistent FP8 and decode unchanged. Includes conversion and temporary memory. Not corpus accuracy.',precision_reference_run_id='20261020134751392443-6b1662b8',same_precision_reference_run_id=parent,kernel_evidence_runs=['20261020170127736154-28a52ca6','20261020162743785090-f439dad8','20261020163304124890-b70dbebc'],performance_target=dict(prefill_speedup=2,decode_speedup=1.5,baselines=['20261018202355478915-6adad97d','20261018203703691867-28072a0a'],precision_comparison='FP8 KV versus F16 KV; report separately'))
rec=Recorder('benchmarks',meta,kind='algorithm',change='Qualify compiler-owned shared decoded FP8 prefill through native execution and HTTP serving',parent=parent)
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
@contextmanager
def factory():
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(checkpoint,program,device,graphs=True) as model:
   with Engine(model,config=EngineConfig(prefill_chunk=4096,prefill_requests=1)) as engine:
    yield Service(engine,model.checkpoint.tokenizer.encode,dict(model=meta['model'],provider='cuda',kv_dtype='fp8'))
try:
 started=time.monotonic();print('COMPILING_INTEGRATED_STAGING',flush=True)
 with tensor.Device(provider='cuda') as device:
  compile_program(loaded,program,definition=LFM2MoE.compose,geometries=[(1,1),(1,32),(1,4096)],max_requests=1,max_context=131072,provider='cuda',target='sm_89',device=device,model_options=dict(kv_dtype='fp8',optimized=True),schedules=seed,policy=meta['policy'],search_settings=SearchSettings(budget=1,width=1,warmup=0,samples=1,repeats=1))
  meta['compilation_resources']=dict(seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);print('COMPILED',json.dumps(meta['compilation_resources']),flush=True)
  with LFM2MoE(checkpoint,program,device,graphs=True) as model:
   with np.load('build/lfm-moe/fp8-model/cuda/precision-f16.npz',allow_pickle=False) as f16:
    values,checks=collect(model,loaded.tokenizer,chunk=4096,reference=f16)
   np.savez_compressed(root/'precision-fp8.npz',**values);meta['precision_comparison']=checks
   with np.load('build/lfm-moe/codec-model/cuda/precision-fp8.npz',allow_pickle=False) as previous:
    meta['same_precision_logit_comparison']=[dict(name=name,bitwise=bool(np.array_equal(value,previous[name])),max_absolute_error=float(np.max(np.abs(value.astype('float64')-previous[name])))) for name,value in values.items()]
   print('PRECISION',json.dumps(meta['precision_comparison']),flush=True)
   engine=Engine(model,config=EngineConfig(prefill_chunk=4096,prefill_requests=1))
   for prefix in (32,8192,130944):
    print('MEASURE_INTEGRATED_STAGING',prefix,flush=True)
    with Session(model,batch=1,chunk=4096,timeout=120) as session:
     row=run(session,Case(1,prefix,128),warmup=1,repeats=3)
     session.reset();native_ids=list(session.prefill(prompts(model.config.vocab,1,prefix)))
     for _ in range(127):native_ids.extend(session.decode(1))
    report['results'].append(row);write(root/'report.json',report)
    measured=measured_requests(engine,1,prefix,128,timeout=120);expected=next(r for r in reference['results'] if r['case']['prompt']==prefix)['output_ids'];same=measured['output_ids']==expected
    native_same=[native_ids]==expected;row.update(native_output_ids=[native_ids],native_correctness=dict(correct=native_same,reference_run_id=parent))
    row.update(output_ids=measured.pop('output_ids'),serving=measured,correctness=dict(correct=same and native_same,scope=meta['qualification']))
    print('RESULT',prefix,'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],'greedy',same,flush=True)
    write(root/'report.json',report)
    if not (same and native_same):raise ValueError(f'Integrated staging greedy output differs at {prefix}: native={native_same}, Engine={same}')
 print('HTTP_INTEGRATED_STAGING',flush=True)
 os.environ['AKAS_SHAPE_TIMEOUT']='120';os.environ['AKAS_BENCHMARK_WARMUP']='1';os.environ['AKAS_BENCHMARK_REPEATS']='3'
 meta['http']=asyncio.run(http_benchmark(factory,report['results'],vocab=128000));print('HTTP_PASSED',json.dumps(meta['http']),flush=True)
 passed=True
except BaseException as error:
 meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 if (program/'program.json').exists():rec.program(program)
 for path,name in [(Path(__file__),'preflight/runner.py'),(root/'seed-profile.json','preflight/seed-profile.json'),(Path('build/lfm-moe/kv_quality.py'),'precision/probe.py'),(root/'precision-fp8.npz','precision/fp8.npz')]:
  if path.exists():rec.artifact(path,name)
 for name in ('staging-full-tests.log','staging-final-focused-cpu-tests.log','staging-cuda-tests.log','staging-pipeline-cuda-tests.log','staging-webgpu-tests-v2.log','staging-compiler-cuda-tests-v2.log'):
  path=Path('build/lfm-moe')/name
  if path.exists():rec.artifact(path,'validation/'+name)
 write(root/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
