"""Qualify the integrated codec path, precision probe, native phases and HTTP."""
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
from akas.optimization.evaluation import measured_requests,http_benchmark
from akas.serving.http import Service
from akas.engine import Engine
from akas.config import EngineConfig
from kv_quality import collect
root=Path('build/lfm-moe/codec-model/cuda');root.mkdir(parents=True,exist_ok=True)
old_program=Path('build/lfm-moe/fp8-model/cuda/program');program=root/'program'
parent='20261020141834495811-d8e958f0';reference=json.loads((Path('benchmarks/runs')/parent/'report.json').read_text())
checkpoint=Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf').resolve();loaded=LFM2MoE.checkpoint_type(checkpoint)
entries={}
for phase in read_program(old_program)['phases']:
 manifest=json.loads((old_program/phase['path']).read_text())
 for original in manifest.get('tuning',{}).get('profile',{}).get('entries',[]):
  entry=json.loads(json.dumps(original));p=entry['parameters'];schedule=entry['schedule'];kind=entry['operation']
  if kind=='quantized_experts' and p['r']==4096:schedule.update(block_m=64,columns=128,block_k=128,threads=256)
  if kind in ('linear','ffn') and p['r']==4096 and p.get('type')==2 and ((kind=='linear' and (p['k'],p['o'])==(2048,6144)) or (kind=='ffn' and (p['k'],p['o'])==(2048,11776))):
   schedule.update(block_m=64,block_n=128,block_k=128,threads=256)
  entries[json.dumps([kind,p],sort_keys=True)]=entry
seed=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=list(entries.values())))
write(root/'seed-profile.json',seed.data)
meta=dict(runtime='akas',purpose='preflight',protocol='native-token-step-v2',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),model='akas.models.lfm2.model:LFM2MoE',definition='akas.models.lfm2.model:LFM2MoE',checkpoint=str(checkpoint),checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',weight_format='q4_0',kv_dtype='fp8',mode='ar',chunk=4096,graphs=True,warmup=1,repeats=3,output_tokens=128,policy=dict(attention=dict(splits=64)),shape_timeout_seconds=120,cpu_reference_threads=4,qualification='Independent operation references and whole-model FP8 oracle regression; exact 128 FP8 IDs against retained program, Engine and HTTP slot reuse; paired F16 drift probe reported separately, not corpus accuracy',precision_reference_run_id='20261020134751392443-6b1662b8',same_precision_reference_run_id=parent,kernel_evidence_runs=['20261020141038109279-48b4394f','20261020142937738098-3f7d2afa','20261020144439270674-d3b0ff73','20261020145214095534-adf8104a'],performance_target=dict(prefill_speedup=2,decode_speedup=1.5,baselines=['20261018202355478915-6adad97d','20261018203703691867-28072a0a'],precision_comparison='FP8 KV versus F16 KV; report separately'))
rec=Recorder('benchmarks',meta,kind='algorithm',change='Qualify integrated packed-word expert, linear and paired FFN prefill through complete FP8 autoregressive serving',parent=parent)
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
@contextmanager
def factory():
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(checkpoint,program,device,graphs=True) as model:
   with Engine(model,config=EngineConfig(prefill_chunk=4096,prefill_requests=1)) as engine:
    yield Service(engine,model.checkpoint.tokenizer.encode,dict(model=meta['model'],provider='cuda',kv_dtype='fp8'))
try:
 started=time.monotonic();print('COMPILING_INTEGRATED_CODEC',flush=True)
 with tensor.Device(provider='cuda') as device:
  compile_program(loaded,program,definition=LFM2MoE.compose,geometries=[(1,1),(1,32),(1,4096)],max_requests=1,max_context=131072,provider='cuda',target='sm_89',device=device,model_options=dict(kv_dtype='fp8',optimized=True),schedules=seed,policy=meta['policy'],search_settings=SearchSettings(budget=1,width=1,warmup=0,samples=1,repeats=1))
  meta['compilation_resources']=dict(seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);print('COMPILED',json.dumps(meta['compilation_resources']),flush=True)
  with LFM2MoE(checkpoint,program,device,graphs=True) as model:
   with np.load('build/lfm-moe/fp8-model/cuda/precision-f16.npz',allow_pickle=False) as f16:
    values,checks=collect(model,loaded.tokenizer,chunk=4096,reference=f16)
   np.savez_compressed(root/'precision-fp8.npz',**values);meta['precision_comparison']=checks
   engine=Engine(model,config=EngineConfig(prefill_chunk=4096,prefill_requests=1))
   for prefix in (32,8192,130944):
    print('MEASURE_INTEGRATED',prefix,flush=True)
    with Session(model,batch=1,chunk=4096,timeout=120) as session:row=run(session,Case(1,prefix,128),warmup=1,repeats=3)
    report['results'].append(row);write(root/'report.json',report)
    measured=measured_requests(engine,1,prefix,128,timeout=120);expected=next(r for r in reference['results'] if r['case']['prompt']==prefix)['output_ids'];same=measured['output_ids']==expected
    row.update(output_ids=measured.pop('output_ids'),engine=measured,correctness=dict(correct=same,scope=meta['qualification']))
    print('RESULT',prefix,'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],'greedy',same,flush=True)
    write(root/'report.json',report)
    if not same:raise ValueError(f'Integrated codec greedy output differs at {prefix}')
 print('HTTP_INTEGRATED',flush=True)
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
 write(root/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
