"""Compile and qualify FP8, compare its precision with F16, then measure serving."""
import asyncio,json,os,resource,subprocess,time,traceback
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import tensor
from tensor.compiler.search import ScheduleProfile
from akas.models.lfm2.model import LFM2MoE
from akas.compilation.compile import compile_program
from akas.compilation.tuning import SearchSettings
from akas.compilation.policy import lower
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.benchmarking.cases import Case
from akas.optimization.evaluation import diagnostic,measured_requests,http_benchmark
from akas.execution.executor import Executor
from akas.execution.batch import BatchBank
from akas.execution.plan import ExecutionPlan,call_bindings
from akas.tensor.executor import bind
from akas.artifacts.read import read_program
from akas.engine import Engine
from akas.config import EngineConfig
from kv_quality import collect
root=Path('build/lfm-moe/fp8-model/cuda');root.mkdir(parents=True,exist_ok=True)
reference_program=Path('build/lfm-moe/manual-v3/cuda/program')
checkpoint=Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf').resolve()
policy=dict(attention=dict(splits=64));options=dict(kv_dtype='fp8',optimized=True)
loaded=LFM2MoE.checkpoint_type(checkpoint)
entries={}
for phase in read_program(reference_program)['phases']:
 manifest=json.loads((reference_program/phase['path']).read_text())
 for entry in manifest.get('tuning',{}).get('profile',{}).get('entries',[]):
  if entry['operation'] in ('attention','attention_partial','attention_merge','cache_write','quantized_routes'):continue
  entries[json.dumps([entry['operation'],entry['parameters']],sort_keys=True)]=entry
# Bind the measured split/decode schedule to actual FP8 operation requirements.
for chunk in (1,32,4096):
 composed=LFM2MoE.compose(loaded,rows=1,chunk=chunk,max_requests=1,max_context=131072,provider='cuda',**options)
 graph,_=lower(composed.graph,composed.commit,policy=policy,provider='cuda',target='sm_89')
 for kind,p in graph.requirements().values():
  if kind=='attention_partial':
   entries[json.dumps([kind,p],sort_keys=True)]=dict(operation=kind,parameters=p,schedule=dict(implementation='attention.cuda.mma_decode',key_rows=64,stages=1))
seed=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_89',entries=list(entries.values())))
write(root/'seed-profile.json',seed.data)
meta=dict(runtime='akas',purpose='preflight',protocol='native-token-step-v2',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),model='akas.models.lfm2.model:LFM2MoE',definition='akas.models.lfm2.model:LFM2MoE',checkpoint=str(checkpoint),checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',weight_format='q4_0',kv_dtype='fp8',mode='ar',chunk=4096,graphs=True,warmup=1,repeats=3,output_tokens=128,policy=policy,shape_timeout_seconds=120,cpu_reference_threads=4,qualification='Independent operation references plus whole-model FP8 NumPy oracle, exact FP8 greedy IDs across HTTP slot reuse; paired F16 precision probe reported separately, not required to have identical cross-precision IDs',reference_program_run_id='20261020123109563937-619ed8db',f16_reference_expert_overlay='continuous packed-view; exact full-context 128 IDs verified in 20261020130827106502-0ef28491',precision_probe='Fixed prose repeated to prefix length, eight identical F16 teacher-forced continuation tokens; numerical drift probe, not corpus accuracy/perplexity')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Qualify fused FP8 KV decode/prefill through complete autoregressive serving')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False

def borrow(owner,directory):
 program=read_program(directory);weights={name:value.view(value.shape) for name,value in owner.weights.items()}
 bank=BatchBank(directory,owner.device,program['phases'],weights,graphs=True,reset=program.get('reset'))
 executor=Executor(capacity=program['max_requests'],context=program['max_context'],device=owner.device,batches=bank,reset=bank.reset)
 executor.checkpoint=owner.checkpoint;executor.config=owner.config;executor.program=program;executor.bundle=directory;executor.weights=weights
 return executor

def telemetry():
 result=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return dict(zip(('temperature_c','sm_mhz','memory_mhz','watts','power_limit_watts','memory_used_mib'),map(float,result.stdout.strip().split(','))))
try:
 started=time.monotonic();print('COMPILING_FP8',flush=True)
 with tensor.Device(provider='cuda') as device:
  compile_program(loaded,root/'program',definition=LFM2MoE.compose,
                  geometries=[(1,1),(1,32),(1,4096)],max_requests=1,max_context=131072,
                  provider='cuda',target='sm_89',device=device,model_options=options,
                  schedules=seed,policy=policy,
                  search_settings=SearchSettings(budget=1,width=1,warmup=0,samples=1,repeats=1))
  meta['compilation_resources']=dict(seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  print('COMPILED_FP8',json.dumps(meta['compilation_resources']),flush=True)
  with Executor.from_program(loaded,reference_program,device,graphs=True) as owner:
   # Correct only the integrated tile-sum regression in the F16 reference;
   # immutable weights, native KV and the rest of the compiled program stay fixed.
   with ExitStack() as stack:
    modules={}
    for depth in (2048,1536):
     path=Path(f'build/lfm-moe/packed-view/{depth}-128.tbin')
     modules[depth]=device.load(path);stack.callback(modules[depth].release)
    forward=owner.batches.select(1,1).forward;plan=ExecutionPlan(device,labeled=True)
    for entry,call in zip(forward.plan,forward.definition.calls):
     if call.kind=='quantized_experts':
      module=modules[call.parameters['k']];bindings=call_bindings(entry[0],entry[1]);args=tuple(bindings[a['name']] for a in module.manifest['arguments'])
      plan.append((module,bind(device,module,args),entry[2]))
     else:plan.append(entry)
    graph=plan.capture(resources=tuple(forward.buffers.values())+tuple(forward.kernels.values())+tuple(modules.values()))
    original=forward.graph;forward.graph=graph
    def restore():forward.graph=original;graph.close()
    stack.callback(restore)
    print('F16_PRECISION_REFERENCE',flush=True)
    reference,checks=collect(owner,loaded.tokenizer,chunk=4096)
    np.savez_compressed(root/'precision-f16.npz',**reference)
    meta['f16_precision_reference']=checks;write(root/'report.json',report)
   # Keep the uploaded immutable weight owner; release its F16 state/workspace.
   owner.batches.close()
   with borrow(owner,root/'program') as executor:
    print('FP8_PRECISION_PROBE',flush=True)
    values,checks=collect(executor,loaded.tokenizer,chunk=4096,reference=reference)
    np.savez_compressed(root/'precision-fp8.npz',**values)
    meta['precision_comparison']=checks
    engine=Engine(executor,config=EngineConfig(prefill_chunk=4096,prefill_requests=1))
    diagnostics={}
    for prefix in (32,8192,130944):
     print('MEASURE_FP8',prefix,flush=True);before=telemetry()
     values=diagnostic(executor,dict(chunk=4096),1,prefix,timeout=120);diagnostics[f'b1-p{prefix}']=values
     if not np.isfinite(values).all():raise ValueError(f'nonfinite diagnostics at {prefix}')
     with Session(executor,batch=1,chunk=4096,timeout=120) as session:
      row=run(session,Case(1,prefix,128),warmup=1,repeats=3)
     report['results'].append(row);write(root/'report.json',report)
     print('NATIVE_FP8',prefix,'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],flush=True)
     measured=measured_requests(engine,1,prefix,128,timeout=120)
     row.update(output_ids=measured.pop('output_ids'),serving=measured,
                correctness=dict(correct=True,scope='Independent operation references and FP8 NumPy model oracle; FP8 output baseline for HTTP replay; F16 precision drift reported separately'),
                telemetry=dict(before=before,after=telemetry()))
     write(root/'report.json',report)
    np.savez_compressed(root/'diagnostics-fp8.npz',**diagnostics)
    print('HTTP_FP8',flush=True)
    os.environ['AKAS_SHAPE_TIMEOUT']='120';os.environ['AKAS_BENCHMARK_WARMUP']='1';os.environ['AKAS_BENCHMARK_REPEATS']='3'
    meta['http']=asyncio.run(http_benchmark(lambda:engine,report['results'],vocab=executor.config.vocab))
    print('HTTP_PASSED',json.dumps(meta['http']),flush=True)
  passed=True
except BaseException as error:
 meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 if (root/'program/program.json').exists():rec.program(root/'program')
 rec.artifact(__file__,'preflight/runner.py');rec.artifact('build/lfm-moe/kv_quality.py','precision/probe.py');rec.artifact(root/'seed-profile.json','preflight/seed-profile.json')
 for depth in (2048,1536):rec.artifact(f'build/lfm-moe/packed-view/{depth}-128.tbin',f'precision/f16-continuous-{depth}.tbin')
 for name in ('precision-f16.npz','precision-fp8.npz','diagnostics-fp8.npz'):
  if (root/name).exists():rec.artifact(root/name,'precision/'+name)
 write(root/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
