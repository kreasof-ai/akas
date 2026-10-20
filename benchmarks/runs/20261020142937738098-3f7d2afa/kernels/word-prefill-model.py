"""Full-model A/B check of independently qualified packed-word expert tiles."""
import json, traceback
from contextlib import ExitStack
from pathlib import Path
import tensor
from akas.models.lfm2.model import LFM2MoE
from akas.benchmarking.history import Recorder, write
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.benchmarking.cases import Case
from akas.execution.plan import ExecutionPlan, call_bindings
from akas.tensor.executor import bind
from akas.engine import Engine
from akas.config import EngineConfig
from akas.optimization.evaluation import measured_requests
parent='20261020141834495811-d8e958f0'
reference=json.loads((Path('benchmarks/runs')/parent/'report.json').read_text())
meta=dict(reference['metadata']);meta.pop('http',None)
meta.update(qualification='Qualified bitwise Q4 coefficients and independent small expert oracle; full-model exact 128 FP8 autoregressive IDs versus same compiled program; prototype expert artifacts replace prefill only',prototype_overlay=dict(operation='quantized_experts',geometry=dict(block_m=64,columns=128,block_k=128,threads=256)),warmup=1,repeats=3)
rec=Recorder('benchmarks',meta,kind='algorithm',change='Compare packed-word half2 expert prefill in the full FP8 model',parent=parent)
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
root=Path('build/lfm-moe/q4-word-prefill');program=Path('build/lfm-moe/fp8-model/cuda/program')
try:
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(meta['checkpoint'],program,device,graphs=True) as executor, ExitStack() as stack:
   print('BASELINE_8192',flush=True)
   with Session(executor,batch=1,chunk=4096,timeout=120) as session:
    before=run(session,Case(1,8192,128),warmup=1,repeats=3)
   meta['same_process_baseline']=before
   print('BASELINE_RESULT',before['summary']['prefill_tokens_per_second']['median'],before['summary']['decode_tokens_per_second']['median'],flush=True)
   modules={}
   for depth in (2048,1536):
    path=root/f'{depth}-64-128-128-256.tbin';module=device.load(path);modules[depth]=module;stack.callback(module.release)
   for phase in ('prefill','prefill_state'):
    forward=executor.batches.select(1,4096,phase=phase).forward;plan=ExecutionPlan(device,labeled=True)
    for entry,call in zip(forward.plan,forward.definition.calls):
     if call.kind=='quantized_experts':
      module=modules[call.parameters['k']];bindings=call_bindings(entry[0],entry[1]);args=tuple(bindings[a['name']] for a in module.manifest['arguments'])
      plan.append((module,bind(device,module,args),entry[2]))
     else:plan.append(entry)
    graph=plan.capture(resources=tuple(forward.buffers.values())+tuple(forward.kernels.values())+tuple(modules.values()))
    original=forward.graph;forward.graph=graph
    def restore(forward=forward,original=original,graph=graph):forward.graph=original;graph.close()
    stack.callback(restore)
   engine=Engine(executor,config=EngineConfig(prefill_chunk=4096,prefill_requests=1))
   for prefix in (32,8192,130944):
    print('MEASURE_CANDIDATE',prefix,flush=True)
    with Session(executor,batch=1,chunk=4096,timeout=120) as session:
     row=run(session,Case(1,prefix,128),warmup=1,repeats=3)
    report['results'].append(row);write(root/'model-report.json',report)
    measured=measured_requests(engine,1,prefix,128,timeout=120)
    expected=next(r for r in reference['results'] if r['case']['prompt']==prefix)['output_ids']
    same=measured['output_ids']==expected
    row.update(output_ids=measured.pop('output_ids'),engine=measured,correctness=dict(correct=same,scope=meta['qualification']))
    print('CANDIDATE_RESULT',prefix,row['summary']['prefill_tokens_per_second']['median'],row['summary']['decode_tokens_per_second']['median'],'greedy',same,flush=True)
    if not same:raise ValueError(f'FP8 output differs at {prefix}')
   passed=True
except BaseException as error:
 meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 rec.program(program)
 for path in (__file__,'build/lfm-moe/q4_word_tile.py','build/lfm-moe/q4_word_prefill.py'):
  rec.artifact(path,'kernels/'+Path(path).name)
 for depth in (2048,1536):rec.artifact(root/f'{depth}-64-128-128-256.tbin',f'overlay/{depth}.tbin')
 write(root/'model-report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
