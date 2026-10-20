"""Full-model check of continuous expert accumulation and grouped-query prefill."""
import json,traceback,sys
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import tensor
from akas.models.lfm2.model import LFM2MoE
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.session import Session
from akas.benchmarking.runner import run
from akas.benchmarking.cases import Case
from akas.optimization.evaluation import diagnostic,measured_requests
from akas.execution.plan import ExecutionPlan,call_bindings
from akas.tensor.executor import bind
from akas.engine import Engine
from akas.config import EngineConfig
from akas.artifacts.read import artifact_path
root=Path('build/lfm-moe/continuous-and-prefill');root.mkdir(exist_ok=True)
old=Path('build/single-user-long-context/lfm2.5-24b-a2b-q4_0-cuda-default-ar/jobs/ee9b3acb399892d0/attempt-0001')
oldreport=json.loads((old/'report.json').read_text());references=np.load(old/'diagnostics.npz',allow_pickle=False)
meta=dict(runtime='akas',purpose='preflight',protocol='native-token-step-v2',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),model='akas.models.lfm2.model:LFM2MoE',checkpoint='build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf',checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',kv_dtype='float16',weight_format='q4_0',mode='ar',chunk=4096,graphs=True,warmup=1,repeats=3,shape_timeout_seconds=120,qualification='B1 prototype: independently qualified packed continuous expert accumulation and grouped-query prefill; full-model logits and exact 128 retained greedy IDs required at every shape')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Restore continuous expert accumulation and test shared GQA prefill without changing cache precision',parent='20261020123109563937-619ed8db')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False;changes=[];modules={};experiments=[];plans={}
def replace(forward,kind,choose):
 plan=ExecutionPlan(device,labeled=True)
 for entry,call in zip(plans.get(forward,forward.plan),forward.definition.calls):
  module=choose(call) if call.kind==kind else None
  if module is not None:
   bindings=call_bindings(entry[0],entry[1]);args=tuple(bindings[a['name']] for a in module.manifest['arguments'])
   plan.append((module,bind(device,module,args),entry[2]))
  else:plan.append(entry)
 graph=plan.capture(resources=tuple(forward.buffers.values())+tuple(forward.kernels.values())+tuple(modules.values()))
 changes.append((forward,forward.graph,graph));previous=forward.graph;forward.graph=graph;plans[forward]=plan
 def restore():forward.graph=previous;graph.close()
 stack.callback(restore)
try:
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(meta['checkpoint'],'build/lfm-moe/manual-v3/cuda/program',device,graphs=True) as executor,ExitStack() as stack:
   engine=Engine(executor,config=EngineConfig(prefill_chunk=4096,prefill_requests=1))
   for depth in (2048,1536):
    modules[depth]=device.load(f'build/lfm-moe/packed-view/{depth}-128.tbin');stack.callback(modules[depth].release)
   replace(executor.batches.select(1,1).forward,'quantized_experts',lambda call:modules[call.parameters['k']])
   measured=measured_requests(engine,1,130944,128,timeout=120)
   expected=next(r for r in oldreport['results'] if r['case']['prompt']==130944)['output_ids'];same=measured['output_ids']==expected
   experiment=dict(implementation='continuous-packed-decode',greedy_identical=same,engine=measured);experiments.append(experiment);print('CONTINUOUS_CHECK',json.dumps(experiment),flush=True)
   if not same:
    directory=Path('build/lfm-moe/manual-v2/cuda/program');program=json.loads((directory/'program.json').read_text());phase=next(p for p in program['phases'] if p['phase']=='decode');forward=json.loads((directory/phase['path']).read_text())
    record=next(k for k in forward['kernels'].values() if k['kind']=='linear' and 'type' not in k['parameters'])
    modules['router']=device.load(artifact_path(directory,record));stack.callback(modules['router'].release)
    replace(executor.batches.select(1,1).forward,'linear',lambda call:modules['router'] if 'type' not in call.parameters else None)
    measured=measured_requests(engine,1,130944,128,timeout=120);same=measured['output_ids']==expected
    experiment=dict(implementation='continuous-packed-old-router',greedy_identical=same,engine=measured);experiments.append(experiment);print('CONTINUOUS_ROUTER_CHECK',json.dumps(experiment),flush=True)
   if not same:raise ValueError('continuous expert accumulation plus reference router still differs at full context')
   # Small independent and full GPU-reference checks must pass before these load.
   validation=Path('build/lfm-moe/validate-grouped-prefill-v2.log').read_text()
   if not validation.rstrip().endswith('True'):raise ValueError('grouped prefill qualification did not pass')
   qm,bk=64,64
   modules['prefill']=device.load(f'build/lfm-moe/grouped-prefill/float16-{qm}-{bk}.tbin');stack.callback(modules['prefill'].release)
   for phase in ('prefill','prefill_state'):replace(executor.batches.select(1,4096,phase=phase).forward,'attention',lambda call:modules['prefill'])
   meta.update(query_rows=qm,key_rows=bk,pipeline_group=8)
   for prefix in (32,8192,130944):
    print('MEASURE',prefix,flush=True)
    values=diagnostic(executor,dict(chunk=4096),1,prefix,timeout=120);reference=references[f'b1-p{prefix}'];relative=float(np.linalg.norm(values.astype('float64')-reference)/max(np.linalg.norm(reference),1e-30))
    if relative>.01:raise ValueError(f'logit drift {relative} at {prefix}')
    with Session(executor,batch=1,chunk=4096,timeout=120) as session:row=run(session,Case(1,prefix,128),warmup=1,repeats=3)
    measured=measured_requests(engine,1,prefix,128,timeout=120);expected=next(r for r in oldreport['results'] if r['case']['prompt']==prefix)['output_ids'];same=measured['output_ids']==expected
    row.update(output_ids=measured.pop('output_ids'),engine=measured,correctness=dict(correct=same,relative_logit_error=relative,reference=meta['qualification']))
    if not same:row.update(status='error',error='greedy output differs from retained reference')
    report['results'].append(row);write(root/'report.json',report)
    print('RESULT',prefix,'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],'greedy',same,flush=True)
    if not same:raise ValueError(f'grouped prefill greedy mismatch at {prefix}')
   passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 references.close();meta['experiments']=experiments
 rec.artifact(__file__,'preflight/runner.py');rec.artifact('build/lfm-moe/packed_view.py','kernels/packed_view.py');rec.artifact('build/lfm-moe/grouped_prefill.py','kernels/grouped_prefill.py')
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
