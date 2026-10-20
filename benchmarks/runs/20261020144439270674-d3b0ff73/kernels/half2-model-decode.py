"""Decode A/B/A over identical GPU state snapshots; excludes prefill/restore."""
import json,time,traceback,statistics
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import tensor
from akas.models.lfm2.model import LFM2MoE
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.session import Session
from akas.execution.plan import ExecutionPlan,call_bindings
from akas.tensor.executor import bind
parent='20261020141834495811-d8e958f0';program=Path('build/lfm-moe/fp8-model/cuda/program')
reference=json.loads((Path('benchmarks/runs')/parent/'report.json').read_text())
meta=dict(runtime='akas',purpose='kernel',protocol='snapshot-decode-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),model='akas.models.lfm2.model:LFM2MoE',checkpoint=reference['metadata']['checkpoint'],weight_format='q4_0',kv_dtype='fp8',mode='ar',output_tokens=128,shape_timeout_seconds=120,qualification='Whole-model decode-only A/B/A over identical GPU state snapshots; restore excluded; every schedule must match all 128 baseline greedy tokens. Not an end-to-end native or HTTP timing.')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Compare bitwise coefficient half2 expert decoding on the full model',parent=parent)
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
try:
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(meta['checkpoint'],program,device,graphs=True) as executor,ExitStack() as stack:
   frame=executor.batches.select(1,1);forward=frame.forward;original=forward.graph;graphs={'baseline':original};modules={}
   for threads in (128,256):
    for depth,columns in ((2048,1536),(1536,2048)):
     module=device.load(f'build/lfm-moe/half2-decode/{depth}-{columns}-{threads}.tbin');modules[depth,threads]=module;stack.callback(module.release)
    plan=ExecutionPlan(device,labeled=True)
    for entry,call in zip(forward.plan,forward.definition.calls):
     if call.kind=='quantized_experts':
      module=modules[call.parameters['k'],threads];bindings=call_bindings(entry[0],entry[1]);args=tuple(bindings[a['name']] for a in module.manifest['arguments']);plan.append((module,bind(device,module,args),entry[2]))
     else:plan.append(entry)
    graph=plan.capture(resources=tuple(forward.buffers.values())+tuple(forward.kernels.values())+tuple(modules.values()));graphs[str(threads)]=graph;stack.callback(graph.close)
   stack.callback(lambda:setattr(forward,'graph',original))
   for prefix in (8192,130944):
    print('PREFILL_SNAPSHOT',prefix,flush=True)
    rng=np.random.default_rng(17);prompt=rng.integers(0,128000,prefix,dtype='int32').tolist()
    with Session(executor,batch=1,chunk=4096,timeout=120) as session,ExitStack() as snapshots:
     forward.graph=original;session.reset();first=list(session.prefill([prompt]));positions=executor.positions.copy()
     states={}
     for name,buffer in executor.batches.states.items():
      clone=device.empty(buffer.shape,str(buffer.dtype));clone.copy_from(buffer);states[name]=clone;snapshots.callback(clone.release)
     device.synchronize()
     def restore():
      for name,clone in states.items():executor.batches.states[name].copy_from(clone)
      executor.positions[:]=positions;session.tokens=list(first);session.feedback_ready=False;session.deadline=time.perf_counter()+120
      device.synchronize()
     rows=[];expected=None
     for label in ('baseline','128','256','baseline'):
      forward.graph=graphs[label];restore();output=[first[0]]
      for _ in range(127):output.extend(session.decode(1))
      if expected is None:expected=output
      if output!=expected:raise ValueError(f'{label} changes greedy sequence at {prefix}')
      samples=[]
      for iteration in range(4):
       restore();begin=time.perf_counter();session.decode(127,read=False);device.synchronize();elapsed=time.perf_counter()-begin
       if iteration:samples.append(127/elapsed)
      result=dict(prefix=prefix,schedule=label,decode_tokens_per_second=statistics.median(samples),samples=samples,greedy_identical=True);rows.append(result);print('DECODE_RESULT',json.dumps(result),flush=True)
     meta.setdefault('decode_comparisons',[]).extend(rows);write('build/lfm-moe/half2-decode/model-report.json',report)
   passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 rec.program(program)
 for path in (__file__,'build/lfm-moe/half2_decode.py','build/lfm-moe/q4_word_tile.py'):
  rec.artifact(path,'kernels/'+Path(path).name)
 for threads in (128,256):
  for depth,columns in ((2048,1536),(1536,2048)):rec.artifact(f'build/lfm-moe/half2-decode/{depth}-{columns}-{threads}.tbin',f'overlay/{depth}-{threads}.tbin')
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
