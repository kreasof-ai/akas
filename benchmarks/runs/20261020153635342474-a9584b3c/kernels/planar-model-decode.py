"""Full-model streamed-weight A/B/A for a lossless expert-weight byte layout."""
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
program=Path('build/lfm-moe/codec-model/cuda/program');root=Path('build/lfm-moe/planar-q4')
meta=dict(runtime='akas',purpose='kernel',protocol='snapshot-decode-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),model='akas.models.lfm2.model:LFM2MoE',checkpoint='build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf',weight_format='q4_0',kv_dtype='fp8',mode='ar',output_tokens=128,shape_timeout_seconds=120,qualification='Whole-model streamed-weight A/B/A over identical GPU state snapshots; byte-exact GPU layout conversion and independent CPU permutation check; all 128 greedy tokens must match before timing. Decode-only prototype; fresh prefill and HTTP with planar weights not measured.',kernel_evidence_run_id='20261020153056387214-beeea5c5',reference_http_run_id='20261020152431734974-3777f95e')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Compare aligned expert Q4 byte layout under actual full-model weight streaming')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
try:
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(meta['checkpoint'],program,device,graphs=True) as executor,ExitStack() as stack:
   forward=executor.batches.select(1,1).forward;original=forward.graph;modules={}
   for depth,columns in ((2048,1536),(1536,2048)):
    module=device.load(root/f'1-{depth}-{columns}.tbin');modules[depth]=module;stack.callback(module.release)
   pack=device.load(root/'pack.tbin');unpack=device.load(root/'unpack.tbin');stack.callback(pack.release);stack.callback(unpack.release)
   weight_names=sorted({call.arguments[1] for call in forward.definition.calls if call.kind=='quantized_experts'})
   weights=[executor.weights[name] for name in weight_names];count=weights[0].nbytes
   assert len(weights)==114 and all(w.nbytes==count and str(w.dtype)=='uint8' for w in weights)
   temporary=device.empty((count,),'uint8');stack.callback(temporary.release)
   original_bytes=weights[0].to_numpy();blocks=original_bytes.reshape(-1,18);expected=np.concatenate((blocks[:,2:].ravel(),blocks[:,:2].ravel()))
   pack.launch(weights[0],temporary);np.testing.assert_array_equal(temporary.to_numpy(),expected)
   with ExitStack() as qualification:
    restored=device.empty((count,),'uint8');qualification.callback(restored.release);unpack.launch(temporary,restored);np.testing.assert_array_equal(restored.to_numpy(),original_bytes)
   del original_bytes,blocks,expected
   print('DEVICE_PERMUTATION_BITWISE_OK',len(weights),count,flush=True)
   meta['weight_transform']=dict(tensors=len(weights),total_payload_bytes=sum(w.nbytes for w in weights),temporary_bytes=count,resident_payload_change_bytes=0)
   plan=ExecutionPlan(device,labeled=True)
   for entry,call in zip(forward.plan,forward.definition.calls):
    if call.kind=='quantized_experts':
     module=modules[call.parameters['k']];bindings=call_bindings(entry[0],entry[1]);args=tuple(bindings[a['name']] for a in module.manifest['arguments']);plan.append((module,bind(device,module,args),entry[2]))
    else:plan.append(entry)
   candidate=plan.capture(resources=tuple(forward.buffers.values())+tuple(forward.kernels.values())+tuple(modules.values()));stack.callback(candidate.close);stack.callback(lambda:setattr(forward,'graph',original))
   def transform(kernel):
    started=time.perf_counter()
    for weight in weights:kernel.launch(weight,temporary);weight.copy_from(temporary)
    device.synchronize();return time.perf_counter()-started
   for prefix in (8192,130944):
    print('PREFILL_RAW_SNAPSHOT',prefix,flush=True);rng=np.random.default_rng(17);prompt=rng.integers(0,128000,prefix,dtype='int32').tolist()
    with Session(executor,batch=1,chunk=4096,timeout=120) as session,ExitStack() as snapshots:
     forward.graph=original;session.reset();first=list(session.prefill([prompt]));positions=executor.positions.copy();states={}
     for name,buffer in executor.batches.states.items():
      clone=device.empty(buffer.shape,str(buffer.dtype));clone.copy_from(buffer);states[name]=clone;snapshots.callback(clone.release)
     device.synchronize()
     def restore():
      for name,clone in states.items():executor.batches.states[name].copy_from(clone)
      executor.positions[:]=positions;session.tokens=list(first);session.feedback_ready=False;session.deadline=time.perf_counter()+120;device.synchronize()
     expected=None
     for label in ('raw-before','planar','raw-after'):
      if label=='planar':meta.setdefault('transforms',[]).append(dict(prefix=prefix,direction='pack',seconds=transform(pack)));forward.graph=candidate
      elif label=='raw-after':meta.setdefault('transforms',[]).append(dict(prefix=prefix,direction='unpack',seconds=transform(unpack)));forward.graph=original
      restore();output=[first[0]]
      for _ in range(127):output.extend(session.decode(1))
      if expected is None:expected=output
      if output!=expected:raise ValueError(f'{label} changes greedy sequence at {prefix}')
      samples=[]
      for iteration in range(4):
       restore();begin=time.perf_counter();session.decode(127,read=False);device.synchronize();elapsed=time.perf_counter()-begin
       if iteration:samples.append(127/elapsed)
      result=dict(prefix=prefix,layout=label,decode_tokens_per_second=statistics.median(samples),samples=samples,greedy_identical=True);meta.setdefault('decode_comparisons',[]).append(result);write(root/'model-report.json',report);print('DECODE_RESULT',json.dumps(result),flush=True)
   passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 rec.program(program)
 for path in (__file__,'build/lfm-moe/planar_transfer.py','build/lfm-moe/compile-planar-transfer.py','build/lfm-moe/planar_q4.py','build/lfm-moe/planar_decode.py','build/lfm-moe/compile-planar-q4.py'):rec.artifact(path,'kernels/'+Path(path).name)
 for name in ('pack','unpack','1-2048-1536','1-1536-2048'):rec.artifact(root/f'{name}.tbin',f'overlay/{name}.tbin')
 write(root/'model-report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
