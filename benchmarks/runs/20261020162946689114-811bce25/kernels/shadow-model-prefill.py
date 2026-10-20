"""Test transient decoded FP8 attention in the actual ordered model program."""
import json,time,traceback
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import tensor
from akas.models.lfm2.model import LFM2MoE
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.session import Session
from akas.benchmarking.runner import run
from akas.benchmarking.cases import Case
from akas.execution.plan import ExecutionPlan,call_bindings
program=Path('build/lfm-moe/codec-model/cuda/program');root=Path('build/lfm-moe/fp8-shadow');parent='20261020152431734974-3777f95e';reference=json.loads((Path('benchmarks/runs')/parent/'report.json').read_text())
meta=dict(runtime='akas',purpose='preflight',protocol='native-token-step-v2',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),model='akas.models.lfm2.model:LFM2MoE',checkpoint='build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf',weight_format='q4_0',kv_dtype='fp8',mode='ar',output_tokens=128,shape_timeout_seconds=120,chunk=4096,graphs=True,warmup=1,repeats=2,qualification='Native full-model prefill/decode with shared transient FP16 copies of exactly the existing decoded FP8 KV. Complete conversion costs included; persistent KV and decode unchanged. All 128 greedy tokens must match qualified FP8 reference. Research overlay; fresh integrated compile and HTTP not measured.',kernel_evidence_run_id='20261020162743785090-f439dad8',reference_run_id=parent)
rec=Recorder('benchmarks',meta,kind='algorithm',change='Reuse decoded FP8 values across all query tiles in whole-model prefill',parent=parent);report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
try:
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(meta['checkpoint'],program,device,graphs=True) as model,ExitStack() as stack:
   modules={name:device.load(root/path) for name,path in dict(extent='full-extent.tbin',shadow='full-shadow.tbin',attention='full-floating-32-128-1.tbin').items()}
   for m in modules.values():stack.callback(m.release)
   keys=device.empty((1,8,131072,64),'float16');values=device.empty(keys.shape,'float16');unit=device.from_numpy(np.ones((1,8,131072,1),dtype='float32'));lengths=device.empty((1,),'int32')
   for b in (keys,values,unit,lengths):stack.callback(b.release)
   meta['shared_temporary_bytes']=sum(b.nbytes for b in (keys,values,unit,lengths))
   # One shadow pair is reused in ordered execution, across layers and phases.
   for phase in ('prefill','prefill_state'):
    forward=model.batches.select(1,4096,phase=phase).forward;original=forward.graph;plan=ExecutionPlan(device,labeled=True);replacements=0
    for entry,call in zip(forward.plan,forward.definition.calls):
     if call.kind!='attention':plan.append(entry);continue
     bindings=call_bindings(entry[0],entry[1]);shared=dict(bindings,lengths=lengths,keys=keys,values=values)
     for name in ('extent','shadow'):
      module=modules[name];plan.bind(module,tuple(shared[a['name']] for a in module.manifest['arguments']),label=f'{entry[2]}-{name}')
     native=dict(bindings,kc=keys,vc=values,ks=unit,vs=unit);module=modules['attention'];plan.bind(module,tuple(native[a['name']] for a in module.manifest['arguments']),label=entry[2]);replacements+=1
    candidate=plan.capture(resources=tuple(forward.buffers.values())+tuple(forward.kernels.values())+tuple(modules.values())+(keys,values,unit,lengths));stack.callback(candidate.close);stack.callback(lambda f=forward,g=original:setattr(f,'graph',g));forward.graph=candidate
    meta.setdefault('overlays',[]).append(dict(phase=phase,attention_calls=replacements));print('OVERLAY_READY',phase,replacements,flush=True)
   for prefix in (8192,130944):
    print('MEASURE',prefix,flush=True)
    with Session(model,batch=1,chunk=4096,timeout=120) as session:
     row=run(session,Case(1,prefix,128),warmup=1,repeats=2);session.reset();prompt=np.random.default_rng(17).integers(0,128000,prefix,dtype='int32').tolist();ids=list(session.prefill([prompt]))
     for _ in range(127):ids.extend(session.decode(1))
    expected=next(r for r in reference['results'] if r['case']['prompt']==prefix)['output_ids'];same=ids==expected;row.update(output_ids=ids,correctness=dict(correct=same,scope=meta['qualification']));report['results'].append(row);write(root/'model-report.json',report)
    print('RESULT',prefix,'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],'same',same,flush=True)
    if not same:raise ValueError(f'Full-model FP8 shadow changes greedy tokens at {prefix}')
 passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 rec.program(program)
 for path in (__file__,'build/lfm-moe/fp8_shadow.py','build/lfm-moe/shadow_floating.py','build/lfm-moe/compile-fp8-shadow.py','build/lfm-moe/compile-fp8-shadow-floating.py'):rec.artifact(path,'kernels/'+Path(path).name)
 for name in ('full-extent','full-shadow','full-floating-32-128-1'):rec.artifact(root/f'{name}.tbin',f'overlay/{name}.tbin')
 write(root/'model-report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
