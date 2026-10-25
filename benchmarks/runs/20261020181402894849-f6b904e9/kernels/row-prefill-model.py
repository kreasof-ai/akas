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
from akas.optimization.evaluation import prompts
program=Path('build/lfm-moe/staged-model/cuda/program');root=Path('build/lfm-moe/row-prefill-model');root.mkdir(exist_ok=True);attention_root=Path('build/lfm-moe/attention-rows');expert_root=Path('build/lfm-moe/packed-expert-activation');parent='20261020152431734974-3777f95e';reference=json.loads((Path('benchmarks/runs')/parent/'report.json').read_text())
meta=dict(runtime='akas',purpose='preflight',protocol='native-token-step-v2',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),model='akas.models.lfm2.model:LFM2MoE',checkpoint='build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf',checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',weight_format='q4_0',kv_dtype='fp8',mode='ar',output_tokens=128,shape_timeout_seconds=120,chunk=4096,graphs=True,warmup=1,repeats=3,qualification='Native full-model Q4 with compiler-owned FP8 staging, row-partitioned FP16 attention and packed expert FFN: GPU route packing, paired gate/up/SwiGLU F16 epilogue, compact input down projection. Existing F16 operand rounding retained; conversion and packing included. All 128 greedy tokens must match qualified FP8 reference. Research overlay; integrated compiler and HTTP not measured.',kernel_evidence_run_id='20261020175337942340-7a71d91a',reference_run_id=parent,expert_evidence_run_id='20261020163234092922-68babf9f')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Combine row-partitioned attention with packed routed expert prefill on matched baseline prompts',parent=parent);report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
try:
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(meta['checkpoint'],program,device,graphs=True) as model,ExitStack() as stack:
   modules={'attention':device.load(attention_root/'full-64-64-128.tbin')}
   for name,path in dict(offsets=expert_root/'offsets.tbin',gather=expert_root/'gather.tbin',fused=expert_root/'128-64-64-256-1.tbin',down=Path('build/lfm-moe/packed-q4-expert/1536-128-128-128-256-1.tbin')).items():modules[name]=device.load(path)
   for m in modules.values():stack.callback(m.release)
   packed_input=device.empty((4096*4,2048),'float16');packed_activation=device.empty((4096*4,1536),'float16');offsets=device.empty((65,),'int32')
   for b in (packed_input,packed_activation,offsets):stack.callback(b.release)
   meta['shared_temporary_bytes']=sum(b.nbytes for b in (packed_input,packed_activation,offsets))
   # One shadow pair is reused in ordered execution, across layers and phases.
   for phase in ('prefill','prefill_state'):
    forward=model.batches.select(1,4096,phase=phase).forward;original=forward.graph;plan=ExecutionPlan(device,labeled=True);replacements=0
    entries=list(zip(forward.plan,forward.definition.calls));index=0;expert_replacements=0
    def append_module(name,bindings,label):
     module=modules[name];plan.bind(module,tuple(bindings[a['name']] for a in module.manifest['arguments']),label=label)
    while index<len(entries):
     entry,call=entries[index]
     if call.kind=='quantized_experts' and not call.parameters.get('routed_input'):
      gate=call_bindings(entry[0],entry[1]);up_entry,up_call=entries[index+1];act_entry,act_call=entries[index+2];down_entry,down_call=entries[index+3]
      assert up_call.kind=='quantized_experts' and act_call.kind=='swiglu_experts' and down_call.kind=='quantized_experts' and down_call.parameters.get('routed_input')
      up=call_bindings(up_entry[0],up_entry[1]);down=call_bindings(down_entry[0],down_entry[1]);shared=dict(gate,offsets=offsets,packed=packed_input)
      for name in ('offsets','gather'):append_module(name,shared,f'{entry[2]}-{name}')
      append_module('fused',dict(gate,x=packed_input,w2=up['w'],offsets=offsets,out=packed_activation),entry[2])
      append_module('down',dict(down,x=packed_activation,offsets=offsets),down_entry[2]);expert_replacements+=1;index+=4;continue
     if call.kind=='attention':
      assert call.parameters.get('staged_kv') and call.parameters['kv_dtype']=='float16'
      bindings=call_bindings(entry[0],entry[1]);append_module('attention',bindings,entry[2]);replacements+=1
     else:plan.append(entry)
     index+=1
    candidate=plan.capture(resources=tuple(forward.buffers.values())+tuple(forward.kernels.values())+tuple(modules.values())+(packed_input,packed_activation,offsets));stack.callback(candidate.close);stack.callback(lambda f=forward,g=original:setattr(f,'graph',g));forward.graph=candidate
    meta.setdefault('overlays',[]).append(dict(phase=phase,attention_calls=replacements,expert_blocks=expert_replacements));print('OVERLAY_READY',phase,replacements,flush=True)
   for prefix in (32,8192,130944):
    print('MEASURE',prefix,flush=True)
    with Session(model,batch=1,chunk=4096,timeout=120) as session:
     row=run(session,Case(1,prefix,128),warmup=1,repeats=3);row['base_allocated_bytes']=row['allocated_bytes'];row['allocated_bytes']+=meta['shared_temporary_bytes'];session.reset();ids=list(session.prefill(prompts(model.config.vocab,1,prefix)))
     for _ in range(127):ids.extend(session.decode(1))
    expected=next(r for r in reference['results'] if r['case']['prompt']==prefix)['output_ids'];same=[ids]==expected;row.update(output_ids=[ids],correctness=dict(correct=same,scope=meta['qualification']));report['results'].append(row);write(root/'model-report.json',report)
    print('RESULT',prefix,'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],'same',same,flush=True)
    if not same:raise ValueError(f'Full-model FP8 shadow changes greedy tokens at {prefix}')
 passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 rec.program(program)
 for path in (__file__,'build/lfm-moe/attention_warp_rows.py','build/lfm-moe/compile-attention-rows.py','build/lfm-moe/packed_expert_activation.py','build/lfm-moe/compile-packed-expert-activation.py','build/lfm-moe/pack_expert_input.py','build/lfm-moe/compile-pack-expert-input.py','build/lfm-moe/packed_q4_expert.py','build/lfm-moe/q4_word_tile.py'):rec.artifact(path,'kernels/'+Path(path).name)
 for name in ('full-64-64-128',):rec.artifact(attention_root/f'{name}.tbin',f'overlay/{name}.tbin')
 for name,path in dict(offsets=expert_root/'offsets.tbin',gather=expert_root/'gather.tbin',fused=expert_root/'128-64-64-256-1.tbin',down=Path('build/lfm-moe/packed-q4-expert/1536-128-128-128-256-1.tbin')).items():rec.artifact(path,f'overlay/{name}.tbin')
 write(root/'model-report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
