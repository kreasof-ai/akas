"""Qualify finite-tail attention and the short expert tile against the F16 model."""
from contextlib import ExitStack
import json
from pathlib import Path
import traceback
import numpy as np
import tensor
from akas.benchmarking.cases import Case
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.compilation.emit import build_artifact
from akas.compilation.selection import selected_source
from akas.execution.plan import ExecutionPlan,call_bindings
from akas.kernels.definition import definition
from akas.models.lfm2.model import LFM2MoE
from akas.optimization.evaluation import prompts

ROOT=Path('build/lfm-moe/f16-prepared-model');ROOT.mkdir(exist_ok=True)
PROGRAM=Path('build/lfm-moe/cuda-f16-full/cuda/program')
PARENT='20261023072001816909-fb5b388f'
reference=json.loads((Path('benchmarks/runs')/PARENT/'report.json').read_text())
meta=dict(reference['metadata'])
for key in ('http','precision_probe','compilation_resources','performance_target'):meta.pop(key,None)
meta.update(purpose='diagnostic',protocol='native-token-step-v2',requested_prefixes=[32,8192],
 qualification='Fixed F16 KV model, A/B/A native timing. Only unused causal cache tails are zeroed before the qualified ordered attention kernel; 32-token down-expert block_k grows from 64 to 128. All 128 output IDs must match the current F16 native/Engine/HTTP reference. Kernel-only evidence and whole-model native isolation; not a new HTTP deployment qualification.',
 kernel_evidence_runs=['20261023082556125504-a256cd82','20261023083357910933-2ce58fe2'])
rec=Recorder('benchmarks',meta,kind='algorithm',parent=PARENT,
 change='Qualify prepared native F16 attention and bitwise short expert tiles in the real model')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
try:
 spec=dict(kind='attention_cache_extent',parameters=dict(slots=1,chunk=4096,capacity=131072),contract='reference',phase='prefill')
 caps=dict(provider='cuda',target='sm_89');implementation,schedule=definition(spec,{},caps)
 source=ROOT/'extent.py';source.write_text(selected_source(spec,schedule,caps,implementation).render())
 build_artifact(source,source.with_suffix('.tbin'),provider='cuda',target='sm_89')
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(meta['checkpoint'],PROGRAM,device,graphs=True) as model,ExitStack() as owned:
   paths=dict(attention=Path('build/lfm-moe/attention-f16-direct/full.tbin'),pad=Path('build/lfm-moe/attention-f16-direct/full-pad.tbin'),extent=source.with_suffix('.tbin'),down=Path('build/lfm-moe/short-expert-tiles/down-r32-16-32-128-128.tbin'))
   modules={name:device.load(path) for name,path in paths.items()}
   for module in modules.values():owned.callback(module.release)
   originals={};prepared={}
   for chunk in (32,4096):
    for phase in ('prefill','prefill_state'):
     forward=model.batches.select(1,chunk,phase=phase).forward
     key=(chunk,phase);originals[key]=forward.graph
     lengths=device.empty((1,),'int32');owned.callback(lengths.release)
     plan=ExecutionPlan(device,labeled=True);count=0
     for entry,call in zip(forward.plan,forward.definition.calls):
      if chunk==4096 and call.kind=='attention':
       b=call_bindings(entry[0],entry[1]);b['lengths']=lengths
       for name in ('extent','pad','attention'):
        module=modules[name];plan.bind(module,tuple(b[a['name']] for a in module.manifest['arguments']))
       count+=1
      elif chunk==32 and call.kind=='quantized_experts' and call.parameters.get('r')==32 and call.parameters.get('k')==1536:
       b=call_bindings(entry[0],entry[1]);module=modules['down'];plan.bind(module,tuple(b[a['name']] for a in module.manifest['arguments']),label=entry[2]);count+=1
      else:plan.append(entry)
     graph=plan.capture(resources=tuple(forward.buffers.values())+tuple(forward.kernels.values())+tuple(modules.values())+(lengths,))
     owned.callback(graph.close);prepared[key]=graph
     print('OVERLAY',key,count,flush=True)
   def select(use):
    for key in originals:model.batches.select(1,key[0],phase=key[1]).forward.graph=(prepared if use else originals)[key]
   owned.callback(lambda:select(False))
   for variant in ('original-before','prepared','original-after'):
    select(variant=='prepared')
    for prefix in (32,8192):
     with Session(model,batch=1,chunk=4096,timeout=120) as session:
      row=run(session,Case(1,prefix,128),warmup=1,repeats=3)
      session.reset();ids=list(session.prefill(prompts(model.config.vocab,1,prefix)))
      for _ in range(127):ids.extend(session.decode(1))
     expected=next(r['output_ids'][0] for r in reference['results'] if r['case']['prompt']==prefix)
     if ids!=expected:raise ValueError(f'Changed F16 model output: {variant}/{prefix}')
     row.update(variant=variant,output_ids=[ids],correctness=dict(correct=True,reference_run_id=PARENT))
     report['results'].append(row);write(ROOT/'report.json',report)
     print('PASS',variant,prefix,'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],flush=True)
 passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 rec.program(PROGRAM);rec.artifact(__file__,'benchmark/runner.py')
 for path in ROOT.glob('*.tbin'):rec.artifact(path,'kernels/'+path.name)
 write(ROOT/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed')
 print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
