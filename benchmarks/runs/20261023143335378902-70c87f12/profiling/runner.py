"""Keep one paired GPU session while qualifying streamed verifier candidates.

Commands are JSON lines on stdin. Each experiment gets its own sealed record.
Device snapshots preserve inputs before reused workspaces can overwrite them.
"""
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import sys
import traceback
import numpy as np
import tensor
from tensor.providers.cuda.graph import CudaGraph
from tensor.compiler.tuning import measure_cuda
from akas.benchmarking.cases import Case
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.runner import run
from akas.benchmarking.speculative import SpeculativeSession
from akas.execution.plan import ExecutionPlan,call_bindings,bind_named
from akas.generation.speculative.decoding import SpeculativeDecoder
from akas.models.dspark.proposer import load
from akas.models.lfm2.model import LFM2MoE

ROOT=Path('/home/sagemaker-user/akas');OUT=ROOT/'build/lfm-moe/verification-lab';OUT.mkdir(exist_ok=True)
TARGET=ROOT/'build/lfm-moe/dspark-short-optimized/target';DRAFT=ROOT/'build/lfm-moe/dspark-short-q8/draft'
CP=ROOT/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
workload=json.loads((ROOT/'build/lfm-moe/dspark-short-q8/short-workload.json').read_text())

def recorder(purpose,description,parent='20261023135028708663-743b0598'):
 meta=dict(runtime='akas',purpose=purpose,protocol='streamed-verifier-lab-v1',provider='cuda',target='sm_89',
  model='akas.models.lfm2.model:LFM2MoE',device=dict(name='NVIDIA L4'),mode='dspark',weight_format='q4_0',
  draft_weight_dtype='q8_0',kv_dtype='float16',draft_kv_dtype='float16',shape_timeout_seconds=120,
  checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5')
 rec=Recorder(ROOT/'benchmarks',meta,kind='algorithm',parent=parent,change=description)
 return rec,dict(schema='akas.benchmark.v1',metadata=meta,results=[])

with tensor.Device(provider='cuda') as device,ExitStack() as owned:
 target=owned.enter_context(LFM2MoE(CP,TARGET,device,graphs=True))
 head=load(target,checkpoint=CP.with_name('dspark-q8_0.gguf'),bundle=DRAFT,graphs=True,confidence_threshold=.5);owned.callback(head.close)
 decoder=SpeculativeDecoder(target,head,window=10,capture=False)
 snapshots=[];snapshot_run=None;snapshot_owned=ExitStack();owned.callback(snapshot_owned.close)
 print('READY',flush=True)
 for line in sys.stdin:
  try:command=json.loads(line)
  except ValueError:continue
  if command.get('action')=='close':break
  rec,report=recorder('profiling' if command['action']=='snapshot' else 'kernel',str(command),parent=snapshot_run or '20261023135028708663-743b0598')
  meta=report['metadata'];passed=False
  print('RUN',rec.id,command,flush=True)
  try:
   if command['action']=='snapshot':
    snapshot_owned.close();snapshot_owned=ExitStack();owned.callback(snapshot_owned.close);snapshots=[]
    item=next(c for c in workload['cases'] if c['id']==command.get('case','gsm8k-train:0:48'))
    chunk=command.get('chunk',10);original=target.verify;captured=False
    def observe(*args,**kwargs):
     global captured
     result=original(*args,**kwargs)
     if captured or kwargs['chunk']!=chunk:return result
     captured=True;frame=target.batches.select(1,chunk,phase='verify');f=frame.forward;before={};after={}
     for i,(call,entry) in enumerate(zip(f.definition.calls,f.plan)):
      if call.kind!='quantized_experts':continue
      values=call_bindings(entry[0],entry[1]);clones={}
      for name,buffer in values.items():
       if name=='w':clones[name]=buffer;continue
       clone=device.empty(buffer.shape,str(buffer.dtype));snapshot_owned.callback(clone.release);clones[name]=clone
       (after if name=='out' else before).setdefault(i,[]).append((clone,buffer))
      snapshots.append(dict(index=i,parameters=call.parameters,kernel=entry[0],values=clones))
     def submit():
      for i,entry in enumerate(f.plan):
       for dest,source in before.get(i,()):device.driver.call('cuMemcpyDtoDAsync_v2',dest.pointer,source.pointer,source.nbytes,device.stream)
       device._launch(entry[0],entry[1])
       for dest,source in after.get(i,()):device.driver.call('cuMemcpyDtoDAsync_v2',dest.pointer,source.pointer,source.nbytes,device.stream)
     resources=tuple(f.buffers.values())+tuple(f.kernels.values())+tuple(v for s in snapshots for v in s['values'].values())
     with CudaGraph(device,submit,resources=resources) as graph:graph.launch();device.synchronize()
     actual=f.buffers[frame.prediction].to_numpy().reshape(1,chunk)[0,:len(args[1][0])].tolist()
     if actual!=result[0]:raise ValueError('Snapshot capture changed verifier predictions')
     for s in snapshots:s['expected']=s['values']['out'].to_numpy()
     return result
    try:
     with SpeculativeSession(decoder,batch=1,chunk=512,timeout=120) as session:
      session.reset();session.prefill([item['token_ids']]);session.decode(127);expected=list(session.requests[0].output)
     target.verify=observe
     with SpeculativeSession(decoder,batch=1,chunk=512,timeout=120) as session:
      session.reset();session.prefill([item['token_ids']]);session.decode(127)
      if session.requests[0].output!=expected:raise ValueError('Snapshots changed complete generation')
    finally:target.verify=original
    if not snapshots:raise ValueError('No matching verification round')
    selected_bytes=0;counts=[]
    for s in snapshots:
     c=s['values']['counts'].to_numpy();active=int(np.count_nonzero(c));selected_bytes+=active*s['values']['w'].nbytes//len(c)
     if not s['parameters']['routed_input'] and len(counts)%2==0:pass
     counts.append(dict(call=s['index'],depth=s['parameters']['k'],counts=c.tolist(),active_experts=active))
    meta.update(case_id=item['id'],chunk=chunk,projection_calls=len(snapshots),selected_weight_payload_bytes=selected_bytes,
                route_counts=counts,output_exact=True,qualification='Actual verifier inputs copied on the device before workspace reuse. All 114 expert projections retain their original weights. Selected payload bytes are logical requested weight data, not measured DRAM traffic. Complete diagnostic generation matches unchanged speculative generation.')
    rec.program(TARGET);rec.program(DRAFT);snapshot_run=rec.id
    print('SNAPSHOT',len(snapshots),'LOGICAL_WEIGHT_BYTES',selected_bytes,flush=True)
   elif command['action']=='micro':
    if not snapshots:raise ValueError('Create snapshots first')
    path=Path(command['directory']);compiled=json.loads((path/'compilation.json').read_text());geometries=command.get('geometries')
    candidates=[('baseline-before',None)]
    choices=sorted({x['name'].split('-r')[1] for x in compiled})
    for geometry in choices:
     if geometries and geometry not in geometries:continue
     candidates.append((geometry,[x for x in compiled if x['name'].split('-r')[1]==geometry]))
    candidates.append(('baseline-after',None))
    checks=[]
    for label,selection in candidates:
     with ExitStack() as temporary:
      modules={}
      if selection:
       for item in selection:
        module=device.load(path/(item['name']+'.tbin'));temporary.callback(module.release);modules[item['kind']]=module
      plan=ExecutionPlan(device);resources=[];bindings=[]
      for s in snapshots:
       key='down' if s['parameters']['routed_input'] else 'gate';module=modules.get(key,s['kernel'])
       entry=bind_named(device,module,s['values']);plan.append(entry);resources.extend((module,*s['values'].values()));bindings.append(s)
      with plan.capture(resources=tuple(dict.fromkeys(resources))) as graph:
       graph.launch();device.synchronize();bitwise=True;maximum=0.
       for s in bindings:
        actual=s['values']['out'].to_numpy();expected=s['expected'];np.testing.assert_allclose(actual,expected,rtol=.002,atol=.0005)
        bitwise=bitwise and np.array_equal(actual.view('uint32'),expected.view('uint32'))
        maximum=max(maximum,float(np.max(np.abs(actual-expected))))
       timing=measure_cuda(device,graph.launch,warmup=2,samples=7,repeats=2)
      check=dict(geometry=label,timing=timing,bitwise=bool(bitwise),max_absolute_error=maximum,full_outputs_correct=True)
      checks.append(check);meta['checks']=checks;rec.update(report)
      print('MICRO',label,timing['median_gpu_seconds']*1000,'BITWISE',bitwise,flush=True)
      if selection:
       for item in selection:rec.artifact(path/(item['name']+'.tbin'),'kernels/'+item['name']+'.tbin')
    rec.artifact(path/'compilation.json','kernels/compilation.json')
    meta.update(snapshot_run=snapshot_run,qualification='All expert projections replay in one captured stream over actual routed inputs and distinct resident model weight banks. Each full output is checked against the original snapshot. Diagnostic kernel timing, not complete decode throughput.')
   else:raise ValueError('Unknown action')
   passed=True
  except BaseException as error:
   meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
  finally:
   rec.artifact(__file__,'profiling/runner.py');rec.update(report,final=True,status='passed' if passed else 'failed')
   print('RETAINED',rec.id,passed,flush=True)
