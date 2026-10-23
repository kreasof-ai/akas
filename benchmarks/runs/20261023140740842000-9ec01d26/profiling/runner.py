"""Time verifier operations with event nodes inside one captured full forward."""
from collections import defaultdict
from contextlib import ExitStack
import ctypes as c
import json
from pathlib import Path
import resource
import statistics
import traceback
import tensor
from tensor.providers.cuda.graph import CudaGraph
from akas.benchmarking.history import Recorder, write
from akas.benchmarking.speculative import SpeculativeSession
from akas.generation.speculative.decoding import SpeculativeDecoder
from akas.models.dspark.proposer import load
from akas.models.lfm2.model import LFM2MoE

ROOT=Path('/home/sagemaker-user/akas'); OUT=ROOT/'build/lfm-moe/verifier-device-profile'; OUT.mkdir(exist_ok=True)
TARGET=ROOT/'build/lfm-moe/dspark-short-optimized/target'
DRAFT=ROOT/'build/lfm-moe/dspark-short-q8/draft'
CP=ROOT/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
workload=json.loads((ROOT/'build/lfm-moe/dspark-short-q8/short-workload.json').read_text())
meta=dict(runtime='akas',purpose='profiling',protocol='captured-verifier-operation-events-v1',
 model='akas.models.lfm2.model:LFM2MoE',mode='dspark',provider='cuda',target='sm_89',
 device=dict(name='NVIDIA L4'),weight_format='q4_0',draft_weight_dtype='q8_0',kv_dtype='float16',draft_kv_dtype='float16',
 checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',
 shape_timeout_seconds=120,profiles=[],qualification='Event nodes are inserted between operation launches inside a single whole-forward CUDA graph. All operations execute in original order with live inputs, avoiding isolated stale workspace replay and host gaps. Event nodes perturb performance, so these are diagnostic attributions. Unchanged verifier predictions and final speculative output required.')
rec=Recorder(ROOT/'benchmarks',meta,kind='algorithm',parent='20261023135028708663-743b0598',change='Attribute real short verifier GPU execution with event nodes inside complete forwards')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);write(OUT/'run-id.json',dict(run_id=rec.id));print('RUN',rec.id,flush=True);passed=False
try:
 with tensor.Device(provider='cuda') as dev,ExitStack() as owned:
  target=owned.enter_context(LFM2MoE(CP,TARGET,dev,graphs=True))
  head=load(target,checkpoint=CP.with_name('dspark-q8_0.gguf'),bundle=DRAFT,graphs=True,confidence_threshold=.5);owned.callback(head.close)
  decoder=SpeculativeDecoder(target,head,window=10,capture=False);original=target.verify;done=set();case_id=None
  def observe(*args,**kwargs):
   result=original(*args,**kwargs);chunk=kwargs['chunk']
   if chunk not in (2,4,10) or (case_id,chunk) in done:return result
   done.add((case_id,chunk));frame=target.batches.select(1,chunk,phase='verify');f=frame.forward
   events=[]
   try:
    for _ in range(len(f.plan)+1):
     event=c.c_void_p();dev.driver.call('cuEventCreate',c.byref(event),0);events.append(event)
    def submit():
     dev.driver.call('cuEventRecord',events[0],dev.stream)
     for i,entry in enumerate(f.plan):
      dev._launch(entry[0],entry[1]);dev.driver.call('cuEventRecord',events[i+1],dev.stream)
    with CudaGraph(dev,submit,resources=tuple(f.buffers.values())+tuple(f.kernels.values())) as graph:
     samples=[];totals=[]
     for sample in range(5):
      graph.launch();dev.synchronize();times=[]
      for start,end in zip(events,events[1:]):
       elapsed=c.c_float();dev.driver.call('cuEventElapsedTime',c.byref(elapsed),start,end);times.append(elapsed.value)
      if sample:samples.append(times);totals.append(sum(times))
     actual=f.buffers[frame.prediction].to_numpy().reshape(frame.rows,chunk)[0,:len(args[1][0])].tolist()
     if actual!=result[0]:raise ValueError('Instrumented verifier prediction changed')
    entries=[dict(kind=call.kind,parameters=call.parameters,label=call.label,ms=statistics.median(s[i] for s in samples)) for i,call in enumerate(f.definition.calls)]
    grouped=defaultdict(float)
    for entry in entries:grouped[entry['kind']]+=entry['ms']
    profile=dict(id=case_id,chunk=chunk,position=int(target.positions[args[0][0]]),total_ms=statistics.median(totals),grouped_ms=dict(sorted(grouped.items(),key=lambda a:-a[1])),operations=entries,predictions_exact=True)
    meta['profiles'].append(profile);rec.update(report);write(OUT/'report.json',report)
    print('PROFILE',case_id,chunk,json.dumps(profile['grouped_ms']),flush=True)
   finally:
    dev.synchronize()
    for event in events:dev.driver.call('cuEventDestroy_v2',event)
   return result
  for item in workload['cases']:
   if item['id'] not in ('mtbench:81:33','gsm8k-train:0:48'):continue
   case_id=item['id'];print('MEASURE',case_id,flush=True)
   with SpeculativeSession(decoder,batch=1,chunk=512,timeout=120) as session:
    session.reset();session.prefill([item['token_ids']]);session.decode(127);expected=list(session.requests[0].output)
   target.verify=observe
   try:
    with SpeculativeSession(decoder,batch=1,chunk=512,timeout=120) as session:
     session.reset();session.prefill([item['token_ids']]);session.decode(127)
     if expected!=session.requests[0].output:raise ValueError('Instrumentation changed complete speculative output')
   finally:target.verify=original
  passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 rec.artifact(__file__,'profiling/runner.py');rec.program(TARGET);rec.program(DRAFT)
 write(OUT/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
