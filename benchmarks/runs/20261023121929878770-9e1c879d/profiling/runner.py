"""Attribute unchanged DSpark phase latency and captured operation latency."""
from collections import defaultdict
from contextlib import ExitStack
import json,resource,time,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.compiler.tuning import measure
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.speculative import SpeculativeSession
from akas.execution.plan import ExecutionPlan
from akas.generation.speculative.decoding import SpeculativeDecoder
from akas.models.lfm2.model import LFM2MoE
from akas.models.dspark.proposer import load
root=Path('/home/sagemaker-user/akas');out=root/'build/lfm-moe/dspark-round-profile';out.mkdir(exist_ok=True)
workload=json.loads((root/'build/datasets/dspark/workload.json').read_text())
compiled=root/'build/lfm-moe/dspark-long-qualification';checkpoint=root/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
meta=dict(runtime='akas',purpose='profiling',protocol='speculative-round-profile-v1',provider='cuda',target='sm_89',kv_dtype='float16',draft_kv_dtype='float16',checkpoint_sha256=workload['tokenizer_checkpoint_sha256'],cases=[],qualification='Synchronized wall-clock phase attribution adds boundaries to the unchanged coordinator; diagnostic attribution, not end-to-end throughput. Per-operation CUDA-event timings replay captured bindings after generation and do not validate new mathematics.')
rec=Recorder(root/'benchmarks',meta,kind='algorithm',parent='20261023120702655988-e46a7e46',change='Attribute real-text DSpark proposal, verification, commit, repair and coordinator latency before optimization')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);write(out/'run-id.json',dict(run_id=rec.id));print('RUN',rec.id,flush=True);passed=False
try:
 with tensor.Device(provider='cuda') as device,ExitStack() as owned:
  target=owned.enter_context(LFM2MoE(checkpoint,compiled/'target',device,graphs=True))
  head=load(target,checkpoint=checkpoint.with_name('dspark-f16.gguf'),bundle=compiled/'draft',graphs=True);owned.callback(head.close)
  decoder=SpeculativeDecoder(target,head,window=10,capture=False);samples=defaultdict(list)
  for obj,name,label in ((head,'propose','proposal'),(target,'verify','verify'),(target,'commit','commit'),(head,'repair','repair')):
   original=getattr(obj,name)
   def measured(*args,_function=original,_label=label,**kwargs):
    device.synchronize();start=time.perf_counter();value=_function(*args,**kwargs);device.synchronize();samples[_label].append(time.perf_counter()-start);return value
   setattr(obj,name,measured)
  wanted=[('mtbench:82:57',.5),('gsm8k-train:0:48',0),('longbench-v2:66f94f9ebb02136c067c4fde:8192',.8),('longbench-v2:66f94f9ebb02136c067c4fde:130944',.5)]
  for case_id,threshold in wanted:
   item=next(c for c in workload['cases'] if c['id']==case_id);head.confidence_threshold=threshold;samples.clear();print('MEASURE',case_id,threshold,flush=True)
   with SpeculativeSession(decoder,batch=1,chunk=4096,timeout=180) as session:
    session.reset();session.prefill([item['token_ids']]);device.synchronize();start=time.perf_counter();session.decode(127);device.synchronize();total=time.perf_counter()-start;stats=session.stats;outputs=list(session.requests[0].output)
   case=dict(id=case_id,confidence_threshold=threshold,stats=stats,seconds=total,output_ids=outputs,phases={})
   for name,values in samples.items():case['phases'][name]=dict(calls=len(values),total_ms=sum(values)*1000,steady_median_ms=float(np.median(values[2:] or values))*1000,steady_mean_ms=float(np.mean(values[2:] or values))*1000)
   case['coordinator_ms']=(total-sum(sum(v) for v in samples.values()))*1000
   print('PHASES',json.dumps({k:v for k,v in case.items() if k!='output_ids'}),flush=True)
   if item['prefix_tokens'] in (57,8192):
    case['operations']={}
    for label,frame in (('proposal',head.draft.last_frame),('verify',target.batches.select(1,10,phase='verify'))):
     rows=[]
     for index,(call,entry) in enumerate(zip(frame.forward.definition.calls,frame.forward.plan)):
      plan=ExecutionPlan(device,[entry],labeled=True);prepared=plan.capture(resources=tuple(frame.buffers.values())+tuple(frame.forward.kernels.values()))
      try:timed=measure(device,prepared.launch,warmup=1,samples=2,repeats=5)
      finally:prepared.close()
      rows.append(dict(index=index,kind=call.kind,parameters=call.parameters,label=call.label,microseconds=timed['median_seconds']*1e6))
     case['operations'][label]=rows;grouped=defaultdict(float)
     for row in rows:grouped[row['kind']]+=row['microseconds']
     print('OPERATIONS',case_id,label,json.dumps(dict(sorted(grouped.items(),key=lambda x:-x[1]))),flush=True)
   meta['cases'].append(case);rec.update(report);write(out/'report.json',report)
 passed=True
except BaseException as error:meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;rec.artifact(__file__,'profiling/runner.py');write(out/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
