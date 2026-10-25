"""Confirm short real-text decode gains with matching AR and serving outputs."""
from collections import Counter
from contextlib import ExitStack
import json,math,resource,traceback
from pathlib import Path
import tensor
from akas.benchmarking.cases import Case
from akas.benchmarking.history import Recorder,write,digest
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.benchmarking.speculative import SpeculativeSession
from akas.config import EngineConfig
from akas.engine import Engine
from akas.generation.speculative.decoding import SpeculativeDecoder
from akas.models.lfm2.model import LFM2MoE
from akas.models.dspark.proposer import load
root=Path('/home/sagemaker-user/akas');compiled=root/'build/lfm-moe/dspark-short-q8';target_program=root/'build/lfm-moe/dspark-short-optimized/target';checkpoint=root/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
preparation=json.loads((compiled/'run-id.json').read_text());prepared=json.loads((root/'benchmarks/runs'/preparation['run_id']/'run.json').read_text())
if prepared['status']!='passed':raise ValueError('Compiled pair must pass independent qualification')
workload=json.loads((root/'build/datasets/dspark/workload.json').read_text());workload['cases']=[c for c in workload['cases'] if c['prefix_tokens']<128];workload['scope']+=' Short-only latency gate; no long-context execution in this run.'
workload_path=compiled/'short-workload.json';write(workload_path,workload)
with tensor.Device(provider='cuda') as device,ExitStack() as owned:
 target=owned.enter_context(LFM2MoE(checkpoint,target_program,device,graphs=True))
 head=load(target,checkpoint=checkpoint.with_name('dspark-q8_0.gguf'),bundle=compiled/'draft',confidence_threshold=.5,graphs=True);owned.callback(head.close)
 from dspark_reference_q8 import qualify
 full_reference=qualify(target,head.draft);print('FULL_DRAFT_REFERENCE',json.dumps(full_reference),flush=True)
 decoder=SpeculativeDecoder(target,head,window=10,capture=False)
 engine=Engine(decoder,config=EngineConfig(prefill_chunk=512,prefill_requests=1))
 histogram=Counter();verify=target.verify
 def observed(*args,**kwargs):histogram[kwargs['chunk']]+=1;return verify(*args,**kwargs)
 target.verify=observed
 for group,parent in (('calibration','20261023124514712767-7ea28db7'),('heldout','20261023124621285769-4582c3e0')):
  previous=json.loads((root/'benchmarks/runs'/parent/'report.json').read_text());references={r['text_case']['id']:r for r in previous['results']}
  meta=dict(previous['metadata']);meta.pop('cpu_peak_rss_bytes',None);meta.pop('runtime_calibration',None);meta.pop('ar_reference_run_id',None);meta.pop('deployment_recommendation',None)
  meta['full_draft_reference']=full_reference
  meta.update(draft_weight_dtype='q8_0',draft_checkpoint_sha256='eafaf6d19e653ce4b90abd928a832745a981ee48b3ecee8c0d37e894c4e29988',conversion_run_id='20261023131317756274-aec9685a',variant='Q8_0 draft with smaller verification windows and tuned projections',verification_chunks=[1,2,4,10],program_max_context=131072,text_workload_sha256=digest(workload_path),previous_run_id=parent,confidence_threshold=.5,shape_timeout_seconds=120,
   qualification='Short-only real-text latency confirmation. F16 target/draft KV, unchanged Q4_0 target checkpoint and locally quantized Q8_0 draft. Target/draft kernels independently qualified through the public compiler; selected projection geometries have sampled F64 and complete bitwise proof. AR uses device feedback; speculative timing includes the whole coordinator. One warmup, three samples, 128 outputs with EOS disabled. Separate untimed AR token collection and exact native/Engine agreement; old speculative outputs retained separately. No long-context inference or dataset accuracy claim.')
  rec=Recorder(root/'benchmarks',meta,kind='geometry',parent=parent,change='Confirm complete short-prompt latency after dispatching actual verification lengths and tuning packed projections')
  report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);out=compiled/group;out.mkdir(exist_ok=True);write(out/'run-id.json',dict(run_id=rec.id));print('RUN',group,rec.id,flush=True);passed=False
  try:
   for item in workload['cases']:
    if item['split']!=group:continue
    ids=item['token_ids'];case=Case(1,len(ids),128);print('MEASURE',item['id'],flush=True)
    with Session(target,batch=1,chunk=512,timeout=120) as session:
     ar=run(session,case,prompts=[ids],warmup=1,repeats=3);session.reset();reference=[int(session.prefill([ids])[0])]
     for _ in range(127):reference.append(int(session.step([reference[-1]])[0]))
    histogram.clear()
    with SpeculativeSession(decoder,batch=1,chunk=512,timeout=120) as session:
     row=run(session,case,prompts=[ids],warmup=1,repeats=3);actual=list(session.requests[0].output);stats=session.stats
    native_shapes=dict(histogram)
    request=engine.submit(ids,max_tokens=128)
    while len(request.output)<128:engine.step()
    # Finish the completed request before admitting the next native session.
    while request.slot is not None:engine.step()
    if list(request.output)!=actual:raise ValueError('Native and Engine speculative output differ')
    old=references[item['id']]
    if row['prompt_sha256']!=item['prompt_sha256'] or old['prompt_sha256']!=item['prompt_sha256']:raise ValueError('Changed prompt identity')
    row.update(text_case={k:v for k,v in item.items() if k!='token_ids'},ar=ar,native_speculation=stats,verification_shape_counts_including_warmup=native_shapes,
     output_ids=[actual],output_text=target.checkpoint.tokenizer.decode(actual),ar_output_ids=[reference],ar_output_text=target.checkpoint.tokenizer.decode(reference),
     output_agreement=dict(exact=actual==reference,common_prefix_tokens=next((i for i,(a,b) in enumerate(zip(actual,reference)) if a!=b),128)),
     previous_speculative_run=parent,previous_decode_tps=old['summary']['decode_tokens_per_second']['median'],previous_output_ids=old['output_ids'],
     previous_output_exact=[actual]==old['output_ids'],engine_output_exact=True,correctness=dict(correct=True,scope=meta['qualification']))
    row['decode_speedup']=row['summary']['decode_tokens_per_second']['median']/ar['summary']['decode_tokens_per_second']['median'];row['improvement_over_previous']=row['summary']['decode_tokens_per_second']['median']/row['previous_decode_tps']
    report['results'].append(row);rec.update(report);write(out/'report.json',report)
    print('RESULT',item['id'],'AR',ar['summary']['decode_tokens_per_second']['median'],'DS',row['summary']['decode_tokens_per_second']['median'],'vs_AR',row['decode_speedup'],'vs_previous',row['improvement_over_previous'],'stats',stats,'shapes',native_shapes,'exact_old',row['previous_output_exact'],'exact_ar',row['output_agreement']['exact'],flush=True)
   meta['geomean_speedup_over_ar']=math.exp(sum(math.log(r['decode_speedup']) for r in report['results'])/len(report['results']));meta['geomean_improvement_over_previous']=math.exp(sum(math.log(r['improvement_over_previous']) for r in report['results'])/len(report['results']));passed=True
  except BaseException as error:meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
  finally:
   meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
   for program in (target_program,compiled/'draft'):rec.program(program)
   rec.artifact(Path(__file__).with_name('dspark_reference_q8.py'),'validation/full-draft-reference.py');rec.artifact(__file__,'benchmark/runner.py');rec.artifact(workload_path,'benchmark/text-workload.json');write(out/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
  if not passed:raise SystemExit(1)
 engine.close()
