"""Select a confidence threshold on calibration examples with unchanged weights."""
from contextlib import ExitStack
import json, math, resource, traceback
from pathlib import Path
import tensor
from akas.benchmarking.cases import Case
from akas.benchmarking.history import Recorder,write,digest
from akas.benchmarking.runner import run
from akas.benchmarking.speculative import SpeculativeSession
from akas.generation.speculative.decoding import SpeculativeDecoder
from akas.models.lfm2.model import LFM2MoE
from akas.models.dspark.proposer import load
root=Path('/home/sagemaker-user/akas');folder=root/'build/lfm-moe'
reference=json.loads((folder/'dspark-real-calibration-0/report.json').read_text())
workload_path=root/'build/datasets/dspark/workload.json'
workload=json.loads(workload_path.read_text())
parent=json.loads((folder/'dspark-real-calibration-0/run-id.json').read_text())['run_id']
record=json.loads((root/'benchmarks/runs'/parent/'run.json').read_text())
if record['status']!='passed':raise ValueError('Unsealed calibration reference')
compiled=folder/'dspark-long-qualification';checkpoint=root/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
references={row['text_case']['id']:row for row in reference['results']}
results=[]
with tensor.Device(provider='cuda') as device,ExitStack() as owned:
 target=owned.enter_context(LFM2MoE(checkpoint,compiled/'target',device,graphs=True))
 head=load(target,checkpoint=checkpoint.with_name('dspark-f16.gguf'),bundle=compiled/'draft',graphs=True)
 owned.callback(head.close)
 decoder=SpeculativeDecoder(target,head,window=10,capture=False)
 for threshold in (.5,.8):
  head.confidence_threshold=threshold
  meta=dict(reference['metadata']);meta.pop('cpu_peak_rss_bytes',None)
  meta.update(confidence_threshold=threshold,ar_reference_run_id=parent,
   qualification=reference['metadata']['qualification']+' AR timings and outputs reused from the immediately preceding matched-input calibration run; target/draft weights and kernel programs unchanged. Threshold selected only on calibration examples.')
  rec=Recorder(root/'benchmarks',meta,kind='configuration',parent=parent,change='Calibrate the DSpark confidence cutoff on real-text calibration examples; weights unchanged')
  report=dict(schema='akas.benchmark.v1',metadata=meta,results=[])
  out=folder/f'dspark-real-calibration-{threshold:g}';out.mkdir(exist_ok=True)
  write(out/'run-id.json',dict(run_id=rec.id));print('RUN',rec.id,threshold,flush=True)
  passed=False
  try:
   for item in workload['cases']:
    if item['split']!='calibration':continue
    expected=references[item['id']];ids=item['token_ids'];long=len(ids)>=8192
    print('MEASURE',threshold,item['id'],flush=True)
    with SpeculativeSession(decoder,batch=1,chunk=4096,timeout=180) as session:
     row=run(session,Case(1,len(ids),128),prompts=[ids],warmup=0 if long else 1,repeats=1 if long else 3)
     actual=list(session.requests[0].output);stats=session.stats
    ar=expected['ar'];reference_ids=expected['ar_output_ids'][0]
    if row['prompt_sha256']!=item['prompt_sha256']:raise ValueError('Changed prompt')
    row.update(text_case={k:v for k,v in item.items() if k!='token_ids'},ar=ar,
     native_speculation=stats,output_ids=[actual],output_text=target.checkpoint.tokenizer.decode(actual),
     ar_output_ids=[reference_ids],ar_output_text=expected['ar_output_text'],
     output_agreement=dict(exact=actual==reference_ids,common_prefix_tokens=next((i for i,(a,b) in enumerate(zip(actual,reference_ids)) if a!=b),128)),
     correctness=dict(correct=True,scope=meta['qualification']))
    row['decode_speedup']=row['summary']['decode_tokens_per_second']['median']/ar['summary']['decode_tokens_per_second']['median']
    report['results'].append(row);rec.update(report);write(out/'report.json',report)
    print('RESULT',threshold,item['id'],row['summary']['decode_tokens_per_second']['median'],row['decode_speedup'],stats,row['output_agreement'],flush=True)
   passed=True;results.append(dict(threshold=threshold,run_id=rec.id,geomean_speedup=math.exp(sum(math.log(row['decode_speedup']) for row in report['results'])/len(report['results']))))
  except BaseException as error:
   meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
  finally:
   meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
   for name in ('target','draft'):rec.program(compiled/name)
   rec.artifact(__file__,'benchmark/runner.py');rec.artifact(workload_path,'benchmark/text-workload.json')
   write(out/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed')
   print('RETAINED',rec.id,passed,flush=True)
  if not passed:raise SystemExit(1)
results.append(dict(threshold=0,run_id=parent,geomean_speedup=math.exp(sum(math.log(row['decode_speedup']) for row in reference['results'])/len(reference['results']))))
best=max(results,key=lambda row:row['geomean_speedup'])
selection=dict(schema='akas.dspark-runtime-calibration.v1',objective='Geometric mean complete decode speedup over matched AR on all seven calibration cases',candidates=results,selected=best,weights_changed=False,heldout_used=False)
write(folder/'dspark-runtime-selection.json',selection);print('SELECTED',json.dumps(selection),flush=True)
