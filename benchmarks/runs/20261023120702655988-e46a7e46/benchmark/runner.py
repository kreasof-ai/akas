"""Compare complete F16 AR and DSpark execution on pinned real-text inputs."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import resource
import traceback
import numpy as np
import tensor
from akas.benchmarking.cases import Case
from akas.benchmarking.history import Recorder,write,digest
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.benchmarking.speculative import SpeculativeSession
from akas.generation.speculative.decoding import SpeculativeDecoder
from akas.models.lfm2.model import LFM2MoE
from akas.models.dspark.proposer import load

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--split',choices=('calibration','heldout'),required=True)
parser.add_argument('--threshold',type=float,default=0)
args=parser.parse_args()
root=Path('/home/sagemaker-user/akas')
workload_path=root/'build/datasets/dspark/workload.json'
workload=json.loads(workload_path.read_text())
compiled=root/'build/lfm-moe/dspark-long-qualification'
checkpoint=root/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
meta=dict(runtime='akas',purpose='benchmark',protocol='native-token-step-v2',provider='cuda',target='sm_89',
 model='akas.models.lfm2.model:LFM2MoE',mode='dspark',kv_dtype='float16',draft_kv_dtype='float16',
 checkpoint_sha256=workload['tokenizer_checkpoint_sha256'],draft_checkpoint_sha256='0cd847ca049aeabf53cb41d62091ae20eb50245cf89594c75af204784a1b1a0f',
 weight_format='q4_0',output_tokens=128,ignore_eos=True,graphs=True,round_capture=False,window=10,
 confidence_threshold=args.threshold,dataset_split=args.split,text_workload_sha256=digest(workload_path),
 qualification='Runtime calibration probe. Existing independently qualified F16 target/draft programs, unchanged weights. Identical real-token input for AR and speculative execution. AR uses device feedback; speculative timing includes proposal, verification, commit, repair and coordinator readbacks. Separate untimed AR pass retains all output tokens. Finite outputs and exact/common-prefix output comparisons retained; no corpus accuracy claim. Long cases have one sample, short cases one warmup and three samples. Target and draft allocations resident during both timings.')
selection_path=root/'build/lfm-moe/dspark-runtime-selection.json'
parent='20261023105846475312-580635c0'
if args.split=='heldout':
 selection=json.loads(selection_path.read_text())
 if selection['heldout_used'] or selection['weights_changed'] or args.threshold!=selection['selected']['threshold']:
  raise ValueError('Held-out settings must match the predeclared calibration winner')
 meta['runtime_calibration']=selection
 meta['deployment_recommendation']='AR remains preferred: calibrated DSpark is slower overall; this is a diagnostic held-out evaluation.'
 parent=selection['selected']['run_id']
rec=Recorder(root/'benchmarks',meta,kind='configuration',parent=parent,change='Measure real-text DSpark acceptance against the exact Q4 target before runtime calibration')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[])
out=root/'build/lfm-moe'/f'dspark-real-{args.split}-{args.threshold:g}'
out.mkdir(exist_ok=True);write(out/'run-id.json',dict(run_id=rec.id))
print('RUN',rec.id,flush=True)
passed=False
try:
 with tensor.Device(provider='cuda') as device,ExitStack() as owned:
  target=owned.enter_context(LFM2MoE(checkpoint,compiled/'target',device,graphs=True))
  head=load(target,checkpoint=checkpoint.with_name('dspark-f16.gguf'),bundle=compiled/'draft',confidence_threshold=args.threshold,graphs=True)
  owned.callback(head.close)
  decoder=SpeculativeDecoder(target,head,window=10,capture=False)
  for item in workload['cases']:
   if item['split']!=args.split:continue
   ids=item['token_ids'];count=len(ids);case=Case(1,count,128)
   long=count>=8192;warmup=0 if long else 1;repeats=1 if long else 3
   print('MEASURE',item['id'],item['category'],count,flush=True)
   with Session(target,batch=1,chunk=4096,timeout=180) as session:
    ar=run(session,case,prompts=[ids],warmup=warmup,repeats=repeats)
    session.reset();reference=[int(session.prefill([ids])[0])]
    for _ in range(127):reference.append(int(session.step([reference[-1]])[0]))
   with SpeculativeSession(decoder,batch=1,chunk=4096,timeout=180) as session:
    row=run(session,case,prompts=[ids],warmup=warmup,repeats=repeats)
    actual=list(session.requests[0].output)
    row['native_speculation']=session.stats
   if row['prompt_sha256']!=item['prompt_sha256'] or ar['prompt_sha256']!=item['prompt_sha256']:
    raise ValueError('real prompt identity changed')
   row.update(text_case={k:v for k,v in item.items() if k!='token_ids'},ar=ar,
    output_ids=[actual],output_text=target.checkpoint.tokenizer.decode(actual),
    ar_output_ids=[reference],ar_output_text=target.checkpoint.tokenizer.decode(reference),
    output_agreement=dict(exact=reference==actual,common_prefix_tokens=next((i for i,(a,b) in enumerate(zip(reference,actual)) if a!=b),128)),
    correctness=dict(correct=True,scope=meta['qualification']))
   row['decode_speedup']=row['summary']['decode_tokens_per_second']['median']/ar['summary']['decode_tokens_per_second']['median']
   report['results'].append(row);rec.update(report);write(out/'report.json',report)
   print('RESULT',item['id'],'AR',ar['summary']['decode_tokens_per_second']['median'],'DS',row['summary']['decode_tokens_per_second']['median'],'speedup',row['decode_speedup'],'stats',row['native_speculation'],'agreement',row['output_agreement'],flush=True)
 passed=True
except BaseException as error:
 meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 for name in ('target','draft'):rec.program(compiled/name)
 rec.artifact(__file__,'benchmark/runner.py');rec.artifact(workload_path,'benchmark/text-workload.json')
 if args.split=='heldout':rec.artifact(selection_path,'benchmark/runtime-selection.json')
 write(out/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed')
 print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
