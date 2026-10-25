"""Quick pinned C4 continuation sweep through existing qualified public programs."""
from contextlib import ExitStack
import json,resource,sys,time,traceback
from pathlib import Path
import tensor
from akas.artifacts.read import read_program
from akas.benchmarking.cases import Case
from akas.benchmarking.history import Recorder,write,digest
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.benchmarking.speculative import SpeculativeSession
from akas.generation.speculative.decoding import SpeculativeDecoder
from akas.models.lfm2.model import LFM2MoE
from akas.models.dspark.proposer import load
ROOT=Path('/home/sagemaker-user/akas');OUT=ROOT/'build/lfm-moe/c4-final';kv=sys.argv[1];program=OUT/f'target-{kv}';draft=ROOT/'build/lfm-moe/dspark-short-q8/draft';checkpoint=ROOT/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf';workload_path=OUT/'workload.json';workload=json.loads(workload_path.read_text())
meta=dict(runtime='akas',purpose='benchmark',protocol='native-token-step-v2',provider='cuda',target='sm_89',model='akas.models.lfm2.model:LFM2MoE',mode='dspark',weight_format='q4_0',kv_dtype=kv,draft_kv_dtype='float16',draft_weight_dtype='q8_0',checkpoint_sha256=workload['tokenizer_checkpoint_sha256'],draft_checkpoint_sha256='eafaf6d19e653ce4b90abd928a832745a981ee48b3ecee8c0d37e894c4e29988',output_tokens=128,ignore_eos=True,graphs=True,round_capture=False,window=10,block_size=9,confidence_threshold=.5,dataset_split='heldout',text_workload_sha256=digest(workload_path),warmup=0,repeats=1,shape_timeout_seconds=180,variant='Concatenated C4 quick context sweep',qualification='Quick diagnostic: one nested concatenated C4 validation token stream, one measured sample per prefix, no per-shape warmup. Existing independently qualified kernels, unchanged Q4 target and Q8 draft. Same resident target/draft during AR and DSpark timing. AR uses device feedback; speculative timing includes proposal, verification, commit, repair and coordinator. Separate untimed AR pass retains output IDs. One short untimed initialization pass before measurement. No HTTP or corpus accuracy claim.')
rec=Recorder(ROOT/'benchmarks',meta,kind='configuration',parent='20261023151533293524-174061e4',change='Quick C4 AR/DSpark comparison over all eight context lengths');write(OUT/f'{kv}-run-id.json',dict(run_id=rec.id));report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False;print('RUN',kv,rec.id,flush=True)
try:
 with tensor.Device(provider='cuda') as device,ExitStack() as owned:
  meta['device']=device.info;target=owned.enter_context(LFM2MoE(checkpoint,program,device,graphs=True));head=load(target,checkpoint=checkpoint.with_name('dspark-q8_0.gguf'),bundle=draft,confidence_threshold=.5,graphs=True);owned.callback(head.close);decoder=SpeculativeDecoder(target,head,window=10,capture=False)
  with Session(target,batch=1,chunk=4096,timeout=180) as s:run(s,Case(1,32,4),prompts=[workload['cases'][0]['token_ids']],warmup=0,repeats=1)
  with SpeculativeSession(decoder,batch=1,chunk=4096,timeout=180) as s:run(s,Case(1,32,4),prompts=[workload['cases'][0]['token_ids']],warmup=0,repeats=1)
  for item in workload['cases']:
   ids=item['token_ids'];case=Case(1,len(ids),128);print('MEASURE',kv,len(ids),'AR',flush=True)
   with Session(target,batch=1,chunk=4096,timeout=180) as s:
    ar=run(s,case,prompts=[ids],warmup=0,repeats=1);s.reset();reference=[int(s.prefill([ids])[0])]
    for _ in range(127):s._check_timeout();reference.append(int(s.step([reference[-1]])[0]))
   print('MEASURE',kv,len(ids),'DS',flush=True)
   with SpeculativeSession(decoder,batch=1,chunk=4096,timeout=180) as s:
    row=run(s,case,prompts=[ids],warmup=0,repeats=1);actual=list(s.requests[0].output);stats=s.stats
   if len(actual)!=128 or len(reference)!=128 or row['prompt_sha256']!=item['prompt_sha256']:raise ValueError('Incomplete or different workload')
   row.update(text_case={k:v for k,v in item.items() if k!='token_ids'},ar=ar,native_speculation=stats,output_ids=[actual],output_text=target.checkpoint.tokenizer.decode(actual),ar_output_ids=[reference],ar_output_text=target.checkpoint.tokenizer.decode(reference),output_agreement=dict(exact=actual==reference,common_prefix_tokens=next((i for i,(a,b) in enumerate(zip(actual,reference)) if a!=b),128)),correctness=dict(correct=True,scope=meta['qualification']))
   report['results'].append(row);rec.update(report);write(OUT/f'{kv}-report.json',report)
   print('RESULT',kv,len(ids),'AR',round(ar['summary']['prefill_tokens_per_second']['median'],2),round(ar['summary']['decode_tokens_per_second']['median'],2),'DS',round(row['summary']['prefill_tokens_per_second']['median'],2),round(row['summary']['decode_tokens_per_second']['median'],2),'ACCEPT',stats['accepted'],stats['proposed'],flush=True)
  passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 for p in (program,draft):rec.program(p)
 rec.artifact(__file__,'benchmark/runner.py');rec.artifact(workload_path,'benchmark/text-workload.json');rec.artifact(OUT/'prepare-c4.py','benchmark/prepare-c4.py');rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,rec.record['status'],flush=True)
if not passed:raise SystemExit(1)
