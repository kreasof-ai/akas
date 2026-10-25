"""Reproduce both AOT programs with shared immutable weights and separate KV state."""
import json,time,traceback
from pathlib import Path
import numpy as np
import tensor
from akas.models.lfm2.model import LFM2MoE
from akas.execution.executor import Executor
from akas.execution.batch import BatchBank
from akas.artifacts.read import read_program
from akas.benchmarking.session import Session
from akas.benchmarking.history import Recorder,write
from akas.optimization.evaluation import prompts
root=Path('build/lfm-moe/long-greedy');root.mkdir(exist_ok=True)
old_dir=Path('build/lfm-moe/manual-v2/cuda/program');new_dir=Path('build/lfm-moe/manual-v3/cuda/program')
old_report=json.loads(Path('build/single-user-long-context/lfm2.5-24b-a2b-q4_0-cuda-default-ar/jobs/ee9b3acb399892d0/attempt-0001/report.json').read_text())
old_ids=next(row for row in old_report['results'] if row['case']['prompt']==130944)['output_ids'][0]
meta=dict(runtime='akas',purpose='kernel',protocol='teacher-forced-greedy-diagnostic-v1',provider='cuda',device=dict(name='NVIDIA L4',provider='cuda'),kv_dtype='float16',reference_program=str(old_dir),candidate_program=str(new_dir),reference_run_id='20261020102328416670-be712fbb',candidate_run_id='20261020123109563937-619ed8db',qualification='Diagnose candidate greedy rejection; both immutable AOT bundles borrow identical packed weight storage but maintain separate KV states; reproduce retained reference IDs, compare all 128 teacher-forced next-token decisions')
rec=Recorder('benchmarks',meta,kind='diagnostic',change='Locate the first full-context greedy divergence without weakening the acceptance check',parent=meta['candidate_run_id'])
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False;checks=[];saved={}
def borrowed(owner,directory):
 program=read_program(directory);weights={name:value.view(value.shape) for name,value in owner.weights.items()}
 bank=BatchBank(directory,owner.device,program['phases'],weights,graphs=True,reset=program.get('reset'))
 executor=Executor(capacity=program['max_requests'],context=program['max_context'],device=owner.device,batches=bank,reset=bank.reset)
 executor.checkpoint=owner.checkpoint;executor.config=owner.config;executor.program=program;executor.bundle=directory;executor.weights=weights
 return executor
try:
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf',new_dir,device,graphs=True) as candidate:
   reference=[];prompt=prompts(candidate.config.vocab,1,130944)
   with borrowed(candidate,old_dir) as baseline:
    print('REFERENCE_PREFILL',flush=True)
    with Session(baseline,batch=1,chunk=512,timeout=120) as session:
     session.reset();session.prefill(prompt)
     for step,expected_token in enumerate(old_ids):
      values=session.read_feature('logits').reshape(-1,baseline.config.vocab)[-1].copy()
      if int(values.argmax())!=expected_token:raise ValueError(f'reference does not reproduce retained token {step}')
      reference.append(values)
      if step<127:session.step([expected_token])
   print('REFERENCE_REPRODUCED',flush=True)
   for chunk in (4096,512):
    print('CANDIDATE_PREFILL',chunk,flush=True)
    with Session(candidate,batch=1,chunk=chunk,timeout=120) as right:
     right.reset();right.prefill(prompt);mismatches=[];max_relative=0.;first=True
     for step,expected_token in enumerate(old_ids):
      a=reference[step].astype('float64');b=right.read_feature('logits').reshape(-1,candidate.config.vocab)[-1].astype('float64')
      delta=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30));max_relative=max(max_relative,delta)
      if int(b.argmax())!=expected_token:
       second=int(np.argsort(a)[-2]);found=int(b.argmax())
       detail=dict(step=step,expected=expected_token,actual=found,reference_top2_margin=float(a[expected_token]-a[second]),reference_expected_minus_actual=float(a[expected_token]-a[found]),candidate_expected_minus_actual=float(b[expected_token]-b[found]),relative_logit_error=delta)
       mismatches.append(detail);print('MISMATCH',json.dumps(detail),flush=True)
       if first:saved[f'chunk{chunk}-reference']=a.astype('float32');saved[f'chunk{chunk}-candidate']=b.astype('float32');first=False
      if step<127:right.step([expected_token])
     check=dict(prefix=130944,chunk=chunk,teacher_forced_tokens=128,reference_reproduced=True,mismatches=mismatches,top1_agreement=(128-len(mismatches))/128,max_relative_logit_error=max_relative,greedy_equivalent=len(mismatches)==0)
     checks.append(check);print('DIAGNOSED',json.dumps(check),flush=True)
   # Profile the current large-chunk forward in actual model order, once.
   import sys
   sys.path.insert(0,str(Path('build/lfm-moe').resolve()))
   from cuda_profile import Profile,summarize
   frame=candidate.batches.select(1,4096);profile=Profile(frame.forward);original=frame.forward.run;samples=[]
   def measured():profile.graph.launch();samples.append(profile.read())
   frame.forward.run=measured
   try:
    with Session(candidate,batch=1,chunk=4096,timeout=120) as session:
     session.reset();session.prefill(prompts(candidate.config.vocab,1,8192),read=False)
   finally:frame.forward.run=original;profile.close()
   meta['prefill_profile']=summarize(profile,samples)
 meta['checks']=checks;np.savez_compressed(root/'first-mismatch.npz',**saved);write(root/'diagnostic.json',meta);passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks;rec.artifact(__file__,'diagnostic/runner.py');rec.artifact('build/lfm-moe/cuda_profile.py','diagnostic/cuda_profile.py')
 if (root/'first-mismatch.npz').exists():rec.artifact(root/'first-mismatch.npz','diagnostic/first-mismatch.npz')
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
