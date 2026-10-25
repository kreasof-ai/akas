"""Paired natural-prompt outputs and common-history logits, with bounded host RAM."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import traceback
import numpy as np
import tensor
from akas.benchmarking.history import Recorder, digest, write
from akas.benchmarking.session import Session
from akas.models.lfm2.model import LFM2MoE

MODE = sys.argv[1]
ROOT = Path('build/lfm-moe/kv-output-comparison')
ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT = Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf')
PROGRAM = Path('build/lfm-moe') / ('cuda-f16-full/cuda/program' if MODE == 'float16' else 'cuda-ordered-full/cuda/program')
TARGET_STEPS = 32
TASKS = [
 ('garden', 'A gardener records rainfall, soil moisture and harvest dates each season. Explain in three sentences how these notes help her plan next year.'),
 ('bridges', 'Two students test paper bridges. One changes both the paper thickness and the bridge span. Describe a fair experiment that changes only one variable.'),
 ('catalogue', 'Write a small Python function that returns the titles of available books from a list of dictionaries with keys title, author, year and borrowed. Include one example.'),
 ('arithmetic', 'A shop has 17 boxes, each containing 24 pencils. It sells 135 pencils. How many pencils remain? Show the arithmetic briefly.'),
 ('sorting', 'Explain why a stable sorting algorithm is useful when sorting records by two different fields. Give a concrete example.'),
 ('summary', 'Summarize this in two sentences: The library opens at nine on weekdays and ten on Saturdays. Volunteers repair books on Tuesday. The science club meets every other Saturday. Members can borrow six books for three weeks.'),
]
meta = dict(runtime='akas', purpose='diagnostic', protocol='kv-output-comparison-v1',
 provider='cuda', target='sm_89', device=dict(name='NVIDIA L4',provider='cuda'),
 model='akas.models.lfm2.model:LFM2MoE', checkpoint=str(CHECKPOINT.resolve()),
 checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',
 kv_dtype=MODE, max_output_tokens=128, stop_at_eos=True, teacher_forced_steps=TARGET_STEPS,
 shape_timeout_seconds=120, chunk=4096, graphs=True,
 qualification='Six natural chat prompts and four exact-length prose-padded retrieval prompts; greedy output agreement and F16-common-history distribution drift. A controlled numerical probe, not a corpus accuracy evaluation.',
 program=str(PROGRAM.resolve()))
reference = json.loads((ROOT/'float16.json').read_text()) if MODE != 'float16' else None
rec = Recorder('benchmarks',meta,kind='diagnostic',
 parent=reference['run_id'] if reference else None,
 change='Measure F16 versus FP8 greedy text and shared-history logit drift separately from performance')
report = dict(schema='akas.benchmark.v1',metadata=meta,results=[])
output = dict(run_id=rec.id,metadata=meta,cases=[])
passed=False
try:
 checkpoint=LFM2MoE.checkpoint_type(CHECKPOINT)
 tokenizer=checkpoint.tokenizer
 cases=[dict(name=name,ids=tokenizer.chat(text),question=text) for name,text in TASKS]
 header=tokenizer.chat('Context follows.\n')
 # Keep chat closing/generation controls at the end of every padded prompt.
 empty=tokenizer.chat('')
 suffix=empty[empty.index(tokenizer.special['<|im_end|>']):]
 header=header[:-len(suffix)]
 filler=tokenizer.encode('The archive records a quiet day at the local library. Readers returned books, volunteers arranged shelves, and the librarian checked the catalogue.\n',add_bos=False)
 for prefix in (8192,130944):
  for variant,(location,word) in enumerate((('early','MARIGOLD'),('middle','COBALT'))):
   question=tokenizer.encode('\nQuestion: What is the archive access word? Reply with only the word.\n',add_bos=False)
   fact=tokenizer.encode('\nThe archive access word is '+word+'.\n',add_bos=False)
   n=prefix-len(header)-len(suffix)-len(question)-len(fact)
   body=np.resize(np.asarray(filler,dtype='int32'),n).tolist()
   insert=0 if location=='early' else n//2
   ids=header+body[:insert]+fact+body[insert:]+question+suffix
   assert len(ids)==prefix
   cases.append(dict(name=f'retrieval-{prefix}-{location}',ids=ids,answer=word))
 if reference:
  assert [(c['name'],c['prompt_ids']) for c in reference['cases']]==[(c['name'],c['ids']) for c in cases]
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(CHECKPOINT,PROGRAM,device,graphs=True) as model:
   for case in cases:
    name,ids=case['name'],case['ids']
    row=dict(name=name,prefix=len(ids),prompt_ids=ids,
     prompt_sha256=hashlib.sha256(np.asarray(ids,dtype='int32').tobytes()).hexdigest())
    if 'answer' in case: row['expected_answer']=case['answer']
    tokens=[]; sampled=[]
    with Session(model,batch=1,chunk=4096,timeout=120) as session:
     session.reset();predicted=session.prefill([ids])
     for step in range(128):
      token=int(predicted[0]);tokens.append(token)
      if MODE=='float16' and step<TARGET_STEPS:
       logits=session.read_feature('logits').reshape(-1,model.config.vocab)[-1].copy()
       if not np.isfinite(logits).all():raise ValueError('Nonfinite F16 logits')
       sampled.append(logits)
      if token==tokenizer.eos or step==127:break
      predicted=session.decode(1)
    row.update(output_ids=tokens,output_text=tokenizer.decode(tokens,skip_special=True),eos=tokens[-1]==tokenizer.eos)
    if MODE=='float16':
     values=np.asarray(sampled)
     np.save(ROOT/(name+'-f16-logits.npy'),values)
     row['logits_sha256']=digest(ROOT/(name+'-f16-logits.npy'))
     row['teacher_steps']=len(values)
    else:
     old=next(c for c in reference['cases'] if c['name']==name)
     old_ids=old['output_ids']; count=0
     for left,right in zip(old_ids,tokens):
      if left!=right:break
      count+=1
     row['greedy_comparison']=dict(exact=tokens==old_ids,common_prefix_tokens=count,
      positional_token_agreement=sum(a==b for a,b in zip(old_ids,tokens))/max(len(old_ids),len(tokens)),
      f16_output_ids=old_ids,f16_output_text=old['output_text'],f16_eos=old['eos'])
     values=np.load(ROOT/(name+'-f16-logits.npy'),mmap_mode='r')
     if digest(ROOT/(name+'-f16-logits.npy'))!=old['logits_sha256']:raise ValueError('Changed F16 reference')
     metrics=[]
     retained=[]
     with Session(model,batch=1,chunk=4096,timeout=120) as session:
      session.reset();session.prefill([ids])
      for step,left in enumerate(values):
       right=session.read_feature('logits').reshape(-1,model.config.vocab)[-1].copy()
       if not np.isfinite(right).all():raise ValueError('Nonfinite FP8 logits')
       left,right=left.astype('float64'),right.astype('float64')
       def lsm(x):
        y=x-x.max();return y-np.log(np.exp(y).sum())
       lp,lq=lsm(left),lsm(right)
       lp_top=np.argsort(left)[-20:][::-1];lq_top=np.argsort(right)[-20:][::-1]
       metrics.append(dict(step=step,top1_equal=int(left.argmax())==int(right.argmax()),
        reference_token=int(old_ids[step]),kl_f16_to_fp8=float(np.dot(np.exp(lp),lp-lq)),
        reference_token_nll_delta=float(lp[old_ids[step]]-lq[old_ids[step]]),
        rms_logit_error=float(np.sqrt(np.mean((left-right)**2))),
        max_absolute_logit_error=float(np.max(np.abs(left-right))),
        f16_top20_ids=lp_top.tolist(),f16_top20_logits=left[lp_top].tolist(),
        fp8_top20_ids=lq_top.tolist(),fp8_top20_logits=right[lq_top].tolist()))
       if step in (0,len(values)//2,len(values)-1):retained.extend([left.astype('float32'),right.astype('float32')])
       if step+1<len(values):session.step([old_ids[step]])
     row['teacher_forced']=dict(steps=metrics,top1_agreement=float(np.mean([m['top1_equal'] for m in metrics])),
      mean_kl_f16_to_fp8=float(np.mean([m['kl_f16_to_fp8'] for m in metrics])),
      mean_reference_token_nll_delta=float(np.mean([m['reference_token_nll_delta'] for m in metrics])))
     np.savez_compressed(ROOT/(name+'-boundaries.npz'),logits=np.asarray(retained))
     rec.artifact(ROOT/(name+'-boundaries.npz'),'precision/'+name+'-boundaries.npz')
    output['cases'].append(row);write(ROOT/(MODE+'.json'),output)
    summary={k:row[k] for k in ('name','prefix','eos','output_text','greedy_comparison') if k in row}
    if MODE!='float16':summary['teacher_top1']=row['teacher_forced']['top1_agreement'];summary['mean_KL']=row['teacher_forced']['mean_kl_f16_to_fp8']
    print('OUTPUT',json.dumps(summary),flush=True)
 meta['checks']=dict(finite=True,cases=len(output['cases']))
 if reference:
  comparisons=[r['greedy_comparison'] for r in output['cases']]
  meta['comparison_summary']=dict(exact_outputs=sum(c['exact'] for c in comparisons),cases=len(comparisons),
   common_prefix_tokens=[c['common_prefix_tokens'] for c in comparisons],
   teacher_top1_agreement=float(np.mean([m['top1_equal'] for c in output['cases'] for m in c['teacher_forced']['steps']])),
   mean_kl_f16_to_fp8=float(np.mean([m['kl_f16_to_fp8'] for c in output['cases'] for m in c['teacher_forced']['steps']])))
  print('COMPARISON',json.dumps(meta['comparison_summary']),flush=True)
 passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 write(ROOT/(MODE+'.json'),output)
 rec.artifact(__file__,'precision/runner.py');rec.artifact(ROOT/(MODE+'.json'),'precision/outputs.json')
 rec.update(report,final=True,status='passed' if passed else 'failed')
 print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
