"""Paired cache-precision diagnostics on fixed prose and identical teacher-forced tokens.
This is a numerical precision probe, not a corpus quality/perplexity evaluation.
"""
import json,sys,traceback,hashlib
from pathlib import Path
import numpy as np
import tensor
from akas.models.lfm2.model import LFM2MoE
from akas.benchmarking.session import Session
from akas.benchmarking.history import Recorder,write
mode=sys.argv[1];program=Path(sys.argv[2]);root=Path('build/lfm-moe/kv-quality');root.mkdir(exist_ok=True)
checkpoint=Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf')
chunk=int(sys.argv[3])
texts=[
 'A gardener keeps a notebook for each season. In spring she plants peas and lettuce, checks the soil each morning, and records how much rain has fallen. She waters only when the soil is dry. In summer she harvests the vegetables and saves seeds for the following year. Explain how her notes help her plan the next garden.\n',
 'The community library holds science workshops on Saturday. Visitors build small bridges from paper, measure the load each bridge can support, and compare the results. The librarian asks everyone to change only one feature at a time so they can explain the difference between two designs. Describe a fair experiment for testing these bridges.\n',
 'A computer program stores a collection of books. Each book has a title, an author, and a publication year. A reader can search the collection, borrow an available book, and return it later. The program must preserve the catalogue while it updates the borrowing records. Explain how to organize these records clearly.\n']
meta=dict(runtime='akas',purpose='kernel',protocol='kv-precision-probe-v1',provider='cuda',device=dict(name='NVIDIA L4',provider='cuda'),kv_dtype=mode,chunk=chunk,model='akas.models.lfm2.model:LFM2MoE',checkpoint=str(checkpoint),checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',qualification='Fixed prose repeated to exact prefix lengths; eight identical F16-generated teacher-forced continuation tokens; distribution drift and top-1 agreement, not a corpus accuracy claim')
rec=Recorder('benchmarks',meta,kind='diagnostic',change='Measure KV cache precision separately from implementation correctness')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False;saved={};checks=[]
try:
 checkpoint_data=LFM2MoE.checkpoint_type(checkpoint)
 tokenizer=checkpoint_data.tokenizer
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(checkpoint,program,device,graphs=True) as executor:
   reference=np.load(root/'float16.npz',allow_pickle=False) if mode!='float16' else None
   for index,prefix in enumerate((32,8192,130944)):
    ids=np.resize(np.asarray(tokenizer.encode(texts[index]),dtype='int32'),prefix).tolist();key=f'p{prefix}'
    logits=[];tokens=[]
    with Session(executor,batch=1,chunk=chunk,timeout=120) as session:
     session.reset();session.prefill([ids])
     for step in range(9):
      values=session.read_feature('logits').reshape(-1,executor.config.vocab)[-1].copy()
      if not np.isfinite(values).all():raise ValueError('nonfinite logits')
      logits.append(values)
      if step<8:
       token=int(values.argmax()) if reference is None else int(reference[key+'-tokens'][step])
       tokens.append(token);session.step([token])
    logits=np.asarray(logits);tokens=np.asarray(tokens,dtype='int32');saved[key+'-logits']=logits;saved[key+'-tokens']=tokens
    check=dict(prefix=prefix,teacher_forced_steps=8,prompt_sha256=hashlib.sha256(np.asarray(ids,dtype='int32').tobytes()).hexdigest(),output_text=tokenizer.decode(tokens.tolist()))
    if reference is not None:
     left=reference[key+'-logits'].astype('float64');right=logits.astype('float64')
     def logsoftmax(x):
      x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
     lp,lq=logsoftmax(left),logsoftmax(right);kl=(np.exp(lp)*(lp-lq)).sum(-1)
     check.update(top1_agreement=float(np.mean(left.argmax(-1)==right.argmax(-1))),relative_logit_error=float(np.linalg.norm(left-right)/max(np.linalg.norm(left),1e-30)),mean_kl_f16_to_fp8=float(kl.mean()),max_kl_f16_to_fp8=float(kl.max()),mean_reference_token_nll_delta=float(np.mean(lq[np.arange(8),tokens]-lp[np.arange(8),tokens])*-1))
    checks.append(check);print('QUALITY',json.dumps(check),flush=True)
   if reference is not None:reference.close()
 meta['checks']=checks;np.savez_compressed(root/(mode+'.npz'),**saved);write(root/(mode+'.json'),meta);passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 rec.artifact(__file__,'precision/runner.py')
 if (root/(mode+'.npz')).exists():rec.artifact(root/(mode+'.npz'),'precision/logits.npz')
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
