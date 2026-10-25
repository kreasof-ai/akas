"""Numerical probe; fixed prose is not a corpus accuracy/perplexity evaluation."""
import hashlib,json
import numpy as np
from akas.benchmarking.session import Session

TEXTS=[
 'A gardener keeps a notebook for each season. In spring she plants peas and lettuce, checks the soil each morning, and records how much rain has fallen. She waters only when the soil is dry. In summer she harvests the vegetables and saves seeds for the following year. Explain how her notes help her plan the next garden.\n',
 'The community library holds science workshops on Saturday. Visitors build small bridges from paper, measure the load each bridge can support, and compare the results. The librarian asks everyone to change only one feature at a time so they can explain the difference between two designs. Describe a fair experiment for testing these bridges.\n',
 'A computer program stores a collection of books. Each book has a title, an author, and a publication year. A reader can search the collection, borrow an available book, and return it later. The program must preserve the catalogue while it updates the borrowing records. Explain how to organize these records clearly.\n']

def collect(executor,tokenizer,*,chunk,reference=None):
 saved={};checks=[]
 for index,prefix in enumerate((32,8192,130944)):
  ids=np.resize(np.asarray(tokenizer.encode(TEXTS[index]),dtype='int32'),prefix).tolist();key=f'p{prefix}'
  logits=[];tokens=[]
  with Session(executor,batch=1,chunk=chunk,timeout=120) as session:
   session.reset();session.prefill([ids])
   for step in range(9):
    values=session.read_feature('logits').reshape(-1,executor.config.vocab)[-1].copy()
    if not np.isfinite(values).all():raise ValueError('nonfinite precision probe logits')
    logits.append(values)
    if step<8:
     token=int(values.argmax()) if reference is None else int(reference[key+'-tokens'][step])
     tokens.append(token);session.step([token])
  logits=np.asarray(logits);tokens=np.asarray(tokens,dtype='int32');saved[key+'-logits']=logits;saved[key+'-tokens']=tokens
  check=dict(prefix=prefix,teacher_forced_steps=8,prompt_sha256=hashlib.sha256(np.asarray(ids,dtype='int32').tobytes()).hexdigest(),reference_continuation=tokenizer.decode(tokens.tolist()))
  if reference is not None:
   left=reference[key+'-logits'].astype('float64');right=logits.astype('float64')
   def logsoftmax(x):
    x=x-x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
   lp,lq=logsoftmax(left),logsoftmax(right);kl=(np.exp(lp)*(lp-lq)).sum(-1)
   check.update(top1_agreement=float(np.mean(left.argmax(-1)==right.argmax(-1))),relative_logit_error=float(np.linalg.norm(left-right)/max(np.linalg.norm(left),1e-30)),mean_kl_f16_to_fp8=float(kl.mean()),max_kl_f16_to_fp8=float(kl.max()),mean_reference_token_nll_delta=float(np.mean(lp[np.arange(8),tokens]-lq[np.arange(8),tokens])))
  checks.append(check);print('QUALITY',json.dumps(check),flush=True)
 return saved,checks
