import json
from pathlib import Path
import numpy as np,tensor
from akas.models.spark import Spark
from akas.benchmarking.session import Session
from tools.spark_benchmark import prompts
from akas.benchmarking.history import write
p=Path('build/spark-parity/compact-program');program=json.loads((p/'program.json').read_text())
for phase in program['phases']:
 if phase['phase']=='prefill_state' and phase['chunk']>1:
  graph=json.loads((p/phase['path']).read_text())['forward']
  assert sum(c['kind']=='window_attention' for c in graph['calls'])==8
with tensor.Device() as device,Spark('build/checkpoints/spark-x2.6-4b/gguf/Spark-X2.6-4B-Q4_K_M.gguf',p,device,graphs=True,checkpoint_options={'tokenizer':False}) as model:
 for phase in program['phases']:model.batches.select(1,phase['chunk'],phase=phase['phase'])
 original=model.batches.records;results=[]
 for optimized in (True,False):
  model.batches.records=original if optimized else tuple(r for r in original if r['phase']!='prefill_state')
  with Session(model,batch=1,chunk=2048) as session:
   session.reset();predicted=session.prefill(prompts(8192));logits=session.read_feature('logits').copy();results.append((predicted,logits))
 np.testing.assert_array_equal(results[0][1],results[1][1]);assert results[0][0]==results[1][0]
 write('build/spark-parity/pruning-parity.json',dict(status='passed',prefix=8192,logits_compared=int(results[0][1].size),bitwise_equal=True,full_attention_calls_per_intermediate_phase=8,scope='Same native kernels: pruned intermediate prefill versus complete forward at every chunk; not an independent pretrained-model reference'))
 print('passed: all 131072 logits bitwise equal with state prefill enabled or disabled',flush=True)
