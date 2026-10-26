from pathlib import Path
import json
import tensor
import numpy as np
from akas.models.spark import Spark,SparkCheckpoint
from akas.compilation import compile_checkpoint
from akas.compilation.tuning import SearchSettings
from akas.benchmarking.history import write
path=Path('build/checkpoints/spark-x2.6-4b/fp8');out=Path('build/spark-packed/program-fp8')
compile_checkpoint(Spark,path,out,checkpoint_options={'tokenizer':False},geometries=[(1,1),(1,32)],max_requests=1,max_context=128,model_options=dict(prefill_chunk=32,rotary_cache=True,optimized_attention=True),search_settings=SearchSettings(budget=1,samples=1,repeats=1))
c=SparkCheckpoint(path,tokenizer=False)
with tensor.Device() as d,Spark(path,out,d,checkpoint_options={'tokenizer':False}) as model:
 for name,hf in c.names.items():
  if hf in c.scales:
   assert str(model.batches.weights[name].dtype)=='uint8'
   assert str(model.batches.weights[name+'.scale'].dtype)=='bfloat16'
 slot=model.admit();token=model.prefill([slot],[[1,2,3,4]],chunk=32)[0];ids=[int(token)]
 for _ in range(7):ids.append(int(model.decode([slot],[ids[-1]])[0]))
 logits=model.batches.select(1,1).feature('logits').to_numpy();assert np.isfinite(logits).all()
 write('build/spark-packed/fp8-smoke.json',dict(status='passed',inventory=c.inventory(),allocated_weight_bytes=sum(v.nbytes for v in model.batches.weights.values()),output_ids=ids,finite_logits=True))
 print('FP8 SMOKE PASSED',ids,flush=True)
