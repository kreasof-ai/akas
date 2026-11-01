"""Independent original-weight probes with the selected storage boundaries."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from akas.models.lfm2.weights import LFM2MoECheckpoint
from support.native_fp8_reference import NativeFP8Reference
from akas.benchmarking.history import Recorder

main=Path('/home/sagemaker-user/akas');root=main/'build/lfm-serving/precision-matched-reference';root.mkdir(parents=True,exist_ok=True)
precision=dict(projection_half=True,contribution_dtype='bfloat16',activation_dtype='bfloat16',
               residual_dtype='float32',attention_half=True,attention_decode_half=True)
metadata=dict(runtime='numpy',purpose='numerical-reference',provider='cpu',
 model='LiquidAI/LFM2.5-24B-A2B-FP8',model_revision='f54bf39547fd5b7955abb9fe8d1b151e31fbbddb',
 precision=precision,qualification_scope='Independent F64 products and declared activation/storage boundaries for eight natural prompts; not a serving or hardware qualification')
record=Recorder(main/'benchmarks',metadata,source_root=main,kind='qualification',
 change='Generate separate natural-text references matching BF16 expert intermediates and FP16 attention operands',
 dependency_sources={'tensor':main.parent/'tensor'})
record.artifact(Path(__file__),'producer.py');print('STARTED',record.id,flush=True)
try:
 checkpoint=LFM2MoECheckpoint(main/'build/checkpoints/lfm2.5-24b-a2b-fp8')
 original=json.loads((main/'build/lfm-serving/reference-text-suite.json').read_text())
 record.artifact(main/'build/lfm-serving/reference-text-suite.json','original-inputs.json')
 arrays={};rows=[];started=time.monotonic()
 for item in original['rows']:
  index=item['index'];tokens=item['tokens']
  oracle=NativeFP8Reference(checkpoint,**precision);logits=oracle.forward(tokens)
  predicted=int(logits.argmax());arrays[f'tokens{index}']=np.asarray(tokens,dtype='int32');arrays[f'logits{index}']=logits
  row=dict(index=index,text=item['text'],tokens=tokens,prediction=predicted,predicted_text=checkpoint.tokenizer.decode([predicted]));rows.append(row)
  np.savez(root/'reference.npz',**arrays)
  (root/'reference.json').write_text(json.dumps(dict(model_revision=metadata['model_revision'],precision=precision,
      oracle_source_sha256=hashlib.sha256((main/'tests/support/native_fp8_reference.py').read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
  print('REFERENCE',row,'elapsed',time.monotonic()-started,flush=True)
 for suffix in ('json','npz'):record.artifact(root/f'reference.{suffix}',f'reference.{suffix}')
 record.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',record.id,flush=True)
except BaseException as error:
 if record.record['completed_at'] is None:record.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise
