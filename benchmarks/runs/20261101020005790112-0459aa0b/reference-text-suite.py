from pathlib import Path
import json,time,hashlib
import numpy as np
from akas.models.lfm2.weights import LFM2MoECheckpoint
from support.native_fp8_reference import NativeFP8Reference
root=Path('build/lfm-serving')
checkpoint=LFM2MoECheckpoint('build/checkpoints/lfm2.5-24b-a2b-fp8')
texts=['The capital of France is','The opposite of hot is','The first three prime numbers are','def add(a, b):\n    return','Water freezes at zero degrees','A triangle has three','The largest planet in our solar system is','To calculate the area of a circle, use']
arrays={};rows=[];started=time.monotonic()
for index,text in enumerate(texts):
 tokens=checkpoint.tokenizer.encode(text)
 oracle=NativeFP8Reference(checkpoint,projection_half=True)
 logits=oracle.forward(tokens)
 predicted=int(logits.argmax())
 arrays[f'tokens{index}']=np.asarray(tokens,dtype='int32');arrays[f'logits{index}']=logits
 row=dict(index=index,text=text,tokens=tokens,prediction=predicted,predicted_text=checkpoint.tokenizer.decode([predicted]))
 rows.append(row)
 print(row,'elapsed',time.monotonic()-started,flush=True)
 np.savez(root/'reference-text-suite.npz',**arrays)
 (root/'reference-text-suite.json').write_text(json.dumps(dict(model_revision='f54bf39547fd5b7955abb9fe8d1b151e31fbbddb',oracle_source_sha256=hashlib.sha256(Path('tests/support/native_fp8_reference.py').read_bytes()).hexdigest(),projection_half=True,rows=rows),indent=2)+'\n')
