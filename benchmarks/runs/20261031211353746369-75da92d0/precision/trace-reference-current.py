from pathlib import Path
import hashlib,json,time
import numpy as np
from akas.models.lfm2.weights import LFM2MoECheckpoint
from support.native_fp8_reference import NativeFP8Reference
root=Path('build/lfm-serving');tokens=[597,5205,302,3980,355]
cp=LFM2MoECheckpoint('build/checkpoints/lfm2.5-24b-a2b-fp8',tokenizer=False)
oracle=NativeFP8Reference(cp,projection_half=True)
arrays={};start=time.monotonic()
def observer(index,values):
 arrays[f'layer{index}']=values.copy()
 np.savez(root/'reference-current-trace.npz',**arrays)
 print('LAYER',index,'elapsed',time.monotonic()-start,flush=True)
logits=oracle.forward(tokens,observer=observer)
arrays['logits']=logits;arrays['tokens']=np.asarray(tokens)
np.savez(root/'reference-current-trace.npz',**arrays)
(root/'reference-current-trace.json').write_text(json.dumps(dict(tokens=tokens,prediction=int(logits.argmax()),oracle_source_sha256=hashlib.sha256(Path('tests/support/native_fp8_reference.py').read_bytes()).hexdigest(),projection_half=True,seconds=time.monotonic()-start),indent=2)+'\n')
print('REFERENCE',int(logits.argmax()),flush=True)
