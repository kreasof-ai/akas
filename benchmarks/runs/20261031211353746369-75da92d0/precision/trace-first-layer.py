from pathlib import Path
import numpy as np
from akas.models.lfm2.weights import LFM2MoECheckpoint
from support.native_fp8_reference import NativeFP8Reference
root=Path('build/lfm-serving');arrays={}
class Oracle(NativeFP8Reference):
 def project(self,name,values,*,single=False):
  actual=super().project(name,values,single=single)
  arrays[name+'.input']=values.copy();arrays[name+'.output']=actual.copy()
  return actual
class Complete(Exception):pass
cp=LFM2MoECheckpoint('build/checkpoints/lfm2.5-24b-a2b-fp8',tokenizer=False)
oracle=Oracle(cp,projection_half=True)
def observer(index,values):
 arrays['layer0']=values.copy();raise Complete
try:oracle.forward([597,5205,302,3980,355],observer=observer)
except Complete:pass
np.savez(root/'first-layer-reference.npz',**arrays)
g=np.load(root/'gpu-current-trace.npz')
comparisons=[
 ('normalized','shortconv.in_proj.weight.input'),('projection','shortconv.in_proj.weight.output'),
 ('mixed','shortconv.out_proj.weight.input'),('operator_output','shortconv.out_proj.weight.output'),
 ('ffn_normalized','ffn_gate.weight.input'),('gate','ffn_gate.weight.output'),('up','ffn_up.weight.output'),
 ('activated','ffn_down.weight.input'),('ffn_output','ffn_down.weight.output')]
for left,right in comparisons:
 actual=g['blk.0.'+left].reshape(arrays['blk.0.'+right].shape);expected=arrays['blk.0.'+right]
 print(left,'max',float(np.max(np.abs(actual-expected))),'relative_rms',float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),flush=True)
print('GPU FIELDS',[n for n in g.files if 'blk.0.' in n])
