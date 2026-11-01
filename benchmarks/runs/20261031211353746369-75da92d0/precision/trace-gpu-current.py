from pathlib import Path
import json,re,time
import numpy as np
import tensor
from akas.models.lfm2 import LFM2MoE
from akas.kernels.registry import graph_outputs
root=Path('build/lfm-serving');tokens=[597,5205,302,3980,355];arrays={};steps=[]
started=time.monotonic()
with tensor.Device() as device, LFM2MoE('build/checkpoints/lfm2.5-24b-a2b-fp8',root/'program-concurrent',device,graphs=False,checkpoint_options={'tokenizer':False}) as model:
 print('LOADED',time.monotonic()-started,flush=True)
 slot=model.admit();frame=model.batches.select(1,2048)
 frame.prepare([slot],[tokens],model.positions)
 graph=frame.forward.definition
 def root_name(name):
  while graph.buffers[name].alias_of is not None:name=graph.buffers[name].alias_of
  return name
 for index,(call,entry) in enumerate(zip(graph.calls,frame.forward.plan)):
  entry[1].launch()
  written=graph_outputs(call,contract=graph.contract) or ()
  for name in written:
   base=root_name(name);match=re.search(r'blk\.(\d+)\.',base)
   layer=int(match[1]) if match else -1
   if graph.buffers[name].role=='state':continue
   if layer>=4 and not re.fullmatch(r'blk\.\d+\.output',base):continue
   value=frame.buffers[name];shape=value.shape
   if shape[0]==2048:logical=(len(tokens),*shape[1:])
   elif len(shape)==1 and shape[0]%2048==0:logical=(len(tokens)*shape[0]//2048,)
   elif value.nbytes<=2*1024**2:logical=shape
   else:continue
   view=value.prefix(logical)
   try:arrays[base]=view.to_numpy().copy()
   finally:view.release()
   steps.append(dict(index=index,kind=call.kind,name=base,parameters=call.parameters,shape=list(logical)))
   if re.fullmatch(r'blk\.\d+\.output',base):print('LAYER',base,flush=True)
 np.savez(root/'gpu-current-trace.npz',**arrays)
 logits=frame.feature('logits').to_numpy().reshape(-1).copy()
 print('MANUAL',int(logits.argmax()),flush=True)
 model.release(slot)
 frame.forward.graph=frame.forward.plan.capture(resources=tuple(frame.buffers.values())+tuple(frame.kernels.values()))
 slot=model.admit()
 prediction=model.prefill([slot],[tokens],chunk=2048)[0]
 captured=frame.feature('logits').to_numpy().reshape(-1)
 print('CAPTURED',prediction,'DIFFERENT',int(np.count_nonzero(logits.view('uint32')!=captured.view('uint32'))),flush=True)
 arrays['logits']=logits;arrays['captured_logits']=captured
 np.savez(root/'gpu-current-trace.npz',**arrays)
 (root/'gpu-current-trace.json').write_text(json.dumps(dict(tokens=tokens,manual_prediction=int(logits.argmax()),captured_prediction=prediction,steps=steps),indent=2)+'\n')
 model.release(slot)
