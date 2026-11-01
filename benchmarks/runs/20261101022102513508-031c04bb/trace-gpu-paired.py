from pathlib import Path
import argparse,json,re,time
from akas.benchmarking.history import Recorder,digest
import numpy as np
import tensor
from akas.models.lfm2 import LFM2MoE
from akas.kernels.registry import graph_outputs
root=Path('build/lfm-serving');tokens=[597,5205,302,3980,355];arrays={};steps=[]
parser=argparse.ArgumentParser();parser.add_argument('--chunk',type=int,default=2048);args=parser.parse_args()
metadata=dict(runtime='akas',purpose='numerical-trace',provider='cuda',program_sha256=digest(root/'program-concurrent-fused/program.json'),qualification_scope='Untimed layer trace against retained independent reference; no numerical acceptance or throughput claim')
rec=Recorder('benchmarks',metadata,kind='diagnostic',change='Locate the first divergence in natural-text FP8 serving',dependency_sources={'tensor':Path('../tensor')})
rec.artifact(Path(__file__),'trace-gpu-paired.py');print('STARTED',rec.id,flush=True)
started=time.monotonic()
with tensor.Device() as device, LFM2MoE('build/checkpoints/lfm2.5-24b-a2b-fp8',root/'program-concurrent-fused',device,graphs=False,checkpoint_options={'tokenizer':False}) as model:
 print('LOADED',time.monotonic()-started,flush=True)
 slot=model.admit();frame=model.batches.select(1,args.chunk)
 frame.prepare([slot],[tokens],model.positions)
 graph=frame.forward.definition
 def root_name(name):
  while graph.buffers[name].alias_of is not None:name=graph.buffers[name].alias_of
  return name
 for index,(call,entry) in enumerate(zip(graph.calls,frame.forward.plan)):
  device._launch(entry[0],entry[1])
  written=graph_outputs(call,contract=graph.contract) or ()
  for name in written:
   base=root_name(name);match=re.search(r'blk\.(\d+)\.',base)
   layer=int(match[1]) if match else -1
   if graph.buffers[name].role=='state':continue
   if layer>=4 and not re.fullmatch(r'blk\.\d+\.output',base):continue
   value=frame.buffers[name];shape=value.shape
   if shape[0]==args.chunk:logical=(len(tokens),*shape[1:])
   elif len(shape)==1 and shape[0]%args.chunk==0:logical=(len(tokens)*shape[0]//args.chunk,)
   elif value.nbytes<=2*1024**2:logical=shape
   else:continue
   view=value.prefix(logical)
   try:arrays[base]=view.to_numpy().copy()
   finally:view.release()
   steps.append(dict(index=index,kind=call.kind,name=base,parameters=call.parameters,shape=list(logical)))
   if re.fullmatch(r'blk\.\d+\.output',base):print('LAYER',base,flush=True)
 np.savez(root/'gpu-paired-trace.npz',**arrays)
 logits=frame.feature('logits').to_numpy().reshape(-1).copy()
 print('MANUAL',int(logits.argmax()),flush=True)
 model.release(slot)
 slot=model.admit()
 prediction=model.prefill([slot],[tokens],chunk=args.chunk)[0]
 actual_frame=model.last_frame
 public=actual_frame.feature('logits').to_numpy().reshape(-1)
 print('PUBLIC',prediction,'DIFFERENT',int(np.count_nonzero(logits.view('uint32')!=public.view('uint32'))),flush=True)
 arrays['logits']=logits;arrays['public_logits']=public
 np.savez(root/'gpu-paired-trace.npz',**arrays)
 (root/'gpu-paired-trace.json').write_text(json.dumps(dict(tokens=tokens,manual_prediction=int(logits.argmax()),public_prediction=prediction,steps=steps),indent=2)+'\n')
 model.release(slot)
for filename in ('gpu-paired-trace.json','gpu-paired-trace.npz','reference-current-trace.json','reference-current-trace.npz'):
 rec.artifact(root/filename,filename)
rec.program(root/'program-concurrent-fused')
rec.update(dict(metadata=metadata,results=[]),final=True,status='passed')
print('RUN',rec.id,flush=True)
