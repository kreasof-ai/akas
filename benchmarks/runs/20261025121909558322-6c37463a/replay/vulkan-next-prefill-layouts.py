import sys
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from akas.compilation.emit import build_artifact
from akas.kernels.registry import validation
from akas.execution.plan import ExecutionPlan,call_bindings,bind_named
from akas.models.lfm2.model import LFM2
from akas.memory.transfer import write as transfer
from tensor.compiler.tuning import measure
from tools.context_sweep import vulkan_device
from akas.benchmarking.history import write
from tensor.artifacts.format import read_artifact
root=Path('build/vulkan-next/prefill-layouts');root.mkdir(parents=True,exist_ok=True)
with vulkan_device() as device, LFM2('build/checkpoints/lfm2.5-2.6b/q4_0.gguf','build/lfm2.5-2.6b-vulkan/next-mixed-input',device,graphs=True,checkpoint_options={'tokenizer':False}) as model:
 slot=model.admit();frame=model.batches.select(1,512)
 frame.prepare([slot],[[1]*512],[0]);transfer(device,frame.buffers['positions'],np.array([0],'int32'));frame.forward.run();device.synchronize()
 call,entry=next((c,e) for c,e in zip(frame.forward.definition.calls,frame.forward.plan) if c.kind=='ffn' and c.parameters['r']==512)
 named=call_bindings(entry[0],entry[1]);results=[]
 for index,tile in enumerate([(64,64,4,4,256),(128,64,4,4,512),(64,128,4,4,512),(128,128,4,4,1024),(128,64,8,4,256)]):
  p={**call.parameters,'half_tile':tile,'half_pad':1,'half_k':16}
  path=root/str(index);path.mkdir(exist_ok=True);src,artifact=path/'kernel.py',path/'kernel.tbin'
  src.write_text('from akas.ops.linear.webgpu import native_half_pairs\ndef tensor_export():\n    return {"kernel": native_half_pairs('+repr(p)+', paired=True)}\n')
  if not artifact.exists():build_artifact(src,artifact,provider='webgpu',target='webgpu-portable-v1')
  meta,_=read_artifact(artifact);case=validation(call.kind,p,meta['arguments'])
  with ExitStack() as stack:
   arrays=[]
   for arg in meta['arguments']:
    buf=device.from_numpy(case.arguments[arg['name']]);stack.callback(buf.release);arrays.append(buf)
   kernel=stack.enter_context(device.load(artifact));kernel.launch(*arrays)
   for arg,buf in zip(meta['arguments'],arrays):
    actual=buf.to_numpy();expected=case.expected[arg['name']]
    if arg['name'] in case.comparators:case.comparators[arg['name']](actual,expected)
    else:np.testing.assert_allclose(actual,expected,rtol=case.rtol,atol=case.atol)
  del arrays,case
  with device.load(artifact) as kernel:
   replacement=bind_named(device,kernel,named);plan=ExecutionPlan(device,[replacement]*16)
   prepared=plan.capture(resources=tuple(frame.buffers.values())+(kernel,))
   try:timing=measure(device,prepared.launch,warmup=2,samples=5,repeats=1)
   finally:prepared.close()
   row=dict(tile=tile,seconds=timing['median_seconds']/16,timing=timing,scope='qualified isolated hot-buffer diagnostic; not end-to-end throughput')
   results.append(row);write(root/'results.json',results);print(tile,row['seconds'],flush=True)
