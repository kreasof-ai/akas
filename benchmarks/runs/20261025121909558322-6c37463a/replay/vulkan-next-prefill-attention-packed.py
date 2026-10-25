import sys,json
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from akas.compilation.selection import source
from akas.compilation.emit import build_artifact
from akas.kernels.registry import validation
from akas.execution.plan import ExecutionPlan,call_bindings,bind_named
from akas.models.lfm2.model import LFM2
from akas.memory.transfer import write as transfer
from tensor.compiler.tuning import measure
from tools.context_sweep import vulkan_device
from akas.benchmarking.history import write
root=Path('build/vulkan-next/prefill-attention-packed');root.mkdir(parents=True,exist_ok=True)
program=Path('build/lfm2.5-2.6b-vulkan/next-mixed-input')
with vulkan_device() as device, LFM2('build/checkpoints/lfm2.5-2.6b/q4_0.gguf',program,device,graphs=True,checkpoint_options={'tokenizer':False}) as model:
 slot=model.admit();frame=model.batches.select(1,512)
 frame.prepare([slot],[[1]*512],[32768]);transfer(device,frame.buffers['positions'],np.array([32768],'int32'))
 frame.forward.run();device.synchronize()
 call,entry=next((c,e) for c,e in zip(frame.forward.definition.calls,frame.forward.plan) if c.kind=='attention' and c.parameters.get('r')==512)
 named=call_bindings(entry[0],entry[1]);results=[];case=None
 for rows in (4,8):
  for granule in (16,32):
   p={**call.parameters,'block_q':rows,'block_k':granule,'key_pad':0,'shared_dtype':'float16','dot_width':4,'packed_shared':True}
   path=root/f'r{rows}-g{granule}';path.mkdir(exist_ok=True)
   src,artifact=path/'kernel.py',path/'kernel.tbin'
   src.write_text(source(call.kind,p,contract='reference',provider='webgpu',target='webgpu-portable-v1',preferred='attention.webgpu.tiled_prefill').render())
   if not artifact.exists(): build_artifact(src,artifact,provider='webgpu',target='webgpu-portable-v1')
   from tensor.artifacts.format import read_artifact
   meta,_=read_artifact(artifact)
   if case is None:case=validation(call.kind,p,meta['arguments'])
   with ExitStack() as stack:
    arrays=[]
    for arg in meta['arguments']:
     buf=device.from_numpy(case.arguments[arg['name']]);stack.callback(buf.release);arrays.append(buf)
    kernel=stack.enter_context(device.load(artifact));kernel.launch(*arrays)
    for arg,buf in zip(meta['arguments'],arrays):
     np.testing.assert_allclose(buf.to_numpy(),case.expected[arg['name']],rtol=case.rtol,atol=case.atol)
   del arrays
   with device.load(artifact) as kernel:
    replacement=bind_named(device,kernel,named)
    plan=ExecutionPlan(device,[replacement]*4)
    prepared=plan.capture(resources=tuple(frame.buffers.values())+(kernel,))
    try:
     timing=measure(device,prepared.launch,warmup=1,samples=3,repeats=1)
    finally:
     prepared.close()
    row=dict(key_rows=rows,granule=granule,position=32768,scope='isolated repeated hot-buffer attention; diagnostic only',seconds=timing['median_seconds']/4,timing=timing)
    results.append(row);write(root/'results.json',results);print(row['key_rows'],row['granule'],row['seconds'],flush=True)
