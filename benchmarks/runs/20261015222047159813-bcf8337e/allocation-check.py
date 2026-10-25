import json, math, subprocess, resource, sys, traceback
from pathlib import Path
import tensor
from akas.optimization.context import discover,symbol
from akas.optimization.planning import memory_bound
from akas.compilation.buffers import scratch_slots
from akas.benchmarking.history import write,utc,source_identity
from tensor.runtime.dtypes import dtype
name,provider,kv,window=sys.argv[1:]
window=int(window)
path=Path('build/checkpoints')/name/'bf16'; model=symbol(discover(path)); loaded=model.checkpoint_type(path)
options={**model.model_defaults,'kv_dtype':kv}; batch=256;limit=2176+window
geometries=[(256,1,None),(1,32,None),(32,32,None)]
if window: geometries += [(256,window,'verify'),(256,1,'proposal'),(1,32,'proposal'),(32,32,'proposal')]
graphs=[model.compose(loaded,rows=b,chunk=t,max_requests=batch,max_context=limit,provider=provider,phase=phase,**options).graph for b,t,phase in geometries]
persistent={};private=[]
for g in graphs:
 for n,s in g.buffers.items():
  if s.alias_of is None and s.role in ('weight','state'):
   if n not in persistent or math.prod(s.shape)*dtype(s.dtype).itemsize>math.prod(persistent[n].shape)*dtype(persistent[n].dtype).itemsize:persistent[n]=s
 private += [g.buffers[names[0]] for names in scratch_slots(g)]
 private += [s for s in g.buffers.values() if s.alias_of is None and s.role not in ('weight','state','scratch')]
specs=[*persistent.values(),*private]
need=max(math.prod(s.shape)*dtype(s.dtype).itemsize for s in specs)
result=dict(source=source_identity(Path.cwd()),model=name,provider=provider,kv_dtype=kv,window=window,batch=256,prefix=2048,output=128,created_at=utc(),bound=memory_bound(graphs),scope='Actual allocation of graph weights, cache/state and all selected scratch frames; no model execution')
try:
 with tensor.Device(0,provider=provider,max_buffer_size=need if provider=='webgpu' else None) as device:
  buffers=[]
  for s in specs: buffers.append(device.zeros(s.shape,s.dtype) if provider=="cuda" else device.empty(s.shape,s.dtype))
  if provider=="webgpu":
   encoder=device._gpu.create_command_encoder()
   for b in buffers: encoder.clear_buffer(b._storage)
   device._gpu.queue.submit([encoder.finish()])
  device.synchronize()
  result.update(status='passed',device=device.info,allocated_bytes=sum(b.nbytes for b in buffers),sampled_device_bytes=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())*1024**2,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  for b in reversed(buffers):b.release()
except Exception as e:result.update(status='failed',error=str(e),traceback=traceback.format_exc())
out=Path('build/optimization-pilots')/f'allocation-{name}-{provider}-{kv}-{window}.json';write(out,result);print(json.dumps(result,indent=2))
