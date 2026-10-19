"""Qualify representative expert/attention geometries through the public compiler."""
import json
from pathlib import Path
import resource
import sys
import time
import tensor
from akas.graph import Graph
from akas.compilation.compile import produce
from akas.compilation.tuning import SearchSettings
from akas.models.lfm2.model import LFM2MoE
provider=sys.argv[1]
target='sm_89' if provider=='cuda' else 'webgpu-portable-v1'
checkpoint=LFM2MoE.checkpoint_type('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf',tokenizer=False)
entries=[]
started=time.monotonic()
with tensor.Device(provider=provider,max_buffer_size=268435456 if provider=='webgpu' else None) as device:
 for chunk in (32,256,512):
  full=LFM2MoE.compose(checkpoint,rows=1,chunk=chunk,max_requests=1,max_context=131072,provider=provider,**dict(LFM2MoE.model_defaults)).graph
  graph=Graph(contract=full.contract,phase='prefill')
  required=set()
  for call in full.calls:
   if call.kind not in ('attention','quantized_experts') or call.key in required:
    continue
   required.add(call.key)
   for name in call.arguments:
    if name not in graph.buffers:
     b=full.buffers[name]
     graph.buffer(name,b.shape,b.dtype,role=b.role)
   graph(call.kind,call.parameters,*call.arguments)
  out=Path(f'build/lfm-moe/{provider}-seed-{chunk}')
  manifest=produce(graph,out,provider=provider,target=target,device=device,search_settings=SearchSettings(budget=16,width=4,warmup=1,samples=3,repeats=2))
  entries.extend(manifest['tuning']['profile']['entries'])
  print(json.dumps(dict(provider=provider,chunk=chunk,seconds=time.monotonic()-started,entries=manifest['tuning']['profile']['entries'],cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)),flush=True)
Path(f'build/lfm-moe/{provider}-seed-profile.json').write_text(json.dumps(dict(schema='tensor.schedule-profile.v1',provider=provider,target=target,entries=entries),indent=2)+'\n')
