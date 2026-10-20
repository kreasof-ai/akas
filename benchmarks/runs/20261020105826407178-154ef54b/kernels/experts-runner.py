import json
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.compilation.emit import build_artifact
from akas.compilation.selection import source
from akas.kernels.registry import validation
from akas.execution.plan import ExecutionPlan
root=Path('build/lfm-moe/moe-decode-explore');root.mkdir(exist_ok=True)
for depth,columns,routed in ((2048,1536,False),(1536,2048,True)):
 p=dict(r=1,top=4,experts=64,k=depth,o=columns,input_dtype='float32',routed_input=routed,round_inputs=True,weight_parts=({'rows':64*columns,'format':'ggml','options':{'encoding':2}},))
 paths=[]
 for width,threads,unroll in ((1,128,1),(2,128,2),(4,128,2),(4,128,4),(2,256,2),(4,256,4)):
  schedule=dict(columns_per_warp=width,threads=threads,unroll=unroll)
  entry=root/f'{depth}-{width}-{threads}-{unroll}.py';artifact=entry.with_suffix('.tbin');artifact.unlink(missing_ok=True);entry.write_text(source('quantized_experts',{**p,**schedule},contract='reference',provider='cuda',target='sm_89',preferred='akas.ops.moe.cuda:quantized_experts_warp').render());build_artifact(entry,artifact,provider='cuda',target='sm_89');paths.append((schedule,artifact))
 meta,_=read_artifact(paths[0][1]);case=validation('quantized_experts',p,meta['arguments'])
 reference=None
 with tensor.Device(provider='cuda') as device:
  buffers={name:device.from_numpy(array) for name,array in case.arguments.items()}
  try:
   for schedule,path in [*paths,paths[0]]:
    meta,_=read_artifact(path);args=[buffers[a['name']] for a in meta['arguments']]
    with device.load(path) as module:
     module.launch(*args);actual=buffers['out'].to_numpy()
     np.testing.assert_allclose(actual,case.expected['out'],rtol=case.rtol,atol=case.atol)
     if reference is None:reference=actual.copy()
     same=np.array_equal(actual,reference)
     plan=ExecutionPlan(device);plan.bind(module,args)
     with plan.capture(resources=(module,*args)) as graph:
      timing=measure(device,graph.launch,warmup=2,samples=5,repeats=20)
     print(json.dumps(dict(depth=depth,columns=columns,routed=routed,schedule=schedule,timing=timing,correct=True,bitwise_baseline=same)),flush=True)
  finally:
   for b in buffers.values():b.release()
