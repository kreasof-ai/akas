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
p=dict(r=512,h=32,kh=8,d=64,capacity=8192,chunk=512,kv_dtype='float16',provider='webgpu',optimized=True)
entry=Path('build/lfm-moe/attention-reuse.py');artifact=entry.with_suffix('.tbin');artifact.unlink(missing_ok=True);entry.write_text(source('attention',p,contract='reference',provider='webgpu',target='webgpu-portable-v1',preferred='attention.webgpu.tiled_prefill').render());build_artifact(entry,artifact,provider='webgpu',target='webgpu-portable-v1')
meta,_=read_artifact(artifact);case=validation('attention',p,meta['arguments']);reference=None
with tensor.Device(provider='webgpu') as device:
 buffers={name:device.from_numpy(array) for name,array in case.arguments.items()}
 try:
  for name,path in (('before',Path('build/lfm-moe/attention-tiled_prefill.tbin')),('reuse',artifact),('before-repeat',Path('build/lfm-moe/attention-tiled_prefill.tbin'))):
   meta,_=read_artifact(path);args=[buffers[a['name']] for a in meta['arguments']]
   with device.load(path) as module:
    module.launch(*args);actual=buffers['out'].to_numpy();np.testing.assert_allclose(actual,case.expected['out'],rtol=case.rtol,atol=case.atol)
    if reference is None:reference=actual.copy()
    plan=ExecutionPlan(device);plan.bind(module,args)
    prepared=plan.capture(resources=(module,*args))
    try:
     timing=measure(device,prepared.launch,warmup=2,samples=5,repeats=3)
    finally:prepared.close()
    timing['execution']='prepared-plan'
    print(json.dumps(dict(name=name,timing=timing,correct=True,max_delta=float(np.max(np.abs(actual-reference))))),flush=True)
 finally:
  for b in buffers.values():b.release()
