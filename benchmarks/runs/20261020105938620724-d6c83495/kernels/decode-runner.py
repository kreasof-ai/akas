import json,sys
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.compilation.emit import build_artifact
from akas.compilation.selection import source
from akas.kernels.registry import validation
from akas.execution.plan import ExecutionPlan
provider=sys.argv[1];target='sm_89' if provider=='cuda' else 'webgpu-portable-v1'
root=Path('build/lfm-moe/attention-decode-explore')/provider;root.mkdir(parents=True,exist_ok=True)
for cap in (32,8192,131072):
 p=dict(r=1,h=32,kh=8,d=64,capacity=cap,chunk=1,splits=16,kv_dtype='float16',provider=provider,optimized=True)
 paths=[]
 for name in (('attention.cuda.floating' if provider=='cuda' else 'attention.floating'),'attention.grouped_decode'):
  entry=root/f'{cap}-{name}.py';artifact=entry.with_suffix('.tbin');artifact.unlink(missing_ok=True);entry.write_text(source('attention_partial',p,contract='reference',provider=provider,target=target,preferred=name).render());build_artifact(entry,artifact,provider=provider,target=target);paths.append((name,artifact))
 meta,_=read_artifact(paths[0][1]);case=validation('attention_partial',p,meta['arguments'])
 with tensor.Device(provider=provider) as device:
  buffers={name:device.from_numpy(array) for name,array in case.arguments.items()}
  try:
   for name,path in [*paths,('baseline-repeat',paths[0][1])]:
    meta,_=read_artifact(path);args=[buffers[a['name']] for a in meta['arguments']]
    with device.load(path) as module:
     module.launch(*args);actual=buffers['out'].to_numpy()
     np.testing.assert_allclose(actual,case.expected['out'],rtol=case.rtol,atol=case.atol)
     plan=ExecutionPlan(device);plan.bind(module,args)
     prepared=plan.capture(resources=(module,*args))
     try:
      timing=measure(device,prepared.launch,warmup=1,samples=3,repeats=10)
     finally:prepared.close()
     timing['execution']='prepared-plan'
     print(json.dumps(dict(provider=provider,cap=cap,name=name,timing=timing,correct=True)),flush=True)
  finally:
   for b in buffers.values():b.release()
