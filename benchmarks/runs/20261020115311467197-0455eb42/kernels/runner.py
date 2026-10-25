import json
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from akas.kernels.registry import validation
from akas.benchmarking.history import Recorder
root=Path('build/lfm-moe/packed-view')
metadata=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',device=dict(name='NVIDIA L4',provider='cuda'),qualification='Independent bounded CPU oracle plus bitwise comparison to the validated byte-load packed expert')
rec=Recorder('benchmarks',metadata,kind='diagnostic',change='Validate packed CUDA expert word views without a derived weight copy')
report=dict(schema='akas.benchmark.v1',metadata=metadata,results=[])
passed=False;checks=[]
try:
 for depth,columns,routed in ((2048,1536,False),(1536,2048,True)):
  p=dict(r=1,top=4,experts=64,k=depth,o=columns,input_dtype='float32',routed_input=routed,round_inputs=True,weight_parts=({'rows':64*columns,'format':'ggml','options':{'encoding':2}},))
  meta,_=read_artifact(root/f'{depth}-128.tbin');case=validation('quantized_experts',p,meta['arguments'])
  with tensor.Device(provider='cuda') as device:
   buffers={name:device.from_numpy(array) for name,array in case.arguments.items()}
   baseline=None
   for directory,threads in (('packed-experts',128),('packed-view',128),('packed-view',256)):
    path=Path('build/lfm-moe')/directory/f'{depth}-{threads}.tbin';meta,_=read_artifact(path)
    with device.load(path) as kernel:
     kernel.launch(*[buffers[a['name']] for a in meta['arguments']]);actual=buffers['out'].to_numpy()
     np.testing.assert_allclose(actual,case.expected['out'],rtol=case.rtol,atol=case.atol)
     if baseline is None:baseline=actual.copy()
     np.testing.assert_array_equal(actual,baseline)
     checks.append(dict(depth=depth,routed=routed,implementation=directory,threads=threads,correct=True,bitwise=True))
     print('PASS',checks[-1],flush=True)
   for b in buffers.values():b.release()
 metadata['checks']=checks;passed=True
finally:
 rec.artifact(__file__,'kernels/runner.py');rec.artifact('build/lfm-moe/packed_view.py','kernels/packed_view.py')
 rec.update(report,final=True,status='passed' if passed else 'failed')
 print('RETAINED',rec.id,passed,flush=True)
