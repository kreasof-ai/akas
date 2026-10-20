import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.kernels.registry import validation
from akas.kernels.contracts import sample_arguments
from akas.quantization.weights.ggml.validation import sample_weight,decode_rows
from akas.ops.moe.validation import tile_tables
from akas.execution.plan import ExecutionPlan
from akas.memory.transfer import write
from akas.benchmarking.history import Recorder
root=Path('build/lfm-moe/q4-async-word-prefill');baseline_root=Path('build/lfm-moe/q4-word-prefill')
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),weight_format='q4_0',operand_dtype='float16',accumulator_dtype='float32',qualification='Independent bitwise decoded-weight check including signed zero/subnormals; independent small expert oracle; large outputs against qualified scalar-decode tensor-core reference; kernel timings only')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Asynchronously stage compressed expert tiles before adjacent half2 dequantization')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
try:
 with tensor.Device(provider='cuda') as device:
  path=root/'small.tbin';metadata,_=read_artifact(path)
  p=dict(r=33,top=2,experts=4,k=256,o=128,input_dtype='float32',routed_input=False,round_inputs=True,tile_map=True,weight_parts=({'rows':512,'format':'ggml','options':{'encoding':2}},))
  case=validation('quantized_experts',p,metadata['arguments']);buffers={n:device.from_numpy(a) for n,a in case.arguments.items()}
  with device.load(path) as kernel:
   kernel.launch(*[buffers[a['name']] for a in kernel.manifest['arguments']]);np.testing.assert_allclose(buffers['out'].to_numpy(),case.expected['out'],rtol=case.rtol,atol=case.atol)
  for b in buffers.values():b.release()
  checks.append(dict(phase='small-experts',correct=True));print('SMALL_ORACLE_OK',flush=True)
  for depth,columns,routed in ((2048,1536,False),(1536,2048,True)):
   metadata,_=read_artifact(baseline_root/f'baseline-{depth}.tbin');arrays=sample_arguments(metadata['arguments']);sample_weight(arrays['w'],2,decode=False)
   ids=(np.arange(4096)[:,None]+np.arange(4)[None,:])%64;counts=np.zeros(64,dtype='int32');routes=np.full((64,4096),-1,dtype='int32')
   for expert in range(64):
    owners,ranks=np.where(ids[:-1]==expert);counts[expert]=len(owners);routes[expert,:len(owners)]=owners*4+ranks
   arrays['counts'][:]=counts;arrays['routes'][:]=routes;tiles,local=tile_tables(counts,arrays['tile_experts'].shape[1]);arrays['tile_experts'][:]=tiles;arrays['tile_rows'][:]=local
   buffers={n:device.from_numpy(a) for n,a in arrays.items()};expected=None
   paths=[('baseline',baseline_root/f'baseline-{depth}.tbin'),('word-best',baseline_root/f'{depth}-64-128-128-256.tbin')]+[(f'{bm}-{bn}-{bk}-{threads}-{stages}',root/f'{depth}-{bm}-{bn}-{bk}-{threads}-{stages}.tbin') for bm,bn,bk,threads,stages in ((64,128,64,128,2),(64,128,64,256,2),(64,128,64,128,3),(64,128,128,256,2))]
   for label,path in paths:
    write(device,buffers['out'],arrays['out'])
    with device.load(path) as kernel:
     args=tuple(buffers[a['name']] for a in kernel.manifest['arguments']);kernel.launch(*args);actual=buffers['out'].to_numpy()
     if expected is None:expected=actual.copy()
     np.testing.assert_allclose(actual,expected,rtol=.002,atol=.0002)
     bitwise=bool(np.array_equal(actual,expected))
     plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
     try:timing=measure(device,prepared.launch,warmup=50,samples=7,repeats=10)
     finally:prepared.close()
     check=dict(depth=depth,columns=columns,routed_input=routed,geometry=label,correct=True,bitwise=bitwise,launch=kernel.manifest['launch'],timing=timing);checks.append(check);print('PASS',json.dumps(check),flush=True)
    rec.artifact(path.with_suffix('.py'),'kernels/'+path.with_suffix('.py').name)
   for b in buffers.values():b.release()
 passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 for path in (__file__,'build/lfm-moe/q4_word_tile.py','build/lfm-moe/q4_async_word_tile.py','build/lfm-moe/q4_async_word_prefill.py','build/lfm-moe/compile-async-word-prefill.py'):
  rec.artifact(path,'kernels/'+Path(path).name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
