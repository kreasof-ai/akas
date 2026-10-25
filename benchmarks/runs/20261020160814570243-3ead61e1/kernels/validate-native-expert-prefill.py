import json,time,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.artifacts.read import read_program,artifact_path
from akas.kernels.contracts import sample_arguments
from akas.quantization.weights.ggml.validation import sample_weight,decode_rows
from akas.ops.moe.validation import tile_tables
from akas.execution.plan import ExecutionPlan
from akas.benchmarking.history import Recorder
root=Path('build/lfm-moe/native-expert-prefill');program=Path('build/lfm-moe/codec-model/cuda/program');top=read_program(program)
manifest=json.loads((program/next(r['path'] for r in top['phases'] if r['phase']=='prefill' and r['chunk']==4096)).read_text())
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),weight_format='q4_0',operand_dtype='float16',accumulator_dtype='float32',qualification='Original Q4 coefficients independently decoded in bounded CPU row tiles and rounded to the existing F16 operand precision. Native-cache GEMM must match every current AOT packed GEMM output bitwise. Kernel timing excludes one-time decoding; transient cache and preparation costs reported separately. Not a serving result.')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Measure reusable native F16 expert weights for a layer-major long-prompt plan')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
try:
 with tensor.Device(provider='cuda') as device:
  for depth,columns in ((2048,1536),(1536,2048)):
   record=next(k for k in manifest['kernels'].values() if k['kind']=='quantized_experts' and k['parameters']['k']==depth);baseline=artifact_path(program,record);metadata,_=read_artifact(baseline)
   arrays=sample_arguments(metadata['arguments']);sample_weight(arrays['w'],2,decode=False)
   ids=(np.arange(4096)[:,None]+np.arange(4)[None,:])%64;counts=np.zeros(64,dtype='int32');routes=np.full((64,4096),-1,dtype='int32')
   for expert in range(64):
    owners,ranks=np.where(ids[:-1]==expert);counts[expert]=len(owners);routes[expert,:len(owners)]=owners*4+ranks
   arrays['counts'][:]=counts;arrays['routes'][:]=routes;tiles,local=tile_tables(counts,arrays['tile_experts'].shape[1]);arrays['tile_experts'][:]=tiles;arrays['tile_rows'][:]=local
   started=time.perf_counter();native=np.empty((64*columns,depth),dtype='float16')
   for begin in range(0,64*columns,128):native[begin:begin+128]=decode_rows(arrays['w'],2,depth,begin,min(begin+128,64*columns)).astype('float16')
   preparation=time.perf_counter()-started;buffers={n:device.from_numpy(a) for n,a in arrays.items()};native_buffer=device.from_numpy(native.ravel());expected=None
   paths=[('packed-baseline',baseline)]+[(f'{bm}-{bn}-{bk}-{threads}-{stages}',root/f'{depth}-{bm}-{bn}-{bk}-{threads}-{stages}.tbin') for bm,bn,bk,threads,stages in ((64,128,64,256,1),(64,128,64,256,2),(128,128,64,256,1),(128,128,64,256,2))]
   for label,path in paths:
    if not path.exists():continue
    with device.load(path) as kernel:
     args=tuple(native_buffer if a['name']=='w' and label!='packed-baseline' else buffers[a['name']] for a in kernel.manifest['arguments']);kernel.launch(*args);actual=buffers['out'].to_numpy()
     if expected is None:expected=actual.copy()
     np.testing.assert_array_equal(actual,expected)
     plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
     try:timing=measure(device,prepared.launch,warmup=30,samples=7,repeats=5)
     finally:prepared.close()
     check=dict(depth=depth,columns=columns,geometry=label,bitwise=True,cpu_preparation_seconds=preparation,transient_cache_bytes=native.nbytes,launch=kernel.manifest['launch'],timing=timing);checks.append(check);print('PASS',depth,label,'ms',timing['median_seconds']*1000,flush=True)
   native_buffer.release()
   for buffer in buffers.values():buffer.release()
   del arrays,native,expected,actual,buffers
 passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 for path in (__file__,'build/lfm-moe/native_expert_prefill.py','build/lfm-moe/compile-native-expert-prefill.py'):rec.artifact(path,'kernels/'+Path(path).name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
