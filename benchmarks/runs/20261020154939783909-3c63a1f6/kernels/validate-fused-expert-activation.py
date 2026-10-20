"""Independent small equations and full-size paired-projection comparison."""
import json,traceback
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
from akas.memory.transfer import write
from akas.benchmarking.history import Recorder
root=Path('build/lfm-moe/fused-expert-activation');program=Path('build/lfm-moe/codec-model/cuda/program');top=read_program(program)
manifest=json.loads((program/next(r['path'] for r in top['phases'] if r['phase']=='prefill' and r['chunk']==4096)).read_text())
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),weight_format='q4_0',operand_dtype='float16',accumulator_dtype='float32',qualification='Independent small F16-operand/F32-SwiGLU reference with odd output tails and inactive routes; full-size outputs against current qualified gate/up/SwiGLU sequence. F16 output relocates existing down-projection operand rounding. Kernel timings only.')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Fuse expert gate/up projections and SwiGLU while avoiding intermediate traffic')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False

def fixture(metadata,rows,top,experts,depth,columns):
 arrays=sample_arguments(metadata['arguments']);sample_weight(arrays['w'],2,decode=False)
 if 'w2' not in arrays:arrays['w2']=np.empty_like(arrays['w'])
 sample_weight(arrays['w2'],2,seed=32,decode=False);arrays['out'].fill(0)
 ids=(np.arange(rows)[:,None]+np.arange(top)[None,:])%experts;counts=np.zeros(experts,dtype='int32');routes=np.full((experts,rows),-1,dtype='int32')
 for expert in range(experts):
  owners,ranks=np.where(ids[:-1]==expert);counts[expert]=len(owners);routes[expert,:len(owners)]=owners*top+ranks
 arrays['counts'][:]=counts;arrays['routes'][:]=routes
 tiles,local=tile_tables(counts,arrays['tile_experts'].shape[1]);arrays['tile_experts'][:]=tiles;arrays['tile_rows'][:]=local
 return arrays,ids

def timing(device,calls):
 plan=ExecutionPlan(device)
 for kernel,args in calls:plan.bind(kernel,args)
 prepared=plan.capture(resources=tuple(item for kernel,args in calls for item in (kernel,*args)))
 try:return measure(device,prepared.launch,warmup=30,samples=7,repeats=5)
 finally:prepared.close()
try:
 with tensor.Device(provider='cuda') as device:
  for dtype in ('float32','float16'):
   path=root/f'small-{dtype}.tbin';metadata,_=read_artifact(path);arrays,ids=fixture(metadata,33,2,4,256,65);expected=np.zeros((33,2,65),dtype='float32')
   for expert in range(4):
    owners,ranks=np.where(ids[:-1]==expert);x=arrays['x'][owners].astype('float16').astype('float64')
    gate=decode_rows(arrays['w'],2,256,expert*65,(expert+1)*65).astype('float16').astype('float64');up=decode_rows(arrays['w2'],2,256,expert*65,(expert+1)*65).astype('float16').astype('float64')
    g=(x@gate.T).astype('float32');u=(x@up.T).astype('float32');expected[owners,ranks]=g/(1+np.exp(-g))*u
   expected=expected.astype(dtype);buffers={n:device.from_numpy(a) for n,a in arrays.items()}
   with device.load(path) as kernel:
    args=tuple(buffers[a['name']] for a in kernel.manifest['arguments']);kernel.launch(*args);np.testing.assert_allclose(buffers['out'].to_numpy(),expected,rtol=.002,atol=.0002)
    sentinel=np.full_like(arrays['out'],1.25);write(device,buffers['out'],sentinel);write(device,buffers['counts'],np.zeros_like(arrays['counts']));kernel.launch(*args);np.testing.assert_array_equal(buffers['out'].to_numpy(),sentinel)
   for buffer in buffers.values():buffer.release()
   checks.append(dict(phase='independent-small',output_dtype=dtype,correct=True,odd_output_tail=True,inactive_routes_unchanged=True));print('SMALL_PASS',dtype,flush=True)
  record=next(k for k in manifest['kernels'].values() if k['kind']=='quantized_experts' and k['parameters']['k']==2048)
  activation=next(k for k in manifest['kernels'].values() if k['kind']=='swiglu_experts');baseline=artifact_path(program,record);activation_path=artifact_path(program,activation)
  metadata,_=read_artifact(baseline);arrays,ids=fixture(metadata,4096,4,64,2048,1536);buffers={n:device.from_numpy(a) for n,a in arrays.items()}
  gate=device.empty(arrays['out'].shape,'float32');up=device.empty(arrays['out'].shape,'float32');write(device,gate,np.zeros_like(arrays['out']));write(device,up,np.zeros_like(arrays['out']))
  with device.load(baseline) as projection,device.load(activation_path) as swiglu:
   gate_args=tuple(gate if a['name']=='out' else buffers[a['name']] for a in projection.manifest['arguments'])
   up_args=tuple(up if a['name']=='out' else buffers['w2'] if a['name']=='w' else buffers[a['name']] for a in projection.manifest['arguments'])
   activation_args=(gate,up,buffers['out']);projection.launch(*gate_args);projection.launch(*up_args);swiglu.launch(*activation_args);expected=buffers['out'].to_numpy()
   measured=timing(device,[(projection,gate_args),(projection,up_args),(swiglu,activation_args)]);checks.append(dict(phase='large',geometry='three-kernel-baseline',output_dtype='float32',correct=True,timing=measured));print('BASELINE_MS',measured['median_seconds']*1000,flush=True)
  gate.release();up.release()
  for dtype in ('float32','float16'):
   destination=buffers['out'] if dtype=='float32' else device.from_numpy(np.zeros(expected.shape,dtype='float16'))
   reference=expected.astype(dtype)
   for bm,bn,bk,threads in ((64,64,64,256),(64,64,64,512),(32,64,64,256),(32,128,64,512)):
    path=root/f'{dtype}-{bm}-{bn}-{bk}-{threads}.tbin'
    if not path.exists():continue
    write(device,destination,np.zeros(expected.shape,dtype=dtype))
    with device.load(path) as kernel:
     args=tuple(destination if a['name']=='out' else buffers[a['name']] for a in kernel.manifest['arguments']);kernel.launch(*args);actual=destination.to_numpy()
     np.testing.assert_allclose(actual,reference,rtol=.002,atol=.0002);bitwise=bool(np.array_equal(actual,reference))
     measured=timing(device,[(kernel,args)]);check=dict(phase='large',geometry=f'{bm}-{bn}-{bk}-{threads}',output_dtype=dtype,correct=True,bitwise=bitwise,max_absolute_error=float(np.max(np.abs(actual.astype('float32')-reference))),timing=measured,launch=kernel.manifest['launch']);checks.append(check);print('PASS',dtype,check['geometry'],'ms',measured['median_seconds']*1000,'bitwise',bitwise,flush=True)
   if dtype=='float16':destination.release()
  for buffer in buffers.values():buffer.release()
 passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 for path in (__file__,'build/lfm-moe/fused_expert_activation.py','build/lfm-moe/compile-fused-expert-activation.py','build/lfm-moe/compile-fused-activation-small.py'):rec.artifact(path,'kernels/'+Path(path).name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
