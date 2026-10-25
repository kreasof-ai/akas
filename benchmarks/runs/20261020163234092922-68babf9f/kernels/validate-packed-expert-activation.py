"""Measure the fused sequence including GPU route packing, with fresh outputs."""
import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.artifacts.read import read_program,artifact_path
from akas.kernels.contracts import sample_arguments
from akas.quantization.weights.ggml.validation import sample_weight
from akas.ops.moe.validation import tile_tables
from akas.execution.plan import ExecutionPlan
from akas.memory.transfer import write
from akas.benchmarking.history import Recorder
root=Path('build/lfm-moe/packed-expert-activation');program=Path('build/lfm-moe/codec-model/cuda/program');top=read_program(program);manifest=json.loads((program/next(r['path'] for r in top['phases'] if r['phase']=='prefill' and r['chunk']==4096)).read_text());projection_path=artifact_path(program,next(k for k in manifest['kernels'].values() if k['kind']=='quantized_experts' and k['parameters']['k']==2048));swiglu_path=artifact_path(program,next(k for k in manifest['kernels'].values() if k['kind']=='swiglu_experts'))
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),prior_harness_failure_run_id='20261020163209845762-5f9a179c',weight_format='q4_0',operand_dtype='float16',accumulator_dtype='float32',qualification='Full-size packed gate/up/SwiGLU must match current AOT results at existing down-projection F16 operand precision. Includes GPU prefix sum, route gathering and fused projection in timing. Independent CPU packing oracle and inactive output sentinels; original fused arithmetic separately qualified at run 20261020154939783909-3c63a1f6. No serving claim.')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Pack once and fuse gate/up/SwiGLU directly to the compact F16 down-projection operand');report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False

def execute(device,calls):
 plan=ExecutionPlan(device)
 for kernel,args in calls:plan.bind(kernel,args)
 return plan.capture(resources=tuple(item for kernel,args in calls for item in (kernel,*args)))
try:
 with tensor.Device(provider='cuda') as device:
  signature,_=read_artifact(projection_path);arrays=sample_arguments(signature['arguments']);sample_weight(arrays['w'],2,decode=False);arrays['w2']=np.empty_like(arrays['w']);sample_weight(arrays['w2'],2,seed=32,decode=False);arrays['out'].fill(0)
  ids=(np.arange(4096)[:,None]+np.arange(4)[None,:])%64;counts=np.zeros(64,dtype='int32');routes=np.full((64,4096),-1,dtype='int32')
  for expert in range(64):
   owners,ranks=np.where(ids[:-1]==expert);counts[expert]=len(owners);routes[expert,:len(owners)]=owners*4+ranks
  arrays['counts'][:]=counts;arrays['routes'][:]=routes;tiles,local=tile_tables(counts,arrays['tile_experts'].shape[1]);arrays['tile_experts'][:]=tiles;arrays['tile_rows'][:]=local;buffers={n:device.from_numpy(a) for n,a in arrays.items()}
  gate=device.from_numpy(np.zeros_like(arrays['out']));up=device.from_numpy(np.zeros_like(arrays['out']));packed=device.from_numpy(np.full((4096*4,2048),np.nan,dtype='float16'));offsets=device.empty((65,),'int32');out=device.empty((4096*4,1536),'float16');sentinel=np.full(out.shape,1.25,dtype='float16')
  with device.load(projection_path) as projection,device.load(swiglu_path) as swiglu:
   gargs=tuple(gate if a['name']=='out' else buffers[a['name']] for a in projection.manifest['arguments']);uargs=tuple(up if a['name']=='out' else buffers['w2'] if a['name']=='w' else buffers[a['name']] for a in projection.manifest['arguments']);aargs=(gate,up,buffers['out']);prepared=execute(device,[(projection,gargs),(projection,uargs),(swiglu,aargs)])
   try:prepared.launch();expected=buffers['out'].to_numpy();timing=measure(device,prepared.launch,warmup=30,samples=7,repeats=5)
   finally:prepared.close()
   checks.append(dict(geometry='three-kernel-baseline',timing=timing,correct=True));print('BASELINE_MS',timing['median_seconds']*1000,flush=True)
  cpu_offsets=np.concatenate((np.zeros(1,dtype='int32'),np.cumsum(counts,dtype='int32')));reference=sentinel.copy();cpu_packed=np.full(packed.shape,np.nan,dtype='float16')
  for expert in range(64):
   route=routes[expert,:counts[expert]];start,end=cpu_offsets[expert:expert+2];reference[start:end]=expected[route//4,route%4].astype('float16');cpu_packed[start:end]=arrays['x'][route//4].astype('float16')
  with device.load(root/'offsets.tbin') as prefix,device.load(root/'gather.tbin') as gather:
   shared=dict(buffers,offsets=offsets,packed=packed);pargs=tuple(shared[a['name']] for a in prefix.manifest['arguments']);gargs=tuple(shared[a['name']] for a in gather.manifest['arguments']);prefix.launch(*pargs);gather.launch(*gargs);np.testing.assert_array_equal(offsets.to_numpy(),cpu_offsets);np.testing.assert_array_equal(packed.to_numpy(),cpu_packed)
   for geometry in ('64-64-64-256-1','128-64-64-256-1','128-64-64-256-2','64-128-64-256-1'):
    path=root/f'{geometry}.tbin'
    if not path.exists():continue
    with device.load(path) as fused:
     write(device,out,sentinel);bindings=dict(buffers,x=packed,offsets=offsets,out=out);args=tuple(bindings[a['name']] for a in fused.manifest['arguments']);prepared=execute(device,[(prefix,pargs),(gather,gargs),(fused,args)])
     try:
      prepared.launch();actual=out.to_numpy();np.testing.assert_array_equal(actual,reference);timing=measure(device,prepared.launch,warmup=30,samples=7,repeats=5)
      write(device,buffers['counts'],np.zeros_like(counts));write(device,out,sentinel);prepared.launch();np.testing.assert_array_equal(out.to_numpy(),sentinel);write(device,buffers['counts'],counts)
     finally:prepared.close()
     check=dict(geometry=geometry,bitwise=True,inactive_unchanged=True,packing_included=True,timing=timing);checks.append(check);print('PASS',geometry,'ms',timing['median_seconds']*1000,flush=True)
  if len(checks)<2:raise ValueError('No fused candidate qualified')
 passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 for path in (__file__,'build/lfm-moe/packed_expert_activation.py','build/lfm-moe/compile-packed-expert-activation.py','build/lfm-moe/pack_expert_input.py','build/lfm-moe/compile-pack-expert-input.py'):rec.artifact(path,'kernels/'+Path(path).name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
