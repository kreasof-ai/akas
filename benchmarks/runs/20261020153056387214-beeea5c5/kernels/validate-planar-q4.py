"""Compare lossless aligned Q4 layouts with qualified raw-byte kernels."""
import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.artifacts.read import read_program,artifact_path
from akas.kernels.registry import validation
from akas.execution.plan import ExecutionPlan
from akas.benchmarking.history import Recorder
root=Path('build/lfm-moe/planar-q4');program=Path('build/lfm-moe/codec-model/cuda/program');top=read_program(program)
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),weight_format='q4_0',weight_layout='quants followed by original F16 scale bits in one allocation',operand_dtype='float16',accumulator_dtype='float32',qualification='Bitwise payload roundtrip; independent small odd-output expert oracle; bitwise full decode and prefill outputs against current AOT raw-byte kernels. Candidate payload size equals original; microbench holds both payloads to compare. Kernel timing only.')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Align original Q4 quants and scale bits without changing precision or payload size')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False

def repack(raw):
 blocks=raw.reshape(-1,18);packed=np.concatenate((blocks[:,2:].ravel(),blocks[:,:2].ravel()));quants=blocks.shape[0]*16
 restored=np.empty_like(blocks);restored[:,2:]=packed[:quants].reshape(-1,16);restored[:,:2]=packed[quants:].reshape(-1,2)
 np.testing.assert_array_equal(restored.ravel(),raw);assert packed.nbytes==raw.nbytes
 return packed
try:
 with tensor.Device(provider='cuda') as device:
  for rows,depth,columns,routed in ((1,2048,7,True),(1,2048,1536,False),(1,1536,2048,True),(4096,2048,1536,False),(4096,1536,2048,True)):
   path=root/f'{rows}-{depth}-{columns}.tbin';metadata,_=read_artifact(path)
   if columns==7:
    p=dict(r=1,experts=4,top=2,k=depth,o=columns,input_dtype='float32',routed_input=True,round_inputs=True,weight_parts=({'rows':4*columns,'format':'ggml','options':{'encoding':2}},));baseline=None
   else:
    manifest=json.loads((program/next(r['path'] for r in top['phases'] if r['phase']==('decode' if rows==1 else 'prefill') and r['chunk']==rows)).read_text())
    record=next(k for k in manifest['kernels'].values() if k['kind']=='quantized_experts' and k['parameters']['k']==depth);p=record['parameters'];baseline=artifact_path(program,record)
   case=validation('quantized_experts',p,metadata['arguments']);arrays=case.arguments;plane=repack(arrays['w']);buffers={n:device.from_numpy(a) for n,a in arrays.items()};packed_buffer=device.from_numpy(plane);expected=case.expected['out'] if baseline is None else None
   paths=([('baseline',baseline)] if baseline else [])+[('planar',path)]
   for label,path in paths:
    with device.load(path) as kernel:
     args=tuple(packed_buffer if a['name']=='w' and label=='planar' else buffers[a['name']] for a in kernel.manifest['arguments']);kernel.launch(*args);actual=buffers['out'].to_numpy()
     if expected is None:expected=actual.copy()
     np.testing.assert_allclose(actual,expected,rtol=case.rtol,atol=case.atol)
     bitwise=bool(np.array_equal(actual,expected))
     if baseline is not None:assert bitwise
     plan=ExecutionPlan(device);plan.bind(kernel,args);prepared=plan.capture(resources=(kernel,*args))
     try:timing=measure(device,prepared.launch,warmup=50,samples=7,repeats=10)
     finally:prepared.close()
     check=dict(rows=rows,depth=depth,columns=columns,layout=label,correct=True,bitwise=bitwise,payload_bytes=plane.nbytes,timing=timing);checks.append(check);print('PASS',rows,depth,columns,label,'ms',timing['median_seconds']*1000,'bitwise',bitwise,flush=True)
   packed_buffer.release()
   for buffer in buffers.values():buffer.release()
 passed=True
except BaseException as error:
 meta['error']=str(error);meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 for path in (__file__,'build/lfm-moe/planar_q4.py','build/lfm-moe/planar_decode.py','build/lfm-moe/planar_prefill.py','build/lfm-moe/compile-planar-q4.py'):rec.artifact(path,'kernels/'+Path(path).name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)
