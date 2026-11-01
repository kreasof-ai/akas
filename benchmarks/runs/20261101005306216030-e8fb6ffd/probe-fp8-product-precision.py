"""Compare product arithmetic before changing shared projection implementations."""
from pathlib import Path
import json
import numpy as np
import tensor
from akas.benchmarking.history import Recorder
from akas.quantization._shared.e4m3 import encode_storage,decode_storage

root=Path('build/lfm-serving/fp8-product-precision');root.mkdir(parents=True,exist_ok=True)
metadata=dict(runtime='akas',purpose='numerical-experiment',provider='cuda',
 qualification_scope='Raw matrix product precision experiment; no model or throughput qualification')
record=Recorder('benchmarks',metadata,kind='benchmark',change='Locate the FP8 product error before selecting a numerical implementation')
record.artifact(Path(__file__),'probe-fp8-product-precision.py')
rng=np.random.default_rng(41)
x=encode_storage(rng.normal(0,96,(33,128)))
w=encode_storage(rng.normal(0,96,(128,128)))
expected=(decode_storage(x).astype('float64')@decode_storage(w).astype('float64').T).astype('float32')
results=[]
try:
 with tensor.Device() as device:
  for label,dtype,interval in [('fp8-k128','float8_e4m3fn',128),('fp8-k32','float8_e4m3fn',32),('fp16-promoted','float16',128)]:
   entry=root/(label+'.py');artifact=root/(label+'.tbin')
   entry.write_text(f'''import tilelang.language as T
@T.prim_func
def compute(x:T.Tensor((33,128),'uint8'),w:T.Tensor((128,128),'uint8'),out:T.Tensor((33,128),'float32')):
 with T.Kernel(2,3,threads=128) as (bx,by):
  a=T.alloc_shared((16,128),'{dtype}')
  b=T.alloc_shared((64,128),'{dtype}')
  total=T.alloc_fragment((16,64),'float32')
  partial=T.alloc_fragment((16,64),'float32')
  source=T.view(x,dtype='float8_e4m3fn')
  weights=T.view(w,dtype='float8_e4m3fn')
  T.copy(source[by*16:by*16+16,0:128],a)
  T.copy(weights[bx*64:bx*64+64,0:128],b)
  T.clear(total)
  for q in T.unroll({128//interval}):
   T.gemm(a[0:16,q*{interval}:q*{interval}+{interval}],b[0:64,q*{interval}:q*{interval}+{interval}],partial,transpose_B=True,clear_accum=True)
   for i,j in T.Parallel(16,64):
    total[i,j]+=partial[i,j]
  T.copy(total,out[by*16:by*16+16,bx*64:bx*64+64])
def tensor_export():return {{'kernel':compute}}
''')
   tensor.build(entry,artifact,target='sm_89')
   with device.load(artifact) as executable:
    buffers=[device.from_numpy(x),device.from_numpy(w),device.zeros((33,128))]
    try:
     executable.launch(*buffers)
     actual=buffers[-1].to_numpy()
     result=dict(candidate=label,relative_rms=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
                 max_absolute=float(np.abs(actual-expected).max()),different=int(np.count_nonzero(actual!=expected)))
     results.append(result);print(result,flush=True)
     path=root/(label+'.npz');np.savez(path,x=x,w=w,actual=actual,expected=expected)
     record.artifact(entry,entry.name);record.artifact(path,path.name)
    finally:
     for buffer in buffers:buffer.release()
 path=root/'results.json';path.write_text(json.dumps(dict(metadata=metadata,candidates=results),indent=2)+'\n')
 record.artifact(path,'results.json')
 record.update(dict(metadata=metadata,results=[]),final=True,status='passed')
 print('RUN',record.id,flush=True)
except BaseException as error:
 record.update(dict(metadata=metadata,results=[],error=repr(error),candidates=results),final=True,status='failed')
 raise
