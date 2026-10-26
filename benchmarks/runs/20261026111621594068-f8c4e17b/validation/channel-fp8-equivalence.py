from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from akas.compilation.selection import source
from akas.compilation.emit import build_artifact
from akas.kernels.registry import validation
from akas.quantization._shared.e4m3 import decode_storage
p=dict(r=32,k=10240,o=2560,weight_format='fp8',weight_options={'scale_layout':'channel'})
s=Path('build/spark-packed/rounding.py');a=s.with_suffix('.tbin');s.write_text(source('quantized_linear',p,contract='reference',provider='cuda',target='sm_89').render());None if a.exists() else build_artifact(s,a,provider='cuda',target='sm_89');m,_=read_artifact(a);c=validation('quantized_linear',p,m['arguments']);types={v['name']:v['dtype'] for v in m['arguments']}
with tensor.Device() as d:
 b={n:d.from_numpy(v,**({'dtype':'bfloat16'} if types[n]=='bfloat16' else {})) for n,v in c.arguments.items()};mod=d.load(a);mod.launch(*(b[v['name']] for v in m['arguments']));y=b['out'].to_numpy()
 x=c.arguments['x'].astype('float16').astype('float64');w=(decode_storage(c.arguments['w'])*c.arguments['weight_scales']).astype('float16').astype('float64')
 from akas.compilation.selection import selected_source
 from akas.kernels.definition import definition
 spec=dict(kind='linear',parameters=dict(r=32,k=10240,o=2560,type=1,ggml=True,optimized=True),contract='reference',phase='prefill');caps=dict(provider='cuda',target='sm_89')
 impl,sched=definition(spec,dict(block_m=32,block_n=64,block_k=32,stages=2,threads=128),caps,preferred='linear.cuda.packed:prefill')
 ss=Path('build/spark-packed/floating-comparison.py');aa=ss.with_suffix('.tbin');ss.write_text(selected_source(spec,sched,caps,impl).render());None if aa.exists() else build_artifact(ss,aa,provider='cuda',target='sm_89');mm=d.load(aa)
 xx=d.from_numpy(c.arguments['x'].reshape(-1));ww=d.from_numpy(w.astype('float16').reshape(-1));yy=d.empty((32*2560,),'float32');mm.launch(xx,ww,yy);other=yy.to_numpy().reshape(y.shape)
 print('quantized_vs_same_fp16_matmul max_abs_error',np.max(abs(y-other)), 'equal_elements',np.sum(y==other),'of',y.size,flush=True)
 for flushx,flushw in [(False,False),(True,False),(False,True),(True,True)]:
  xx=np.where(abs(x)<2**-14,0,x) if flushx else x;ww=np.where(abs(w)<2**-14,0,w) if flushw else w;ref=xx@ww.T;diff=abs(y-ref);print(flushx,flushw,'max',diff.max(),'norm',np.linalg.norm(diff),'failed',np.sum(diff>1e-4+.002*abs(ref)),flush=True)
