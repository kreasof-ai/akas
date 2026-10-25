"""Reuse each F32 activation across several output columns in one warp."""
from akas.kernels.kernel import Tensor,kernel,output

def factory(p):
 import tilelang.language as T
 rows,depth,columns=(p[n] for n in ('r','k','o'));threads=p.get('threads',128);micro=p.get('columns_per_warp',4);warps=threads//32
 stride,offset=p.get('output_stride',columns),p.get('output_offset',0)
 shape=(rows,stride) if p.get('matrix_output',False) else (rows*stride,)
 @kernel
 def compute(x:Tensor((rows*depth,),'float32'),w:Tensor((columns*depth,),'float32'))->Tensor(shape,'float32'):
  out=output(shape,'float32')
  with T.Kernel(T.ceildiv(columns,warps*micro),rows,threads=threads) as (block,row):
   tx=T.get_thread_binding();lane=tx%32;column=(block*warps+tx//32)*micro
   total=T.alloc_local((micro,),'float32');T.clear(total)
   for tile in T.serial(T.ceildiv(depth,32)):
    channel=tile*32+lane
    if channel<depth:
     value=x[row*depth+channel]
     for c in T.unroll(micro):
      if column+c<columns:total[c]+=value*w[(column+c)*depth+channel]
   for step in T.unroll(5):
    for c in T.unroll(micro):total[c]+=T.shfl_down(total[c],16>>step)
   if lane==0:
    for c in T.unroll(micro):
     if column+c<columns:
      if p.get('matrix_output',False):out[row,offset+column+c]=total[c]
      else:out[row*stride+offset+column+c]=total[c]
  return out
 return compute
