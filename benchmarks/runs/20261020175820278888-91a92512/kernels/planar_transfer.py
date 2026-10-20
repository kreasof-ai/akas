"""Reversible byte permutation; allocation size and every stored bit survive."""
from akas.kernels.kernel import Tensor,kernel,output

def factory(count,inverse=False):
 import tilelang.language as T
 blocks=count//18
 @kernel
 def compute(w:Tensor((count,),'uint8'))->Tensor((count,),'uint8'):
  out=output((count,),'uint8')
  with T.Kernel(T.ceildiv(blocks,128),threads=128) as bx:
   for i,j in T.Parallel(128,16):
    block=bx*128+i
    if block<blocks:
     if inverse:out[block*18+2+j]=w[block*16+j]
     else:out[block*16+j]=w[block*18+2+j]
   for i,j in T.Parallel(128,2):
    block=bx*128+i
    if block<blocks:
     if inverse:out[block*18+j]=w[blocks*16+block*2+j]
     else:out[blocks*16+block*2+j]=w[block*18+j]
  return out
 return compute
