"""Reuse decoded FP8 coefficients in a bounded temporary F16 cache."""
from akas.kernels.kernel import Tensor,kernel,output

def extent_factory(p):
 import tilelang.language as T
 slots,chunk=p['slots'],p['chunk'];rows=slots*chunk;padded=1<<(chunk-1).bit_length()
 @kernel
 def compute(positions:Tensor((rows,),'int32'),enabled:Tensor((rows,),'int32'))->Tensor((slots,),'int32'):
  lengths=output((slots,),'int32')
  with T.Kernel(slots,threads=128) as slot:
   limits=T.alloc_fragment((padded,),'int32');last=T.alloc_fragment((1,),'int32')
   for i in T.Parallel(padded):limits[i]=T.if_then_else(i<chunk and enabled[slot*chunk+T.min(i,chunk-1)]!=0,positions[slot*chunk+T.min(i,chunk-1)]+1,0)
   T.reduce_max(limits,last,dim=0);lengths[slot]=T.min(T.max(last[0],0),p['capacity'])
  return lengths
 return compute

def shadow_factory(p):
 import tilelang.language as T
 from akas.quantization.kv.codec import get
 from akas.quantization._shared.access import packed_arguments
 slots,heads,cap,d=(p[n] for n in ('slots','kh','capacity','d'));shape=(slots,heads,cap,d);scale_shape=(slots,heads,cap,(d+127)//128)
 codec=get('fp8').bind(shape,provider='cuda');decode=packed_arguments(codec.require('read_packed'));block=128
 @kernel
 def compute(kc:Tensor(shape,'uint8'),vc:Tensor(shape,'uint8'),ks:Tensor(scale_shape,'float32'),vs:Tensor(scale_shape,'float32'),lengths:Tensor((slots,),'int32'))->tuple[Tensor(shape,'float16'),Tensor(shape,'float16')]:
  keys=output(shape,'float16');values=output(shape,'float16')
  encoded_k=T.meta_var(T.view(kc,dtype=codec.packed_dtype));encoded_v=T.meta_var(T.view(vc,dtype=codec.packed_dtype))
  with T.Kernel(T.ceildiv(cap,block),heads,slots,threads=256) as (tile,head,slot):
   # Guard loads, including scales: never decode poisoned future cache entries.
   for i,j in T.Parallel(block,d):
    position=tile*block+i
    if position<cap:
     keys[slot,head,position,j]=T.if_then_else(position<lengths[slot],T.cast(encoded_k[slot,head,position,j],'float32')*ks[slot,head,position,j//128],0)
     values[slot,head,position,j]=T.if_then_else(position<lengths[slot],T.cast(encoded_v[slot,head,position,j],'float32')*vs[slot,head,position,j//128],0)
  return keys,values
 return compute
