"""Paired expert projections with a fused SwiGLU epilogue."""
from akas.kernels.kernel import Tensor,kernel,output

def factory(p):
 import tilelang.language as T
 from akas.quantization.weights import bind_parts
 rows,top,experts,columns,depth=(p[n] for n in ('r','top','experts','o','k'))
 bm,bn,bk,threads=(p[n] for n in ('block_m','columns','block_k','threads'))
 capacity=(rows*top+3)//4+experts;map_block=min(bm,64);pairs=bm//map_block;groups=(rows*top+map_block-1)//map_block+experts;layout=map_block.bit_length()-3
 binding=bind_parts(p['weight_parts'],depth,provider='cuda');storage=binding.storage.values;load=binding.require('read_tile')
 dtype=p.get('output_dtype','float32')
 @kernel
 def compute(x:Tensor((rows*top,depth),'float16'),w:Tensor(storage.shape,storage.dtype),w2:Tensor(storage.shape,storage.dtype),counts:Tensor((experts,),'int32'),offsets:Tensor((experts+1,),'int32'),tile_experts:Tensor((5,capacity),'int32'),tile_rows:Tensor((5,capacity),'int32'))->Tensor((rows*top,columns),dtype):
  out=output((rows*top,columns),dtype)
  with T.Kernel(T.ceildiv(columns,bn),groups,threads=threads) as (bx,group):
   expert=tile_experts[layout,group];mapped_row=tile_rows[layout,group];by=mapped_row//pairs
   if expert>=0 and mapped_row%pairs==0 and by*bm<counts[expert]:
    lhs=T.alloc_shared((bm,bk),'float16');gate=T.alloc_shared((bn,bk),'float16');up=T.alloc_shared((bn,bk),'float16')
    g=T.alloc_fragment((bm,bn),'float32');u=T.alloc_fragment((bm,bn),'float32');T.clear(g);T.clear(u)
    for tile in T.Pipelined(T.ceildiv(depth,bk),num_stages=p['stages']):
     T.copy(x[offsets[expert]+by*bm,tile*bk],lhs)
     load((w,),gate,expert*columns+bx*bn,tile*bk)
     load((w2,),up,expert*columns+bx*bn,tile*bk)
     T.gemm(lhs,gate,g,transpose_B=True);T.gemm(lhs,up,u,transpose_B=True)
    for i,j in T.Parallel(bm,bn):
     index=by*bm+i
     if index<counts[expert] and bx*bn+j<columns:
      out[offsets[expert]+index,bx*bn+j]=g[i,j]/(1+T.exp(-g[i,j]))*u[i,j]
  return out
 return compute
