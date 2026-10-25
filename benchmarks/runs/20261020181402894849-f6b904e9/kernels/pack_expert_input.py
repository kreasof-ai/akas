"""Materialize the existing F16 expert operands in compact expert-row order."""
from akas.kernels.kernel import Tensor,kernel,output

def offsets_factory(p):
 import tilelang.language as T
 experts=p['experts']
 @kernel
 def compute(counts:Tensor((experts,),'int32'))->Tensor((experts+1,),'int32'):
  offsets=output((experts+1,),'int32')
  with T.Kernel(1,threads=128):
   for i in T.Parallel(experts+1):
    total=T.alloc_var('int32');total=0
    for j in T.serial(i):total+=counts[j]
    offsets[i]=total
  return offsets
 return compute

def gather_factory(p):
 import tilelang.language as T
 rows,top,experts,depth=(p[n] for n in ('r','top','experts','k'));capacity=(rows*top+3)//4+experts;groups=(rows*top+63)//64+experts;bk=256
 @kernel
 def compute(x:Tensor((rows,depth),'float32'),counts:Tensor((experts,),'int32'),offsets:Tensor((experts+1,),'int32'),routes:Tensor((experts,rows),'int32'),tile_experts:Tensor((5,capacity),'int32'),tile_rows:Tensor((5,capacity),'int32'))->Tensor((rows*top,depth),'float16'):
  packed=output((rows*top,depth),'float16')
  with T.Kernel(T.ceildiv(depth,bk),groups,threads=256) as (tile,group):
   expert=tile_experts[4,group];by=tile_rows[4,group]
   if expert>=0:
    for i,j in T.Parallel(64,bk):
     index=by*64+i
     if index<counts[expert] and tile*bk+j<depth:
      route=routes[expert,index]
      packed[offsets[expert]+index,tile*bk+j]=x[route//top,tile*bk+j]
  return packed
 return compute
