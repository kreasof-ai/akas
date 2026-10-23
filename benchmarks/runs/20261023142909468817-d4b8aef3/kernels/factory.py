"""Small expert groups reuse decoded register coefficients across live rows."""
from akas.kernels.kernel import Tensor,kernel,output

def factory(spec,schedule):
 import tilelang.language as T
 from experimental_weight_codec import bind
 p={**spec.parameters,**schedule};rows,top,experts,columns,depth=(p[n] for n in ('r','top','experts','o','k'))
 if not 1<rows<=16 or depth%256:raise ValueError('Requires 2-16 rows and K256 depth')
 binding=bind(p['weight_parts'],depth,provider='cuda');storage,read_pair=binding.storage.values,binding.require('read_pair')
 br=p.get('route_rows',2);width=p.get('columns_per_warp',1);threads=p.get('threads',128);warps=threads//32
 capacity=(rows*top+3)//4+experts;groups=min(rows*top,experts)
 routed=p.get('routed_input',False);shape=(rows,top,depth) if routed else (rows,depth)
 @T.macro
 def activation(x,route,channel):
  if routed:return x[route//top,route%top,channel]
  else:return x[route//top,channel]
 @kernel
 def compute(x:Tensor(shape,p.get('input_dtype','float32')),w:Tensor(storage.shape,storage.dtype),counts:Tensor((experts,),'int32'),routes:Tensor((experts,rows),'int32'),tile_experts:Tensor((5,capacity),'int32'),tile_rows:Tensor((5,capacity),'int32'))->Tensor((rows,top,columns),'float32'):
  out=output((rows,top,columns),'float32')
  with T.Kernel(T.ceildiv(columns,warps*width),groups,T.ceildiv(rows,br),threads=threads) as (bx,group,rb):
   tx=T.get_thread_binding();warp=tx//32;lane=tx%32
   expert=tile_experts[2,group];start=tile_rows[2,group]*16+rb*br
   if expert>=0 and start<counts[expert]:
    accum=T.alloc_local((br,width),'float32');coefficient=T.alloc_local((8,),'float32')
    for i,j in T.unroll(br,width):accum[i,j]=0
    for tile in T.serial(depth//256):
     for col in T.unroll(width):
      column=(bx*warps+warp)*width+col
      if column<columns:
       for pair in T.unroll(4):
        bits=read_pair((w,),expert*columns+column,tile*256+lane*8+pair*2)
        coefficient[pair*2]=T.cast(T.reinterpret('float16',T.cast(bits,'uint16')),'float32')
        coefficient[pair*2+1]=T.cast(T.reinterpret('float16',T.cast(bits>>16,'uint16')),'float32')
       for i in T.unroll(br):
        if start+i<counts[expert]:
         route=routes[expert,T.min(start+i,rows-1)]
         for component in T.unroll(8):
          value=T.cast(T.cast(activation(x,route,tile*256+lane*8+component),'float16'),'float32')
          accum[i,col]+=value*coefficient[component]
    for i in T.unroll(br):
     for col in T.unroll(width):
      for step in T.unroll(5):accum[i,col]+=T.shfl_down(accum[i,col],16>>step)
      column=(bx*warps+warp)*width+col
      if lane==0 and start+i<counts[expert] and column<columns:
       route=routes[expert,T.min(start+i,rows-1)];out[route//top,route%top,column]=accum[i,col]
  return out
 return compute

def make(spec,schedule):
 from akas.kernels.definition import KernelSpec
 return factory(KernelSpec(**spec),schedule)
