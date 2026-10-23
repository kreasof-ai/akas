"""Codec-backed warp MMA prototype with register operands and small route groups."""
from akas.kernels.kernel import Tensor,kernel,output

CUDA_SOURCE=r'''
__device__ __forceinline__ int akas_register_mma(float* d,
 unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
 return 0;
}
'''

def factory(spec,schedule):
 import tilelang.language as T
 from akas.quantization.weights import bind_parts
 p={**spec.parameters,**schedule}; rows,top,experts,columns,depth=(p[n] for n in ('r','top','experts','o','k'))
 if not 1<rows<=16 or depth%16:raise ValueError('Register MMA prototype requires 2-16 rows and K16 depth')
 binding=bind_parts(p['weight_parts'],depth,provider='cuda');storage,read=binding.storage.values,binding.read
 routed=p.get('routed_input',False); shape=(rows,top,depth) if routed else (rows,depth)
 threads=p.get('threads',128);warps=threads//32;unroll=p.get('unroll',1)
 capacity=(rows*top+3)//4+experts;groups=min(rows*top,experts)
 @T.macro
 def activation(x,route,channel):
  if routed:return x[route//top,route%top,channel]
  else:return x[route//top,channel]
 @kernel
 def compute(x:Tensor(shape,p.get('input_dtype','float32')),w:Tensor(storage.shape,storage.dtype),counts:Tensor((experts,),'int32'),routes:Tensor((experts,rows),'int32'),tile_experts:Tensor((5,capacity),'int32'),tile_rows:Tensor((5,capacity),'int32'))->Tensor((rows,top,columns),'float32'):
  out=output((rows,top,columns),'float32')
  with T.Kernel(T.ceildiv(columns,warps*8),groups,threads=threads) as (bx,group):
   T.import_source(CUDA_SOURCE)
   tx=T.get_thread_binding();lane=tx%32;warp=tx//32;g=lane//4;t=lane%4
   expert=tile_experts[2,group];start=tile_rows[2,group]*16
   if expert>=0:
    accum=T.alloc_local((4,),'float32');a=T.alloc_local((4,),'uint32');b=T.alloc_local((2,),'uint32')
    for i in T.unroll(4):accum[i]=0
    for outer in T.serial(T.ceildiv(depth//16,unroll)):
     for local in T.unroll(unroll):
      k=(outer*unroll+local)*16
      for half in T.unroll(2):
       index=start+g+half*8
       route=routes[expert,T.min(index,rows-1)]
       for step in T.unroll(2):
        lo=T.alloc_var('float32');hi=T.alloc_var('float32');lo=0;hi=0
        if index<counts[expert] and k+step*8+2*t+1<depth:
         lo=activation(x,route,k+step*8+2*t)
         hi=activation(x,route,k+step*8+2*t+1)
        a[half+step*2]=T.call_extern('uint32','tensor_pack_f16x2',lo,hi)
      for step in T.unroll(2):
       b[step]=0
       if bx*warps*8+warp*8+g<columns and k+step*8+2*t+1<depth:
        b[step]=T.call_extern('uint32','tensor_pack_f16x2',read((w,),expert*columns+bx*warps*8+warp*8+g,k+step*8+2*t),read((w,),expert*columns+bx*warps*8+warp*8+g,k+step*8+2*t+1))
      T.ptx_mma('float32','m16n8k16','row','col','float16','float16','float32',a.data,0,b.data,0,accum.data,0,False)
    for half in T.unroll(2):
     index=start+g+half*8
     if index<counts[expert]:
      route=routes[expert,index]
      for component in T.unroll(2):
       column=bx*warps*8+warp*8+2*t+component
       if column<columns:out[route//top,route%top,column]=accum[half*2+component]
  return out
 return compute

def make(spec,schedule):
 from akas.kernels.definition import KernelSpec
 return factory(KernelSpec(**spec),schedule)
