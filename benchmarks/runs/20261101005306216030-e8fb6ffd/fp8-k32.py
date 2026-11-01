import tilelang.language as T
@T.prim_func
def compute(x:T.Tensor((33,128),'uint8'),w:T.Tensor((128,128),'uint8'),out:T.Tensor((33,128),'float32')):
 with T.Kernel(2,3,threads=128) as (bx,by):
  a=T.alloc_shared((16,128),'float8_e4m3fn')
  b=T.alloc_shared((64,128),'float8_e4m3fn')
  total=T.alloc_fragment((16,64),'float32')
  partial=T.alloc_fragment((16,64),'float32')
  source=T.view(x,dtype='float8_e4m3fn')
  weights=T.view(w,dtype='float8_e4m3fn')
  T.copy(source[by*16:by*16+16,0:128],a)
  T.copy(weights[bx*64:bx*64+64,0:128],b)
  T.clear(total)
  for q in T.unroll(4):
   T.gemm(a[0:16,q*32:q*32+32],b[0:64,q*32:q*32+32],partial,transpose_B=True,clear_accum=True)
   for i,j in T.Parallel(16,64):
    total[i,j]+=partial[i,j]
  T.copy(total,out[by*16:by*16+16,bx*64:bx*64+64])
def tensor_export():return {'kernel':compute}
