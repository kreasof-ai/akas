"""Packed word loads and contiguous half2 dequantization; experimental codec tile."""
CUDA_SOURCE=r'''
__device__ __forceinline__ unsigned int akas_q4_adjacent_pair(unsigned int bits, unsigned int scale) {
    unsigned int pair = (bits & 15u) | ((bits & 0x0f00u) << 8u) | 0x64006400u;
    unsigned int integers, result;
    asm("sub.f16x2 %0, %1, %2;" : "=r"(integers) : "r"(pair), "r"(0x64086408u));
    asm("mul.f16x2 %0, %1, %2;" : "=r"(result) : "r"(integers), "r"(scale | (scale << 16u)));
    return result;
}
'''

def tile_reader(rows,depth):
 import tilelang.language as T
 row_bytes=depth//32*18;words_count=rows*row_bytes//4
 @T.macro
 def load(encoded,destination,row,column):
  T.import_source(CUDA_SOURCE)
  words=T.view(encoded[0],(words_count,),dtype='uint32')
  for i,j in T.Parallel(destination.shape[0],destination.shape[1]//8):
   local=j//4*32+j%4*4;channel=column+local
   if row+i<rows and channel+19<depth:
    base=(row+i)*row_bytes+channel//32*18
    scale=T.cast(words[base//4]>>(base%4*8),'uint16')
    address=base+2+channel%16
    packed=T.call_extern('uint32','__funnelshift_r',words[address//4],words[(address+3)//4],T.cast(address%4*8,'uint32'))
    lo0=T.call_extern('uint32','akas_q4_adjacent_pair',packed,T.cast(scale,'uint32'))
    lo1=T.call_extern('uint32','akas_q4_adjacent_pair',packed>>16,T.cast(scale,'uint32'))
    hi0=T.call_extern('uint32','akas_q4_adjacent_pair',packed>>4,T.cast(scale,'uint32'))
    hi1=T.call_extern('uint32','akas_q4_adjacent_pair',packed>>20,T.cast(scale,'uint32'))
    destination[i,local]=T.reinterpret('float16',T.cast(lo0,'uint16'))
    destination[i,local+1]=T.reinterpret('float16',T.cast(lo0>>16,'uint16'))
    destination[i,local+2]=T.reinterpret('float16',T.cast(lo1,'uint16'))
    destination[i,local+3]=T.reinterpret('float16',T.cast(lo1>>16,'uint16'))
    destination[i,local+16]=T.reinterpret('float16',T.cast(hi0,'uint16'))
    destination[i,local+17]=T.reinterpret('float16',T.cast(hi0>>16,'uint16'))
    destination[i,local+18]=T.reinterpret('float16',T.cast(hi1,'uint16'))
    destination[i,local+19]=T.reinterpret('float16',T.cast(hi1>>16,'uint16'))
   else:
    for offset in T.unroll(4):
     destination[i,local+offset]=0;destination[i,local+offset+16]=0
 return load
