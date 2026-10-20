"""Decode adjacent Q4 coefficients directly into MMA register fragments."""
from akas.quantization.weights.ggml.cuda import _Q4_HALF2_SOURCE

def reader(rows,depth):
 import tilelang.language as T
 stride=depth//32*18
 @T.macro
 def load(encoded,destination,row,column):
  T.import_source(_Q4_HALF2_SOURCE)
  for i,j in T.Parallel(destination.shape[0],destination.shape[1]//2):
   channel=column+j*2
   bits=T.alloc_var('uint32');bits=0
   if row+i<rows and channel+1<depth:
    base=(row+i)*stride+channel//32*18
    scale=T.call_extern('uint32','tensor_load_u16',T.address_of(encoded[0][base]))
    packed=T.call_extern('uint32','tensor_load_u16',T.address_of(encoded[0][base+2+channel%16]))
    bits=T.call_extern('uint32','akas_q4_adjacent_pair',packed>>(channel%32//16*4),scale)
   for c in T.vectorized(2):
    destination[i,j*2+c]=T.reinterpret('float16',T.cast(bits>>(c*16),'uint16'))
 return load
