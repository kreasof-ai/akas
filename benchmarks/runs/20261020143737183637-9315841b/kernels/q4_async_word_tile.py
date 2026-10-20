"""Prototype asynchronous raw Q4 tile staging, dequantized in shared memory."""
from q4_word_tile import CUDA_SOURCE

def tile_reader(rows, depth):
 import tilelang.language as T
 row_bytes=depth//32*18
 @T.macro
 def load(encoded,destination,row,column):
  T.import_source(CUDA_SOURCE)
  width=destination.shape[1]//64*9
  raw=T.alloc_shared((destination.shape[0],width),'uint32')
  matrix=T.view(encoded[0],(rows,row_bytes//4),dtype='uint32')
  T.copy(matrix[row,column//64*9],raw,prefer_instruction='cp_async')
  for i,j in T.Parallel(destination.shape[0],destination.shape[1]//8):
   local=j//4*32+j%4*4
   base=local//32*18
   scale=T.cast(raw[i,base//4]>>(base%4*8),'uint16')
   address=base+2+local%16
   packed=T.call_extern('uint32','__funnelshift_r',raw[i,address//4],raw[i,(address+3)//4],T.cast(address%4*8,'uint32'))
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
 return load
