"""Prototype compressed staging for the GGML Q4 tile reader."""
from dataclasses import replace

def bind(parts,depth,*,provider):
 from akas.quantization.weights import bind_parts
 codec=bind_parts(parts,depth,provider=provider)
 if (provider!='cuda' or len(parts)!=1 or parts[0]['format']!='ggml'
     or parts[0]['options'].get('encoding')!=2
     or parts[0]['options'].get('layout','raw')!='raw'
     or parts[0].get('row_order') or parts[0].get('column_order') or depth%128):return codec
 import tilelang.language as T
 from akas.quantization.weights.ggml.cuda import _Q4_HALF2_SOURCE
 rows=parts[0]['rows'];stride=depth//32*18;fallback=codec.require('read_tile')
 @T.macro
 def read_tile(encoded,destination,row,column):
  if destination.dtype=='float16' and destination.shape[1]%128==0:
   T.import_source(_Q4_HALF2_SOURCE)
   word_columns=destination.shape[1]//32*18//8
   raw=T.meta_var(T.view(encoded[0],(rows,stride//8),dtype='uint64'))
   staged=T.alloc_shared((destination.shape[0],word_columns),'uint64')
   T.copy(raw[row:row+destination.shape[0],column//32*18//8:column//32*18//8+word_columns],staged)
   words=T.meta_var(T.view(staged,(destination.shape[0]*word_columns*2,),dtype='uint32'))
   destination_words=T.meta_var(T.view(destination,(destination.shape[0],destination.shape[1]//4),dtype='uint64'))
   for i,j in T.Parallel(destination.shape[0],destination.shape[1]//8):
    local=j//4*32+j%4*4
    base=i*word_columns*8+local//32*18
    scale=T.cast(words[base//4]>>(base%4*8),'uint16')
    address=base+2+local%16
    packed=T.call_extern('uint32','__funnelshift_r',words[address//4],words[(address+3)//4],T.cast(address%4*8,'uint32'))
    lo0=T.call_extern('uint32','akas_q4_adjacent_pair',packed,T.cast(scale,'uint32'))
    lo1=T.call_extern('uint32','akas_q4_adjacent_pair',packed>>16,T.cast(scale,'uint32'))
    hi0=T.call_extern('uint32','akas_q4_adjacent_pair',packed>>4,T.cast(scale,'uint32'))
    hi1=T.call_extern('uint32','akas_q4_adjacent_pair',packed>>20,T.cast(scale,'uint32'))
    destination_words[i,local//4]=T.cast(lo0,'uint64') | (T.cast(lo1,'uint64')<<32)
    destination_words[i,local//4+4]=T.cast(hi0,'uint64') | (T.cast(hi1,'uint64')<<32)
  else:fallback(encoded,destination,row,column)
 return replace(codec,read_tile=read_tile)
