"""Prototype staging of losslessly separated Q4 payloads and scale bits."""
from dataclasses import replace

def bind(parts,depth,*,provider):
 from akas.quantization.weights import bind_parts
 codec=bind_parts(parts,depth,provider=provider)
 if not (len(parts)==1 and parts[0]['format']=='ggml' and parts[0]['options'].get('encoding')==2 and parts[0]['options'].get('layout')=='planar'):
  return codec
 import tilelang.language as T
 from tvm import tirx
 from akas.quantization.weights.ggml.cuda import planar_binding
 rows=parts[0]['rows'];fallback=codec.read_tile
 @T.macro
 def read_tile(encoded,destination,row,column):
  if destination.dtype=='float16' and destination.shape[1]%128==0:
   height,width=destination.shape
   staged=T.alloc_shared((height*width//32*18//8,),'uint64')
   payload=T.meta_var(tirx.decl_buffer((rows,depth//8),'uint32',data=encoded[0].data))
   scales=T.meta_var(tirx.decl_buffer((rows,depth//32),'uint16',data=encoded[0].data,elem_offset=rows*depth//4))
   local_payload=T.meta_var(tirx.decl_buffer((height,width//8),'uint32',data=staged.data))
   local_scales=T.meta_var(tirx.decl_buffer((height,width//32),'uint16',data=staged.data,elem_offset=height*width//4))
   T.copy(payload[row:row+height,column//8:column//8+width//8],local_payload)
   T.copy(scales[row:row+height,column//32:column//32+width//32],local_scales)
   local= T.meta_var(planar_binding((height,width),encoding=2))
   raw=T.meta_var(T.view(staged,local.storage.values.shape,dtype=local.storage.values.dtype))
   local.read_tile((raw,),destination,0,0)
  else:fallback(encoded,destination,row,column)
 return replace(codec,read_tile=read_tile)
