"""Prototype aligned adjacent-pair Q4 access, owned by the weight codec."""
from dataclasses import replace

def bind(parts, depth, *, provider):
    from akas.quantization.weights import bind_parts
    result=bind_parts(parts,depth,provider=provider)
    if (provider!='cuda' or len(parts)!=1 or parts[0]['format']!='ggml'
        or parts[0]['options'].get('encoding')!=2
        or parts[0]['options'].get('layout','raw')!='raw'
        or parts[0].get('row_order') or parts[0].get('column_order') or depth%64):
        return result
    import tilelang.language as T
    from akas.quantization.weights.ggml.cuda import _Q4_HALF2_SOURCE
    rows=parts[0]['rows'];stride=depth//32*18;words_count=rows*stride//4
    @T.macro
    def read_pair(encoded,row,column):
        T.import_source(_Q4_HALF2_SOURCE)
        words=T.meta_var(T.view(encoded[0],(words_count,),dtype='uint32'))
        base=row*stride+column//32*18
        scale=T.cast(words[base//4]>>(base%4*8),'uint16')
        address=base+2+column%16
        packed=T.call_extern('uint32','__funnelshift_r',words[address//4],words[(address+1)//4],T.cast(address%4*8,'uint32'))
        return T.call_extern('uint32','akas_q4_adjacent_pair',packed>>(column%32//16*4),T.cast(scale,'uint32'))
    fallback=result.require('read_tile')
    @T.macro
    def read_tile(encoded,destination,row,column):
        if destination.dtype=='float16':
            for i,j in T.Parallel(destination.shape[0],destination.shape[1]//2):
                bits=T.alloc_var('uint32');bits=0
                if row+i<rows and column+j*2+1<depth:
                    bits=read_pair(encoded,row+i,column+j*2)
                for component in T.vectorized(2):
                    destination[i,j*2+component]=T.reinterpret('float16',T.cast(bits>>(component*16),'uint16'))
        else:
            fallback(encoded,destination,row,column)
    return replace(result,read_pair=read_pair,read_tile=read_tile)
