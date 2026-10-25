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
    return replace(result,read_pair=read_pair)
