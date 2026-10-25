"""Experimental packed Q4_0 GEMV with the existing half-operand contract."""
from akas.kernels.kernel import Tensor, kernel, output

def factory(p):
    import tilelang.language as T
    @T.macro
    def word(w,offset):
        return T.call_extern('uint32','__funnelshift_r',w[offset//4],w[(offset+3)//4],T.cast(offset%4*8,'uint32'))
    @T.macro
    def half_value(w,offset):
        return T.cast(T.reinterpret('float16',T.cast(w[offset//4] >> (offset%4*8),'uint16')),'float32')
    depth,columns,experts,top = (p[n] for n in ('k','o','experts','top'))
    routed=p.get('routed_input',False)
    shape=(1,top,depth) if routed else (1,depth)
    threads=p.get('threads',128);warps=threads//32
    row_bytes=depth//32*18
    dtype=p.get('input_dtype','float32')
    @kernel
    def compute(x:Tensor(shape,dtype),w:Tensor((experts*columns*row_bytes,),'uint8'),ids:Tensor((1,top),'int32'),counts:Tensor((experts,),'int32'))->Tensor((1,top,columns),'float32'):
        out=output((1,top,columns),'float32')
        words=T.view(w,(experts*columns*row_bytes//4,),dtype='uint32')
        with T.Kernel(T.ceildiv(columns,warps),top,threads=threads) as (block,rank):
            tx=T.get_thread_binding()
            lane=tx%32
            column=block*warps+tx//32
            expert=ids[0,rank]
            accum=T.alloc_var('float32');accum=0
            for tile in T.serial(depth//256):
                base=(expert*columns+column)*row_bytes+(tile*8+lane//4)*18
                scale=half_value(words,base)
                packed=word(words,base+2+lane%4*4)
                j=tile*256+lane//4*32+lane%4*4
                lo=T.alloc_local((4,),'float32')
                hi=T.alloc_local((4,),'float32')
                for component in T.vectorized(4):
                    if routed:
                        lo[component]=T.cast(T.cast(x[0,rank,j+component],'float16'),'float32')
                        hi[component]=T.cast(T.cast(x[0,rank,j+16+component],'float16'),'float32')
                    else:
                        lo[component]=T.cast(T.cast(x[0,j+component],'float16'),'float32')
                        hi[component]=T.cast(T.cast(x[0,j+16+component],'float16'),'float32')
                for component in T.unroll(4):
                    coefficient=T.cast(T.cast(scale*(T.cast(packed >> (component*8)&15,'float32')-8),'float16'),'float32')
                    accum+=lo[component]*coefficient
                for component in T.unroll(4):
                    coefficient=T.cast(T.cast(scale*(T.cast(packed >> (component*8+4)&15,'float32')-8),'float16'),'float32')
                    accum+=hi[component]*coefficient
            for step in T.unroll(5):
                accum+=T.shfl_down(accum,16>>step)
            if lane==0 and column<columns:
                out[0,rank,column]=T.if_then_else(counts[expert]>0,accum,0)
        return out
    return compute
