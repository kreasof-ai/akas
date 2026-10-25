"""Native-half GQA split attention prototype with tensor-core QK/PV."""
from akas.kernels.kernel import Tensor,kernel,output

def factory(p):
    import tilelang.language as T
    r,h,kh,cap=(p[n] for n in ('r','h','kh','capacity'))
    splits=p.get('splits',16);bk=p.get('key_rows',64);stages=p.get('stages',2)
    group=h//kh;qm=16;d=64
    @kernel
    def compute(q:Tensor((r,h*d),'float32'),kc:Tensor((r,kh,cap,d),'float16'),vc:Tensor((r,kh,cap,d),'float16'),ks:Tensor((r,kh,cap,1),'float32'),vs:Tensor((r,kh,cap,1),'float32'),positions:Tensor(r,'int32'),enabled:Tensor(r,'int32'))->Tensor(r*h*splits*66,'float32'):
        out=output(r*h*splits*66,'float32',position=5)
        with T.Kernel(r*kh,splits,threads=128) as (index,split):
            slot=index//kh;head=index%kh
            length=T.if_then_else(enabled[slot]!=0,T.min(positions[slot]+1,cap),0)
            span=T.ceildiv(T.ceildiv(length,bk),splits)
            query=T.alloc_shared((qm,d),'float16')
            key=T.alloc_shared((bk,d),'float16')
            value=T.alloc_shared((bk,d),'float16')
            prob=T.alloc_shared((qm,bk),'float16')
            scores=T.alloc_fragment((qm,bk),'float32')
            result=T.alloc_fragment((qm,d),'float32')
            maximum=T.alloc_fragment((qm,),'float32')
            previous=T.alloc_fragment((qm,),'float32')
            factor=T.alloc_fragment((qm,),'float32')
            normalizer=T.alloc_fragment((qm,),'float32')
            total=T.alloc_fragment((qm,),'float32')
            for i,j in T.Parallel(qm,d):
                query[i,j]=T.if_then_else(i<group,q[slot,(head*group+T.min(i,group-1))*d+j]*ks[slot,head,0,0],0)
            T.clear(result);T.clear(normalizer);T.fill(maximum,-T.infinity('float32'))
            tiles=T.min(span,T.max(T.ceildiv(length,bk)-split*span,0))
            for block in T.serial(T.ceildiv(tiles,4)):
                for local in T.Pipelined(4,num_stages=stages):
                    tile=block*4+local
                    start=(split*span+tile)*bk
                    T.copy(kc[slot,head,start,0],key)
                    T.copy(vc[slot,head,start,0],value)
                    T.gemm(query,key,scores,transpose_B=True,clear_accum=True)
                    T.copy(maximum,previous)
                    for i,j in T.Parallel(qm,bk):
                        scores[i,j]=T.if_then_else(tile<tiles and start+j<length,scores[i,j]*0.125,-T.infinity('float32'))
                    T.reduce_max(scores,maximum,dim=1,clear=False)
                    for i in T.Parallel(qm):
                        factor[i]=T.if_then_else(T.isfinite(previous[i]),T.exp(previous[i]-maximum[i]),0)
                    for i,j in T.Parallel(qm,bk):
                        scores[i,j]=T.if_then_else(tile<tiles and start+j<length,T.exp(scores[i,j]-maximum[i]),0)
                    T.reduce_sum(scores,total,dim=1)
                    for i in T.Parallel(qm):normalizer[i]=normalizer[i]*factor[i]+total[i]
                    for i,j in T.Parallel(qm,d):result[i,j]*=factor[i]
                    T.copy(scores,prob)
                    T.gemm(prob,value,result)
            for i,j in T.Parallel(qm,d):
                if i<group:
                    out[((slot*h+head*group+i)*splits+split)*66+j]=result[i,j]*vs[slot,head,0,0]
            for i in T.Parallel(qm):
                if i<group:
                    out[((slot*h+head*group+i)*splits+split)*66+64]=maximum[i]
                    out[((slot*h+head*group+i)*splits+split)*66+65]=normalizer[i]
        return out
    return compute
