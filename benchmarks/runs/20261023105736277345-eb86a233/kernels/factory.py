"""Partitioned native F16 attention sharing KV across block queries and GQA heads."""
from akas.kernels.kernel import Tensor, kernel, output


def factory(spec, schedule):
    import tilelang.language as T

    p = {**spec.parameters, **schedule}
    rows, heads, kv_heads, depth, capacity, chunk = (p[n] for n in ('r','h','kh','d','capacity','chunk'))
    splits, tile, stages = p.get('splits',64), p.get('key_rows',64), p.get('stages',1)
    slots, group = rows//chunk, heads//kv_heads
    query_rows = 16 if chunk*group<=16 else 32 if chunk*group<=32 else 64
    if depth!=64 or rows%chunk or heads%kv_heads or chunk*group>64:
        raise ValueError('Unsupported block GQA tile')
    shape=(slots,kv_heads,capacity,depth)
    scale_shape=(slots,kv_heads,capacity,1)

    @kernel
    def compute(q:Tensor((rows,heads*depth),'float32'),
                kc:Tensor(shape,'float16'),vc:Tensor(shape,'float16'),
                ks:Tensor(scale_shape,'float32'),vs:Tensor(scale_shape,'float32'),
                positions:Tensor((rows,),'int32'),enabled:Tensor((rows,),'int32')) -> Tensor((rows*heads*splits*66,),'float32'):
        out=output((rows*heads*splits*66,),'float32',position=5)
        with T.Kernel(kv_heads,slots,splits,threads=128) as (head,slot,part):
            limits=T.alloc_fragment((query_rows,),'int32')
            longest=T.alloc_fragment((1,),'int32')
            for i in T.Parallel(query_rows):
                row=slot*chunk+T.min(i//group,chunk-1)
                limits[i]=T.if_then_else(i<chunk*group and enabled[row]!=0,
                                        T.min(positions[row]+1,capacity),0)
            T.reduce_max(limits,longest,dim=0)
            length=longest[0]
            span=T.ceildiv(T.ceildiv(length,tile),splits)
            tiles=T.min(span,T.max(T.ceildiv(length,tile)-part*span,0))
            query=T.alloc_shared((query_rows,depth),'float16')
            key=T.alloc_shared((tile,depth),'float16')
            value=T.alloc_shared((tile,depth),'float16')
            probability=T.alloc_shared((query_rows,tile),'float16')
            scores=T.alloc_fragment((query_rows,tile),'float32')
            result=T.alloc_fragment((query_rows,depth),'float32')
            maximum=T.alloc_fragment((query_rows,),'float32')
            previous=T.alloc_fragment((query_rows,),'float32')
            factor=T.alloc_fragment((query_rows,),'float32')
            normalizer=T.alloc_fragment((query_rows,),'float32')
            total=T.alloc_fragment((query_rows,),'float32')
            for i,j in T.Parallel(query_rows,depth):
                row=slot*chunk+T.min(i//group,chunk-1)
                query[i,j]=T.if_then_else(i<chunk*group and enabled[row]!=0,
                    q[row,(head*group+i%group)*depth+j],0)
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum,-T.infinity('float32'))
            for step in T.Pipelined(tiles,num_stages=stages):
                start=(part*span+step)*tile
                for i,j in T.Parallel(tile,depth):
                    key[i,j]=T.if_then_else(start+i<length,kc[slot,head,T.min(start+i,capacity-1),j],0)
                    value[i,j]=T.if_then_else(start+i<length,vc[slot,head,T.min(start+i,capacity-1),j],0)
                T.gemm(query,key,scores,transpose_B=True,clear_accum=True)
                T.copy(maximum,previous)
                for i,j in T.Parallel(query_rows,tile):
                    scores[i,j]=T.if_then_else(start+j<limits[i],scores[i,j]*.125,-T.infinity('float32'))
                T.reduce_max(scores,maximum,dim=1,clear=False)
                for i in T.Parallel(query_rows):
                    factor[i]=T.if_then_else(T.isfinite(previous[i]),T.exp(previous[i]-maximum[i]),0)
                for i,j in T.Parallel(query_rows,tile):
                    scores[i,j]=T.if_then_else(start+j<limits[i],T.exp(scores[i,j]-maximum[i]),0)
                T.reduce_sum(scores,total,dim=1)
                for i in T.Parallel(query_rows):
                    normalizer[i]=normalizer[i]*factor[i]+total[i]
                for i,j in T.Parallel(query_rows,depth):
                    result[i,j]*=factor[i]
                T.copy(scores,probability)
                T.gemm(probability,value,result)
            for i,j in T.Parallel(query_rows,depth):
                if i<chunk*group:
                    row=slot*chunk+i//group
                    index=((row*heads+head*group+i%group)*splits+part)*66
                    out[index+j]=result[i,j]
            for i in T.Parallel(query_rows):
                if i<chunk*group:
                    row=slot*chunk+i//group
                    index=((row*heads+head*group+i%group)*splits+part)*66
                    out[index+64]=maximum[i]
                    out[index+65]=normalizer[i]
        return out
    return compute
