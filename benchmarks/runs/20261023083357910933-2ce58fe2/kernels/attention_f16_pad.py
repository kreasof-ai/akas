"""Prepare only the unwritten tail read by four 64-key async tiles."""
from akas.kernels.kernel import Tensor,kernel

def factory(spec,schedule):
    import tilelang.language as T
    p={**spec.parameters,**schedule}
    slots,heads,capacity,depth=(p[n] for n in ('slots','heads','capacity','d'))
    @kernel
    def compute(kc:Tensor((slots,heads,capacity,depth),'float16'),
                vc:Tensor((slots,heads,capacity,depth),'float16'),
                lengths:Tensor((slots,),'int32')) -> tuple[Tensor((slots,heads,capacity,depth),'float16'),Tensor((slots,heads,capacity,depth),'float16')]:
        with T.Kernel(heads,slots,threads=128) as (head,slot):
            live=T.min(T.max(lengths[slot],0),capacity)
            for index in T.Parallel(256*depth):
                position=live+index//depth
                if position<T.ceildiv(live,256)*256 and position<capacity:
                    kc[slot,head,position,index%depth]=0
                    vc[slot,head,position,index%depth]=0
        return kc,vc
    return compute
