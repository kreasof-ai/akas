"""Prototype codec-backed small grouped expert projection; no model constants."""
from akas.kernels.kernel import Tensor, kernel, output


def grouped(spec, schedule):
    import tilelang.language as T
    from akas.quantization.weights import bind_parts

    p={**spec.parameters, **schedule}
    rows,top,experts,columns,depth=(p[n] for n in ('r','top','experts','o','k'))
    if not 1 < rows <= 16 or depth % 256:
        raise ValueError('Small grouped packed projection needs 2-16 rows and block-256 depth')
    binding=bind_parts(p['weight_parts'], depth, provider='cuda')
    storage,dot=binding.storage.values,binding.require('dot_half')
    routed=p.get('routed_input',False)
    shape=(rows,top,depth) if routed else (rows,depth)
    dtype=p.get('input_dtype','float32')
    threads=p.get('threads',128); width=p.get('columns_per_warp',1)
    if threads not in (128,256) or width not in (1,2,4):raise ValueError('Invalid warp geometry')
    warps=threads//32
    capacity=(rows*top+3)//4+experts
    groups=min(rows*top,experts)

    @kernel
    def compute(
        x: Tensor(shape,dtype),
        w: Tensor(storage.shape,storage.dtype),
        counts: Tensor((experts,),'int32'),
        routes: Tensor((experts,rows),'int32'),
        tile_experts: Tensor((5,capacity),'int32'),
        tile_rows: Tensor((5,capacity),'int32'),
    ) -> Tensor((rows,top,columns),'float32'):
        out=output((rows,top,columns),'float32')
        flat=T.view(x,(rows*top*depth if routed else rows*depth,),dtype=dtype)
        with T.Kernel(T.ceildiv(columns,warps*width),groups,threads=threads) as (bx,group):
            tx=T.get_thread_binding();lane=tx%32;warp=tx//32
            expert=tile_experts[2,group]
            if expert>=0:
                for offset in T.serial(counts[expert]-tile_rows[2,group]*16):
                    index=tile_rows[2,group]*16+offset
                    route=routes[expert,index]
                    for part in T.unroll(width):
                        column=bx*warps*width+warp+part*warps
                        accum=T.alloc_var('float32');accum=0
                        if column<columns:
                            for tile in T.serial(depth//256):
                                accum=dot((w,),flat,
                                          (route if routed else route//top)*depth,
                                          tile,lane,expert*columns+column,accum)
                        for step in T.unroll(5):
                            accum+=T.shfl_down(accum,16>>step)
                        if lane==0 and column<columns:
                            out[route//top,route%top,column]=accum
        return out
    return compute


def make(spec,schedule):
    from akas.kernels.definition import KernelSpec
    return grouped(KernelSpec(**spec),schedule)
