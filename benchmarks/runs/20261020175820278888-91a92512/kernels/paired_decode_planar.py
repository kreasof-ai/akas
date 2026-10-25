"""Decode gate/up/SwiGLU fusion using the codec's existing FP16 dot contract."""
from akas.kernels.kernel import Tensor, kernel, output


def factory(p):
    import tilelang.language as T
    from akas.quantization.weights import bind_parts

    depth, columns, experts, top = (p[n] for n in ("k", "o", "experts", "top"))
    binding = bind_parts(p["weight_parts"], depth, provider="cuda")
    storage = binding.storage.values
    from planar_q4 import dot_half
    dot = dot_half((experts*columns,depth))
    threads, width, unroll = (p.get(n, default) for n, default in (("threads",128),("width",1),("unroll",1)))
    warps = threads // 32

    @kernel
    def compute(
        x: Tensor((1,depth),"float32"),
        w: Tensor(storage.shape,storage.dtype),
        w2: Tensor(storage.shape,storage.dtype),
        ids: Tensor((1,top),"int32"),
        counts: Tensor((experts,),"int32"),
    ) -> Tensor((1,top,columns),"float32"):
        out = output((1,top,columns),"float32")
        flat = T.meta_var(T.view(x,(depth,),dtype="float32"))
        with T.Kernel(T.ceildiv(columns,warps*width),top,threads=threads) as (block,rank):
            tx = T.get_thread_binding()
            warp,lane = tx//32,tx%32
            expert = ids[0,rank]
            g = T.alloc_local((width,),"float32")
            u = T.alloc_local((width,),"float32")
            for col in T.unroll(width):
                g[col] = 0
                u[col] = 0
            for tile in T.serial(T.ceildiv(depth//256,unroll)):
                for step in T.unroll(unroll):
                    for col in T.unroll(width):
                        column = (block*warps+warp)*width+col
                        if column < columns and tile*unroll+step < depth//256:
                            g[col] = dot((w,),flat,0,tile*unroll+step,lane,expert*columns+column,g[col])
                            u[col] = dot((w2,),flat,0,tile*unroll+step,lane,expert*columns+column,u[col])
            for col in T.unroll(width):
                for step in T.unroll(5):
                    g[col] += T.shfl_down(g[col],16>>step)
                    u[col] += T.shfl_down(u[col],16>>step)
                column = (block*warps+warp)*width+col
                if lane == 0 and column < columns:
                    out[0,rank,column] = T.if_then_else(counts[expert]>0,g[col]/(1+T.exp(-g[col]))*u[col],0)
        return out

    return compute


def evict_factory(p):
    """Write a footprint exceeding L2 before the timed region; not timed work."""
    import tilelang.language as T
    elements=p.get("elements",128*1024*1024//4)

    @kernel
    def compute() -> Tensor((elements,),"int32"):
        out=output((elements,),"int32")
        with T.Kernel(T.ceildiv(elements,1024),threads=256) as block:
            for lane in T.Parallel(1024):
                i=block*1024+lane
                if i<elements:
                    out[i]=i
        return out

    return compute
