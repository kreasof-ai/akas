from akas.kernels.kernel import Tensor,kernel,output
def factory(spec, schedule):
    """Warp-local expert GEMV over a codec's packed FP16 dot capability."""
    import tilelang.language as T
    from akas.quantization.weights import bind_parts

    p = {**spec.parameters, **schedule}
    rows, top, experts, columns, depth = (
        p[n] for n in ("r", "top", "experts", "o", "k")
    )
    threads = p.get("threads", 128)
    if rows != 1 or depth % 256 or threads not in (128, 256):
        raise ValueError("unsupported packed expert decode geometry")
    binding = bind_parts(p["weight_parts"], depth, provider="cuda")
    storage, dot = binding.storage.values, binding.require("dot_half")
    from planar_q4 import dot_half
    dot = dot_half((experts*columns,depth))
    routed = p.get("routed_input", False)
    shape = (1, top, depth) if routed else (1, depth)
    dtype = p.get("input_dtype", "bfloat16")
    warps = threads // 32

    @kernel
    def compute(
        x: Tensor(shape, dtype),
        w: Tensor(storage.shape, storage.dtype),
        ids: Tensor((1, top), "int32"),
        counts: Tensor((experts,), "int32"),
    ) -> Tensor((1, top, columns), "float32"):
        out = output((1, top, columns), "float32")
        flat = T.view(x, (top * depth if routed else depth,), dtype=dtype)
        with T.Kernel(T.ceildiv(columns, warps), top, threads=threads) as (block, rank):
            tx = T.get_thread_binding()
            lane, column = tx % 32, block * warps + tx // 32
            expert = ids[0, rank]
            accum = T.alloc_var("float32")
            accum = 0
            if column < columns:
                for tile in T.serial(depth // 256):
                    accum = dot(
                        (w,),
                        flat,
                        rank * depth if routed else 0,
                        tile,
                        lane,
                        expert * columns + column,
                        accum,
                    )
            for step in T.unroll(5):
                accum += T.shfl_down(accum, 16 >> step)
            if lane == 0 and column < columns:
                out[0, rank, column] = T.if_then_else(counts[expert] > 0, accum, 0)
        return out

    return compute

