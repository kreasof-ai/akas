from akas.kernels.kernel import Tensor,kernel,output
def factory(spec, schedule):
    """Grouped tensor-core projection dispatched through compact expert tiles."""
    import tilelang.language as T
    from akas.quantization.weights import bind_parts

    p = {**spec.parameters, **schedule}
    rows, top, experts, columns, depth = (
        p[n] for n in ("r", "top", "experts", "o", "k")
    )
    bm, bn, bk = p.get("block_m", 16), p.get("columns", 64), p.get("block_k", 32)
    if bm not in (16, 32, 64, 128) or bn not in (32, 64, 128) or bk not in (32, 64, 128):
        raise ValueError("invalid compact CUDA expert tile")
    capacity = (rows * top + 3) // 4 + experts
    map_block = min(bm, 64)
    pairs = bm // map_block
    groups = (rows * top + map_block - 1) // map_block + experts
    layout = map_block.bit_length() - 3
    binding = bind_parts(p["weight_parts"], depth, provider="cuda")
    storage, read = binding.storage.values, binding.read
    from q4_word_tile import tile_reader
    load_rhs=tile_reader(experts*columns,depth)
    routed = p.get("routed_input", False)
    shape = (rows * top, depth)
    input_dtype = p.get("input_dtype", "float32")

    @T.macro
    def activation(x, route, channel):
        if routed:
            return x[route // top, route % top, channel]
        else:
            return x[route // top, channel]

    @kernel
    def compute(
        x: Tensor(shape, "float16"),
        w: Tensor(storage.shape, storage.dtype),
        counts: Tensor((experts,), "int32"),
        offsets: Tensor((experts+1,), "int32"),
        routes: Tensor((experts, rows), "int32"),
        tile_experts: Tensor((5, capacity), "int32"),
        tile_rows: Tensor((5, capacity), "int32"),
    ) -> Tensor((rows, top, columns), "float32"):
        out = output((rows, top, columns), "float32")
        with T.Kernel(T.ceildiv(columns, bn), groups, threads=p.get("threads",128)) as (bx, group):
            expert = tile_experts[layout, group]
            mapped_row = tile_rows[layout, group]
            by = mapped_row // pairs
            if expert >= 0 and mapped_row % pairs == 0 and by * bm < counts[expert]:
                lhs = T.alloc_shared((bm, bk), "float16")
                rhs = T.alloc_shared((bn, bk), "float16")
                accum = T.alloc_fragment((bm, bn), "float32")
                T.clear(accum)
                for tile in T.Pipelined(T.ceildiv(depth, bk), num_stages=p.get("stages",2)):
                    T.copy(x[offsets[expert]+by*bm,tile*bk],lhs)
                    load_rhs((w,),rhs,expert*columns+bx*bn,tile*bk)
                    T.gemm(lhs, rhs, accum, transpose_B=True)
                for i, j in T.Parallel(bm, bn):
                    index = by * bm + i
                    if index < counts[expert] and bx * bn + j < columns:
                        route = routes[expert, index]
                        out[route // top, route % top, bx * bn + j] = accum[i, j]
        return out

    return compute

