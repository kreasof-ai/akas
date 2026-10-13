"""Uniform request-row kernels; no model or weight-format dispatch."""


def make_kernel(kind, p):
    if kind != "clear_rows":
        raise ValueError("unsupported row operation: "+kind)
    import tilelang.language as T

    rows,pool,width = p["rows"],p["pool"],p["width"]
    dtype,threads = p["dtype"],p.get("threads",256)
    if any(type(n) is not int or n < 1 for n in (rows,pool,width,threads)):
        raise ValueError("row geometry must contain positive integers")
    @T.prim_func
    def kernel(mapping:T.Tensor((rows,),"int32"), state:T.Tensor((pool,width),dtype)):
        with T.Kernel(rows,T.ceildiv(width,threads),threads=threads) as (row,tile):
            request = mapping[row]
            if (request >= 0) & (request < pool):
                for i in T.Parallel(threads):
                    col = tile*threads+i
                    if col < width:
                        state[request,col] = 0
    return kernel
