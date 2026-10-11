"""Operation kernels for sequence.cuda.bf16_batch."""


def make_kernel(kind, p):
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = p.get("slots", r), p.get("chunk", 1), p.get("pool", 256)
    rows = slots * chunk
    if kind == "gather_hidden":
        c = p["c"]

        @T.prim_func
        def kernel(
            saved: T.Tensor((pool, c), "bfloat16"),
            mapping: T.Tensor((slots,), "int32"),
            mode: T.Tensor((1,), "int32"),
            out: T.Tensor((rows, c), "bfloat16"),
        ):
            with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
                if (mode[0] != 0) & (mapping[batch] >= 0):
                    for j in T.Parallel(256):
                        col = tile * 256 + j
                        if col < c:
                            out[batch * chunk, col] = saved[mapping[batch], col]

        return kernel
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = p.get("slots", r), p.get("chunk", 1), p.get("pool", 256)
    if kind == "store_hidden":
        c = p["c"]

        @T.prim_func
        def kernel(
            x: T.Tensor((slots, c), "bfloat16"),
            mapping: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            saved: T.Tensor((pool, c), "bfloat16"),
        ):
            with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
                if (mapping[batch] >= 0) & (lengths[batch] > 0):
                    for j in T.Parallel(256):
                        col = tile * 256 + j
                        if col < c:
                            saved[mapping[batch], col] = x[batch, col]

        return kernel
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = p.get("slots", r), p.get("chunk", 1), p.get("pool", 256)
    rows = slots * chunk
    if kind == "controls":

        @T.prim_func
        def kernel(
            position: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            flat_position: T.Tensor((rows,), "int32"),
            active: T.Tensor((rows,), "int32"),
        ):
            with T.Kernel(T.ceildiv(rows, 256), threads=256) as block:
                for i in T.Parallel(256):
                    row = block * 256 + i
                    if row < rows:
                        flat_position[row] = position[row // chunk] + row % chunk
                        active[row] = T.cast(
                            row % chunk < lengths[row // chunk], "int32"
                        )

        return kernel
    import tilelang.language as T

    r = p.get("r", 1)
    slots, chunk, pool = p.get("slots", r), p.get("chunk", 1), p.get("pool", 256)
    rows = slots * chunk
    if kind == "last_rows":
        c = p["c"]

        @T.prim_func
        def kernel(
            x: T.Tensor((rows, c), "bfloat16"),
            lengths: T.Tensor((slots,), "int32"),
            out: T.Tensor((slots, c), "bfloat16"),
        ):
            with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        out[batch, col] = T.if_then_else(
                            lengths[batch] > 0,
                            x[batch * chunk + T.max(lengths[batch] - 1, 0), col],
                            0,
                        )

        return kernel
    raise ValueError("unsupported operation: " + kind)
