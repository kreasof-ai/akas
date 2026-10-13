"""Operation kernels for sequence.cuda.prefill."""


def make_kernel(kind, p):
    import tilelang.language as T

    slots, chunk = p["slots"], p["chunk"]
    rows = slots * chunk
    if kind == "controls":

        @T.prim_func
        def kernel(
            position: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            flat_position: T.Tensor((rows,), "int32"),
            active: T.Tensor((rows,), "int32"),
        ):
            with T.Kernel(T.ceildiv(rows, 256), threads=256) as tile:
                for j in T.Parallel(256):
                    row = tile * 256 + j
                    if row < rows:
                        flat_position[row] = position[row // chunk] + row % chunk
                        active[row] = T.cast(
                            row % chunk < lengths[row // chunk], "int32"
                        )

        return kernel
    import tilelang.language as T

    slots, chunk = p["slots"], p["chunk"]
    rows = slots * chunk
    if kind == "last_rows":
        width = p["width"]

        @T.prim_func
        def kernel(
            x: T.Tensor((rows, width), "bfloat16"),
            lengths: T.Tensor((slots,), "int32"),
            out: T.Tensor((slots, width), "bfloat16"),
        ):
            with T.Kernel(slots, T.ceildiv(width, 256), threads=256) as (slot, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < width:
                        out[slot, col] = T.if_then_else(
                            lengths[slot] > 0,
                            x[slot * chunk + T.max(lengths[slot] - 1, 0), col],
                            0,
                        )

        return kernel
    import tilelang.language as T

    slots, chunk = p["slots"], p["chunk"]
    if kind == "advance":
        all_tokens = p.get("all_tokens", False)

        @T.prim_func
        def kernel(
            position: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            active: T.Tensor((slots,), "int32"),
        ):
            with T.Kernel(1, threads=32):
                for slot in T.Parallel(slots):
                    # Decode argmax already advances one position for each
                    # active slot; prefill accounts for the remaining rows.
                    active[slot] = T.cast(lengths[slot] > 0, "int32")
                    position[slot] += T.max(lengths[slot] - (0 if all_tokens else 1), 0)

        return kernel
    raise ValueError("unsupported operation: " + kind)
